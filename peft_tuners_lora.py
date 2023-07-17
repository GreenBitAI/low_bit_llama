import math
import re
import torch
import warnings

from peft.tuners import lora
from peft.tuners.lora import Linear, LoraLayer
from peft.utils import _get_submodules, PeftType
from torch import nn
from transformers.pytorch_utils import Conv1D

from model import QuantLinear


class LinearLowbitLt(QuantLinear, LoraLayer):

        # Lora implemented in a dense layer
        def __init__(
            self,
            adapter_name,
            in_features,
            out_features,
            groupsize: int = -1,
            bits = 4,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            **kwargs,
        ):
            QuantLinear.__init__(
                self,
                in_features,
                out_features,
                groupsize,
                bits,
            )
            LoraLayer.__init__(self, in_features=in_features, out_features=out_features)

            # Freezing the pre-trained weight matrix
            self.qweight.requires_grad = False
            self.qscales.requires_grad = False
            self.qscales_scales.requires_grad = False
            self.qscales_zeros.requires_grad= False
            
            self.qzeros.requires_grad = False
            self.g_idx.requires_grad = False
            self.bias.requires_grad = False

            init_lora_weights = kwargs.pop("init_lora_weights", True)
            self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
            self.active_adapter = adapter_name

        def forward(self, x: torch.Tensor):
            result = super().forward(x)
            
            if self.disable_adapters or self.active_adapter not in self.lora_A.keys():
                return result
            elif self.r[self.active_adapter] > 0:
                if not torch.is_autocast_enabled():
                    expected_dtype = result.dtype

                    if x.dtype != torch.float32:
                        x = x.float()
                    output = (
                        self.lora_B[self.active_adapter](
                            self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
                        ).to(expected_dtype)
                        * self.scaling[self.active_adapter]
                    )
                else:
                    output = (
                        self.lora_B[self.active_adapter](
                            self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
                        )
                        * self.scaling[self.active_adapter]
                    )
                result += output
            
            return result
        
        @property
        def weight(self):
            class WeightDeviceClass:
                device = self.qweight.device
            return WeightDeviceClass()


class LoraModel(lora.LoraModel):
    def _find_and_replace(self, adapter_name):
        lora_config = self.peft_config[adapter_name]
        is_target_modules_in_base_model = False
        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
        }
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(lora_config.target_modules, str):
                target_module_found = re.fullmatch(lora_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in lora_config.target_modules)
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = _get_submodules(self.model, key)
                bias = target.bias is not None
                if isinstance(target, LoraLayer):
                    target.update_layer(
                        adapter_name,
                        lora_config.r,
                        lora_config.lora_alpha,
                        lora_config.lora_dropout,
                        lora_config.init_lora_weights,
                    )
                else:
                    if isinstance(target, QuantLinear):
                        new_module = LinearLowbitLt(adapter_name, target.in_features, target.out_features, target.groupsize, target.bits, bias=bias, **kwargs)
                    else:
                        if isinstance(target, torch.nn.Linear):
                            in_features, out_features = target.in_features, target.out_features
                            if kwargs["fan_in_fan_out"]:
                                warnings.warn(
                                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                                    "Setting fan_in_fan_out to False."
                                )
                                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
                        elif isinstance(target, Conv1D):
                            in_features, out_features = (
                                target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                            )
                            if not kwargs["fan_in_fan_out"]:
                                warnings.warn(
                                    "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                                    "Setting fan_in_fan_out to True."
                                )
                                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
                        else:
                            raise ValueError(
                                f"Target module {target} is not supported. "
                                f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
                            )
                        new_module = Linear(adapter_name, in_features, out_features, bias=bias, **kwargs)

                    self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {lora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        if isinstance(old_module, QuantLinear) and isinstance(new_module, LinearLowbitLt):
            new_module.qweight = old_module.qweight
            
            new_module.qscales = old_module.qscales
            new_module.qscales_scales = old_module.qscales_scales
            new_module.qscales_zeros = old_module.qscales_zeros

            new_module.qzeros = old_module.qzeros
            new_module.g_idx = old_module.g_idx
            new_module.bias = old_module.bias
            
            if getattr(old_module, "state", None) is not None:
                new_module.state = old_module.state
                new_module.to(old_module.qweight.device)

            # dispatch to correct device
            for name, module in new_module.named_modules():
                if "lora_" in name:
                    module.to(old_module.qweight.device)
        else:
            new_module.weight = old_module.weight
            if old_module.bias is not None:
                new_module.bias = old_module.bias
            if getattr(old_module, "state", None) is not None:
                new_module.state = old_module.state
                new_module.to(old_module.weight.device)

            # dispatch to correct device
            for name, module in new_module.named_modules():
                if "lora_" in name:
                    module.to(old_module.weight.device)


def replace_peft_model_with_lora_model():
    import peft.peft_model
    peft.peft_model.PEFT_TYPE_TO_MODEL_MAPPING[PeftType.LORA] = LoraModel
