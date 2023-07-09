import math
import time

import torch
import torch.nn as nn
from colorama import init, Fore, Style
from huggingface_hub import hf_hub_download

init(autoreset=True)


# Assumes layer is perfectly divisible into 256 * 256 blocks
class QuantLinear(nn.Module):

    def __init__(self, in_features, out_features, groupsize=-1, bits=4):
        super().__init__()
        bits=bits
        self.in_features = in_features
        self.out_features = out_features
        self.bits=bits
        self.maxq = 2 ** self.bits - 1
        groupsize = groupsize if groupsize != -1 else in_features
        self.groupsize = groupsize
    
        self.disable_bias = True
        self.register_buffer("wf", torch.tensor(list(range(0,32,self.bits)), dtype=torch.int32).unsqueeze(0))
        self.initialize(in_features, out_features, groupsize, bits)

    def initialize(self, in_features, out_features, groupsize, bits):
        if bits > 1:
                self.register_buffer('qzeros', torch.empty((math.ceil(in_features/groupsize), out_features // 256 * (bits * 8)), dtype=torch.int32))

        self.register_buffer('qscales', torch.empty(math.ceil(in_features/groupsize), out_features), torch.uint8)
        self.register_buffer('qscales_zeros', torch.empty((1, out_features)))
        self.register_buffer('qscales_scales', torch.empty((1, out_features)))

        self.register_buffer('g_idx', torch.tensor([i // groupsize  for i in range(in_features)], dtype = torch.int32))

        self.register_buffer('bias', torch.empty(out_features))
        self.register_buffer('qweight', torch.empty((in_features // 256 * (bits * 8), out_features), dtype=torch.int32))

    def forward(self, x):
        if self.bits in [2, 4, 8, 16]:
            weight_unpack = torch.bitwise_right_shift(torch.unsqueeze(self.qweight, 1).expand(-1, 32 // self.bits, -1), self.wf.unsqueeze(-1)).to(torch.int16 if self.bits == 8 else torch.int8).view(-1, self.qweight.size(-1))
            torch.bitwise_and(weight_unpack,(2 ** self.bits) - 1, out=weight_unpack)

            zeros_unpack = torch.bitwise_right_shift(torch.unsqueeze(self.qzeros, 2).expand(-1, -1, 32 // self.bits), self.wf.unsqueeze(0)).to(torch.int16 if self.bits == 8 else torch.int8)
            torch.bitwise_and(zeros_unpack, (2 ** self.bits) - 1, out=zeros_unpack)

            zeros_unpack = zeros_unpack + 1
            zeros_unpack = zeros_unpack.reshape(-1, self.out_features)

            weight_unpack = weight_unpack - zeros_unpack[self.g_idx.long()]

        scales = ((self.qscales.to(x.dtype)-self.qscales_zeros)*self.qscales_scales)[self.g_idx.long()]

        weight = (weight_unpack.detach()*scales)
        out = torch.matmul(x, weight)

        if not self.disable_bias:
            out += self.bias
        return out


def make_quant(module, names, name='', groupsize=-1, bits=4):
    if isinstance(module, QuantLinear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            setattr(
                module, attr, QuantLinear(tmp.in_features, tmp.out_features, groupsize=groupsize, bits=bits)
            )
    for name1, child in module.named_children():
        make_quant(child, names, name + '.' + name1 if name != '' else name1, groupsize=groupsize, bits=bits)


def model_to_half(model):
    model.half()
    for n, m in model.named_modules():
        if isinstance(m, QuantLinear):
            m.qscales_scales = m.qscales_scales.half()
            m.qscales_zeros = m.qscales_zeros.half()
            m.bias = m.bias.half()
    print(Style.BRIGHT + Fore.YELLOW + 'Converted as Half.')


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def load_llama_model(model_uri, cache_dir, groupsize=-1, bits=4, half=False, device_map="auto", seqlen=2048):
    import accelerate
    from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

    print(Style.BRIGHT + Fore.CYAN + "Loading Model ...")
    t0 = time.time()

    with accelerate.init_empty_weights():
        config = LlamaConfig.from_pretrained(model_uri, cache_dir=cache_dir)
        model = LlamaForCausalLM(config)
        model = model.eval()
        layers = find_layers(model)
        for name in ['lm_head']:
            if name in layers:
                del layers[name]
        make_quant(model, layers, groupsize=groupsize, bits=bits)

    model = accelerate.load_checkpoint_and_dispatch(
        model=model,
        checkpoint=hf_hub_download(repo_id=model_uri, filename="pytorch_model.bin", cache_dir=cache_dir),
        device_map=device_map,
        no_split_module_classes=["LlamaDecoderLayer"]
    )

    model.seqlen = seqlen

    if half:
        model_to_half(model)

    tokenizer = LlamaTokenizer.from_pretrained(model_uri, cache_dir=cache_dir)
    tokenizer.truncation_side = 'left'

    print(Style.BRIGHT + Fore.GREEN + f"Loaded the model in {(time.time()-t0):.2f} seconds.")

    return model, tokenizer
