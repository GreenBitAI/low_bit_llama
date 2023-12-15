# GreenBit LLaMA

This is GreenBitAI's research code for running **low-bit (4/2/1)** LLaMA families with extreme compression yet still strong performance, the quantized models are available on the [model zoo](https://huggingface.co/GreenBitAI?sort_models=downloads#models).

This is meant to be a research demo for the quality of the model.
There is no speed-up implemented yet.

## Roadmap

Over the next few months, we will continue offering 2-bit and 1-bit versions of LLaMA models.
Additionally, we are considering the provision of low-bit versions for other open-source LLMs in the future.

## Latest Updates
[14/12/2023] We are happy to release the lossless (<1%) W4A16 01-Yi 6/34B models. The strong 2-bit version will also be made open soon.

## Few Shot Evaluation (officially evaluated by 01-Yi)
| Model          | Yi-34B FP16| [Yi-34B 4 bit](https://huggingface.co/GreenBitAI/yi-34b-w4a16g32) | Yi-6B FP16 | [Yi-6B 4 bit](https://huggingface.co/GreenBitAI/yi-6b-w4a16g32) |
|----------------|-----------|----------|----------|---------|
| GroupSize      | -         | 32       | -        | 8       |
| Model Size (GB)| 68.79     | 19.89    | 12.12    | 4.04    |
| AVG            | 70.64     | 69.7     | 60.11    | 59.14   |
| **Detailed Evaluation** | | | | |
| MMLU           | 76.32     | 75.42    | 63.24    | 62.09   |
| CMMLU          | 83.65     | 83.07    | 75.53    | 72.85   |
| ARC-e          | 84.42     | 84.13    | 77.23    | 76.52   |
| ARC-c          | 61.77     | 59.56    | 50.34    | 48.47   |
| GAOKAO         | 82.8      | 81.37    | 72.2     | 72.87   |
| GSM8K          | 67.24     | 63.61    | 32.52    | 28.05   |
| HumanEval      | 25.6      | 25       | 15.85    | 15.85   |
| BBH            | 54.3      | 52.3     | 42.8     | 41.47   |
| WinoGrande     | 78.68     | 78.53    | 70.63    | 71.19   |
| PIQA           | 82.86     | 82.75    | 78.56    | 79.05   |
| SIQA           | 74.46     | 73.44    | 64.53    | 64.53   |
| HellaSwag      | 83.64     | 83.02    | 74.91    | 73.27   |
| OBQA           | 91.6      | 90.8     | 85.4     | 82.6    |
| CSQA           | 83.37     | 83.05    | 76.9     | 75.43   |
| TriviaQA       | 81.52     | 80.73    | 64.85    | 61.75   |
| SquAD          | 92.46     | 91.12    | 88.95    | 88.39   |
| BoolQ          | 88.25     | 88.17    | 76.23    | 77.1    |
| MBPP           | 41        | 39.68    | 26.32    | 25.13   |
| QUAC           | 48.61     | 47.43    | 40.92    | 40.16   |
| Lambda         | 73.18     | 73.39    | 67.74    | 67.8    |
| NaturalQuestion| 27.67     | 27.21    | 16.69    | 17.42   |


# Zero Shot Evaluation
| Task          | Metric | Yi-6B FP16 | [Yi-6B 4 bit](https://huggingface.co/GreenBitAI/yi-6b-w4a16g32) | [Yi-34B 4 bit](https://huggingface.co/GreenBitAI/yi-34b-w4a16g32) |
|---------------|--------|---------|-------------|--------------|
| Openbookqa    | acc    | 0.314   | 0.324       | 0.344        |
|               | ac_norm| 0.408   | 0.42        | 0.474        |
| arc_challenge | acc    | 0.462   | 0.4573      | 0.569        |
|               | ac_norm| 0.504   | 0.483       | 0.5964       |
| hellawswag    | acc    | 0.553   | 0.5447      | 0.628        |
|               | ac_norm| 0.749   | 0.7327      | 0.83         |
| piqa          | acc    | 0.777   | 0.7709      | 0.8079       |
|               | ac_norm| 0.787   | 0.7894      | 0.828        |
| arc_easy      | acc    | 0.777   | 0.7697      | 0.835        |
|               | ac_norm| 0.774   | 0.7659      | 0.84         |
| Winogrande    | acc    | 0.707   | 0.7095      | 0.7853       |
| boolq         | acc    | 0.755   | 0.7648      | 0.886        |
| truthfulqa_mc | mc1    | 0.29    | 0.2729      | 0.4026       |
|               | mc2    | 0.419   | 0.4033      | 0.5528       |
| anli_r1       | acc    | 0.423   | 0.416       | 0.554        |
| anli_r2       | acc    | 0.409   | 0.409       | 0.518        |
| anli_r3       | acc    | 0.411   | 0.393       | 0.4983       |
| wic           | acc    | 0.529   | 0.545       | 0.5376       |
| rte           | acc    | 0.685   | 0.7039      | 0.7617       |
| record        | f1     | 0.904   | 0.9011      | 0.924        |
|               | em     | 0.8962  | 0.8927      | 0.916        |
| Average       |        | 0.596   | 0.5937      | 0.6708       |

## Requirements

The inference currently requires a machine with CUDA installed.
Then you can simply run:

```bash
pip install -r requirements.txt
```

## Try the model

Use the environment variable `CUDA_VISIBLE_DEVICES` to select the correct GPU.
Multi-GPU is not supported, but the model is very compressed, so 1 GPU should be enough.
To evaluate the compressed model on the harness, you can use the following commands
in ```scripts/```. Predefined scripts already there:

```bash
bash scripts/harness/yi_6b_w4a16g32_harness.sh         # for zero shot evaluation of the lossless 4-bit 01-yi 6b model
bash scripts/harness/yi_34b_w4a16g32_harness.sh         # for zero shot evaluation of the lossless 4-bit 01-yi 34b model
```

# References

This code is based on:

- [LLaMA Reference Implementation](https://github.com/facebookresearch/llama)
- [GPTQ](https://github.com/IST-DASLab/gptq)
- [GPTQ for LLaMA](https://github.com/qwopqwop200/GPTQ-for-LLaMa)
- [Alpaca_lora_4bit](https://github.com/johnsmith0031/alpaca_lora_4bit)

Thanks to Meta AI for releasing [LLaMA](https://arxiv.org/abs/2302.13971), a powerful LLM.

## Citation
If you use our approach in your research, please cite our work as follows:
```
@article{low_bit_llama,
  title={Advanced Ultra-Low Bitrate Compression Techniques for the LLaMA Family of LLMs},
  author={Guo, Nianhui and Bethge, Joseph and Hu, Ting and Meinel, Christoph and Yang, Haojin},
  journal={https://github.com/GreenBitAI/low_bit_llama},
  year={2023}
}
```

# License

The original code was released under its respective license and copyrights, i.e.:

- `datautils.py` and `evaluate.py`:
[GPTQ for LLaMA](https://github.com/qwopqwop200/GPTQ-for-LLaMa) released under Apache 2.0 License
- `model.py`, `peft_tuners_lora.py` and `inference.py` (basis for `llama_2b_*.py` files):
[Alpaca_lora_4bit](https://github.com/johnsmith0031/alpaca_lora_4bit) released under MIT License

We release our changes and additions to these files under the [Apache 2.0 License](LICENSE).
