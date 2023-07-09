# GreenBit LLaMA

This is GreenBitAI's research code for running **1-bit** and **2-bit** LLaMA models with extreme compression yet still strong performance.

This is meant to be a research demo for the quality of the model.
There is no speed-up implemented yet.

## Roadmap

Over the next few weeks, we will continue to offer both 2-bit and 1-bit versions of LLaMA models.
Additionally, we are considering the provision of low-bit versions for other open-source LLMs in the future.

## Results

The model size of LLaMA-7B shrinks from 12.5 GB (FP16) to 2.2GB (2 bits).

<!--<details>
<summary>LLaMA-7B(click me)</summary>-->

| [LLaMA-7B](https://arxiv.org/abs/2302.13971) | Bits | Wikitext2 | C4   | PTB  | checkpoint size (GiB) |
|----------------------------------------------|------|-----------|------|------|-----------------------|
| FP16                                         | 16   | 5.67      | 7.07 | 8.80 | 12.5                  |
| [GPTQ](https://arxiv.org/abs/2210.17323)     | 4    | 5.85      | 7.21 | 9.00 | 3.7                   |
| **Ours**                                     | 2    | 7.65      | 9.04 | 11.47| 2.2                   |

<!--</details>-->

## Requirements

The inference currently requires a machine with CUDA (tested with 11.7) installed.
Then you can simply run:

```bash
pip install -r requirements.txt
```

## Run inference or evaluation

Use the environment variable `CUDA_VISIBLE_DEVICES` to select the correct GPU.
Multi-GPU is not supported, but the model is very compressed, so 1 GPU should be enough.

```bash
CUDA_VISIBLE_DEVICES=0 llama_2b_evaluate.py
CUDA_VISIBLE_DEVICES=0 llama_2b_inference.py
```

## Examples

Model output from prompt "The meaning of life is":

- Output 1:

    The meaning of life is to love and appreciate it.
    The best things in our lives are the people we meet, those who make a difference or impact on us either positively or negatively, but one thing for sure is that these experiences help you grow as an individual both personally & professionally…..so don’t forget to live your dreams!

- Output 2:

    The meaning of life is to discover one’s true purpose.
    Life can be seen as a journey, and the path that takes us through our lives will determine how we live those lives.... [read more]

- Output 3:

    The meaning of life is a simple answer. It’s not what you see when it comes to the surface, but deep down inside there'...
    I think that every man has an inner desire and want for love… I can say this because all women do! Every girl wants somebody they feel safe with at least some sort...... [view more]

# References

This code is based on:

- [LLaMA Reference Implementation](https://github.com/facebookresearch/llama)
- [GPTQ](https://github.com/IST-DASLab/gptq)
- [GPTQ for LLaMA](https://github.com/qwopqwop200/GPTQ-for-LLaMa)
- [Alpaca_lora_4bit](https://github.com/johnsmith0031/alpaca_lora_4bit)

Thanks to Meta AI for releasing [LLaMA](https://arxiv.org/abs/2302.13971), a powerful LLM.

# License

The original code was released under its respective license and copyrights, i.e.:

- `datautils.py` and `evaluate.py`:
[GPTQ for LLaMA](https://github.com/qwopqwop200/GPTQ-for-LLaMa) released under Apache 2.0 License
- `model.py` and `inference.py` (basis for `llama_2b_evaluate.py` and `llama_2b_inference.py`):
[Alpaca_lora_4bit](https://github.com/johnsmith0031/alpaca_lora_4bit) released under MIT License

We release our changes and additions to these files under the [Apache 2.0 License](LICENSE).
