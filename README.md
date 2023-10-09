# GreenBit LLaMA

This is GreenBitAI's research code for running **2-bit** and **1-bit** LLaMA models with extreme compression yet still strong performance, the quantized models are available on the [model zoo](https://huggingface.co/GreenBitAI?sort_models=downloads#models).

This is meant to be a research demo for the quality of the model.
There is no speed-up implemented yet.

## Roadmap

Over the next few months, we will continue offering 2-bit and 1-bit versions of LLaMA models.
Additionally, we are considering the provision of low-bit versions for other open-source LLMs in the future.

## Latest Updates
[10/04/2023] We are happy to release the W2A16 g8/32 TinyLLaMA-1.1B models.

[09/29/2023] We are happy to release the W2A16 g8 LLaMA-1 30B and LLaMA-2 70B models.

[09/12/2023] We are happy to announce the release of the 2-bit LLaMA-2 7B (W2A16 g32/g8) models.

[08/31/2023] We are happy to release the harness benchmarks on 14 zero-shot tasks based on our 2-bit models. Happy trying 😃🚀.

[08/16/2023] We are happy to release the 2-bit OpenLLaMA 3B models, which are quantized into 2-bit representation yet still with strong performance 😃⭐.

## Pretrained Model
| LLM Models            | Method     | Bits | Groupsize | Wikitext2 | C4    | Checkpoint Size (GiB) |
|:-------------------------:|:----------:|:----:|:---------:|:---------:|:-----:|:---------------------:|
| **LLaMA-2-70B**[^3]        | FP16       |  16  |     -     |      3.31 |  5.70 |        130           |                   
|                           | [**Ours**](https://huggingface.co/GreenBitAI/LLaMA-2-70B-2bit-groupsize8)   |   2  |     8    |      3.87 |  5.96 |         26.9           |
| **LLaMA-1-30B**[^3]        | FP16       |  16  |     -     |      4.10 |  5.98 |        60.5          |
|                           | [**Ours**](https://huggingface.co/GreenBitAI/LLaMA-30B-2bit-groupsize8)   |   2  |     8    |      4.75 |  6.57 |         12.9           |
| **LLaMA-2-7B**[^3]        | FP16       |  16  |     -     |      5.47 |  6.97 |        12.5           |
|                           | GPTQ[^4]   |   4  |    128    |      5.61 |  7.12 |         3.6           |
|                           | GPTQ[^4]   |   2  |    128    |     2.2e5 | 1.7e5 |         2.2           |
|                           | OmniQuant[^5]|   4  |    128    |      5.58 |  7.12 |         3.8           |
|                           | OmniQuant[^5]|   3  |    128    |      6.03 |  7.35 |         3.2           |
|                           | OmniQuant[^5]|   2  |     128   |      12.84| 17.40 |         2.2           |
|                           | OmniQuant[^5]|   2  |     64    |      10.56| 13.77 |           -           |
|                           | [**Ours**](https://huggingface.co/GreenBitAI/LLaMA-2-7B-4bit-groupsize32)   |   4  |     32    |      5.55 |  7.08 |         3.7           |
|                           | [**Ours**](https://huggingface.co/GreenBitAI/LLaMA-2-7B-2bit-groupsize8)   |   2  |      8    |      6.09 |  7.63 |         2.9           |
|                           | [**Ours**](https://huggingface.co/GreenBitAI/LLaMA-2-7B-2bit-groupsize32)   |   2  |     32    |      7.13 |  8.67 |         2.2           |
| **LLaMA-1-7B**[^2]        | FP16       |  16  |     -     |      5.67 |  7.07 |        12.5           |
|                           | GPTQ[^4]   |   4  |    128    |      5.85 |  7.21 |         3.6           |
|                           | GPTQ[^4]   |   3  |    128    |      6.61 |  7.85 |         3.0           |
|                           | OmniQuant[^5]|   2  |     128   |      10.53| 13.89 |         2.2           |
|                           | [**Ours**](https://huggingface.co/GreenBitAI/LLaMA-7B-2bit)   |   2  |     32    |      7.59 |  8.96 |         2.2           |
| **LLaMA 3B**[^1]          | FP16       |  16  |     -     |      7.34 |  9.33 |         6.8           |
|                           | GPTQ[^4]   |   4  |    128    |      7.54 |  9.58 |         1.9           |
|                           | [**Ours**](https://huggingface.co/GreenBitAI/LLaMA-3B-4bit-groupsize32)   |   4  |      32    |      7.43 | 9.51 |         2.0           |
|                           | [**Ours**](https://huggingface.co/GreenBitAI/LLaMA-3B-2bit-groupsize8)   |   2  |      8    |      8.32 | 10.56 |         1.5           |
|                           | [**Ours**](https://huggingface.co/GreenBitAI/LLaMA-3B-2bit-groupsize16)   |   2  |     16    |      8.92 | 11.29 |         1.3           |
|                           | [**Ours**](https://huggingface.co/GreenBitAI/LLaMA-3B-2bit-groupsize32)   |   2  |     32    |      9.82 | 12.14 |         1.2           |
| **TinyLLaMA 1.1B**[^6]          | FP16       |  16  |     -     |      9.10 |  10.6 |         4.0           |
|                                 | [**Ours**](https://huggingface.co/GreenBitAI/LLaMA-2-1.1B-2bit-groupsize8)       |  2  |     8     |      9.99 |  11.75 |         0.6           |
|                                 | [**Ours**](https://huggingface.co/GreenBitAI/LLaMA-2-1.1B-2bit-groupsize32)       |  2  |     32     |      12.04 |  14.27 |         0.5           |

##  Fine-tuned Model
| LLM Models                | Method     | Bits | Checkpoint Size (GiB) |
|:-------------------------:|:----------:|:----:|:---------:|
| **LLaMA-2-70B-Chat**[^3]  | FP16       |  16  |     130     |                     
|                           |  [**Ours**](https://huggingface.co/GreenBitAI/LLaMA-2-70B-CHAT-2bit-groupsize8)  |   2  |     26.9     |    
| **CodeLLaMA-7B**[^7]      | FP16       |  16  |     12.5     |                      
|                           |  **Ours**  |   -  |     -     |
| **CodeLLaMA-13B**[^7]      | FP16       |  16  |     24     |                      
|                           |  **Ours**  |   -  |     -     |
| **CodeLLaMA-34B**[^7]      | FP16       |  16  |     63     |                      
|                           |  [**Ours**](https://huggingface.co/GreenBitAI/codellama-34B-w2g16g8)  |   2  |     13.5     |    

[^1]: [OpenLLaMA](https://github.com/openlm-research/open_llama)
[^2]: [LLaMA-1](https://arxiv.org/abs/2302.13971)
[^3]: [LLaMA-2](https://ai.meta.com/llama/)
[^4]: [GPTQ](https://arxiv.org/abs/2210.17323)
[^5]: [OmniQuant](https://arxiv.org/pdf/2308.13137.pdf)
[^6]: [TinyLLaMA](https://github.com/jzhang38/TinyLlama)
[^7]: [CodeLLaMA](https://github.com/facebookresearch/codellama)


## Zero-Shot Evaluation
| Task          | Metric   | TinyLLaMA 1.1B q2g32 | TinyLLaMA 1.1B q2g8 | LLaMA 3B q2g32 | LLaMA 3B q2g16 | LLaMA 3B q2g8 | LLaMA-1 7B q2g32 | LLaMA-2 7B q2g32 | LLaMA-2 7B q2g8 | LLaMA 3B FP16 | LLaMA-1 7B FP16 |
|---------------|----------|----------------|---------------|----------------|----------------|--------------|------------------|------------------|----------------|--------------|-----------------|
| Openbookqa    | acc      | 0.152          | 0.192         | 0.196          | 0.238          | 0.242        | 0.224            | 0.246            | 0.296          | 0.27         | 0.29            |
|               | ac_norm  | 0.328          | 0.338         | 0.332          | 0.358          | 0.362        | 0.388            | 0.376            | 0.4            | 0.4          | 0.41            |
| arc_challenge | acc      | 0.3268         | 0.2278        | 0.279          | 0.2978         | 0.3148       | 0.3422           | 0.3268           | 0.3618         | 0.34         | 0.39            |
|               | ac_norm  | 0.3387         | 0.273         | 0.2944         | 0.3319         | 0.3345       | 0.3387           | 0.3387           | 0.372          | 0.37         | 0.41            |
| hellawswag    | acc      | 0.34           | 0.3769        | 0.4238         | 0.444          | 0.462        | 0.4996           | 0.4961           | 0.5379         | 0.49         | 0.68            |
|               | ac_norm  | 0.4097         | 0.4711        | 0.5685         | 0.5988         | 0.6242       | 0.6447           | 0.6464           | 0.7014         | 0.67         | 0.73            |
| piqa          | acc      | 0.6518         | 0.6931        | 0.7024         | 0.716          | 0.7291       | 0.7476           | 0.7503           | 0.7715         | 0.75         | 0.78            |
|               | ac_norm  | 0.6393         | 0.6812        | 0.7116         | 0.7247         | 0.7312       | 0.7443           | 0.7421           | 0.7568         | 0.76         | 0.78            |
| arc_easy      | acc      | 0.4411         | 0.5109        | 0.5997         | 0.646          | 0.6528       | 0.6061           | 0.6174           | 0.6254         | 0.69         | 0.68            |
|               | ac_norm  | 0.3716         | 0.412         | 0.5417         | 0.58           | 0.5972       | 0.4566           | 0.4781           | 0.4958         | 0.65         | 0.52            |
| Winogrande    | acc      | 0.532          | 0.5249        | 0.5683         | 0.5888         | 0.6054       | 0.6283           | 0.6298           | 0.6582         | 0.62         | 0.68            |
| boolq         | acc      | 0.592          | 0.6174        | 0.6281         | 0.6636         | 0.6327       | 0.6425           | 0.7061           | 0.7242         | 0.68         | 0.75            |
| truthfulqa_mc | mc1      | 0.2338         | 0.2277        | 0.2509         | 0.2118         | 0.2252       | 0.224            | 0.2313           | 0.2399         | 0.22         | 0.21            |
|               | mc2      | 0.4211         | 0.406         | 0.3962         | 0.3501         | 0.3625       | 0.3702           | 0.3854           | 0.3795         | 0.35         | 0.34            |
| anli_r1       | acc      | 0.363          | 0.336         | 0.337          | 0.334          | 0.344        | 0.331            | 0.333            | 0.363          | 0.33         | 0.35            |
| anli_r2       | acc      | 0.331          | 0.346         | 0.335          | 0.332          | 0.331        | 0.326            | 0.349            | 0.347          | 0.32         | 0.34            |
| anli_r3       | acc      | 0.3758         | 0.3633        | 0.3358         | 0.3383         | 0.3425       | 0.3417           | 0.36             | 0.3733         | 0.35         | 0.37            |
| wic           | acc      | 0.5            | 0.5           | 0.4984         | 0.5094         | 0.4969       | 0.4984           | 0.4953           | 0.489          | 0.48         | 0.5             |
| rte           | acc      | 0.4874         | 0.4874        | 0.5596         | 0.5993         | 0.5632       | 0.639            | 0.6065           | 0.6426         | 0.58         | 0.56            |
| record        | f1       | 0.7608         | 0.8023        | 0.8502         | 0.8625         | 0.8687       | 0.8859           | 0.8872           | 0.9037         | 0.88         | 0.91            |
|               | em       | 0.753          | 0.7934        | 0.8427         | 0.8545         | 0.8612       | 0.8781           | 0.8801           | 0.8959         | 0.89         | 0.91            |
| Average       |          | 0.438          | 0.4498        | 0.4881         | 0.5037         | 0.5087       | 0.5122           | 0.5181           | 0.5391         | 0.528        | 0.5519          |
| model size    | GiB      | 0.5            | 0.6           | 1.2            | 1.3            | 1.5          | 2.2              | 2.2              | 2.9            | 6.8          | 12.5           |

## Requirements

The inference currently requires a machine with CUDA installed.
Then you can simply run:

```bash
pip install -r requirements.txt
```

## Try the model

Use the environment variable `CUDA_VISIBLE_DEVICES` to select the correct GPU.
Multi-GPU is not supported, but the model is very compressed, so 1 GPU should be enough.
To use the instruction-tuned model, you can use the following commands
in ```scripts/```. Predefined scripts already there:

```bash
bash scripts/evaluate/tiny_llama_w2a16g32.sh    # for open task evaluation of the base model.
bash scripts/inference/llama2_70b_w2a16g8.sh     # for text generation inference of the base model.
bash scripts/instruction-chat/llama2_70b_w2a16g8.sh  # for instruction following chat of the fine-tuned model.
bash scripts/inference/codellama_34b_w2a16g8.sh         # for text generation inference of the codellama model
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
