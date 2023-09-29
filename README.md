# GreenBit LLaMA

This is GreenBitAI's research code for running **1-bit** and **2-bit** LLaMA models with extreme compression yet still strong performance.

This is meant to be a research demo for the quality of the model.
There is no speed-up implemented yet.

## Roadmap

Over the next few months, we will continue offering 2-bit and 1-bit versions of LLaMA models.
Additionally, we are considering the provision of low-bit versions for other open-source LLMs in the future.

## Updates
#### 09/29/2023
We are happy to release the W2A16 LLaMA-1 30B and LLaMA-2 70B models. Both are quantized into 2-bit representation yet still with strong performance.

#### 09/12/2023 
🎉 We are happy to announce the release of the 2-bit LLaMA-2 7B (W2A16 g32/g8) models, boasting impressive performance. For the first time, the W2A16 model showcases performance on par with the existing W3A16 quantized model while exhibiting superior potential for hardware compatibility. Furthermore, our W2A16 LLaMA-2 7B outperforms FP16 LLaMA-3B on several open evaluation tasks with a 3.09x and 5.68x compression ratio to 3B and 7B LLaMA respectively. Happy trying!

#### 08/31/2023
We are happy to release the harness benchmarks on 14 zero-shot tasks based on our 2-bit models. Happy trying 😃🚀!!

#### 08/16/2023
We are happy to release the 2-bit OpenLLaMA 3B models, which are quantized into 2-bit representation yet still with strong performance 😃⭐.

## Results

| LLaMA Version             | Method     | Bits | Groupsize | Wikitext2 | C4    | Checkpoint Size (GiB) |
|:-------------------------:|:----------:|:----:|:---------:|:---------:|:-----:|:---------------------:|
| **LLaMA-2-70B**[^3]        | FP16       |  16  |     -     |      3.31 |  5.70 |        130           |                   
|                           | [**Ours**](https://huggingface.co/GreenBitAI/LLaMA-2-70B-2bit-groupsize8)   |   2  |     8    |      4.06 |  6.15 |         26.9           |
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
|                           | [**Ours**](https://huggingface.co/GreenBitAI/LLaMA-2-7B-2bit-groupsize32)   |   2  |     32    |      7.13 |  8.68 |         2.2           |
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
| **TinyLLaMA 1.1B**[^6]          | FP16       |  16  |     -     |      9.53 |  11.01 |         4.0           |
|                                 | **Ours**       |  2  |     -     |      - |  - |         -           |


[^1]: [LLaMA-1-3B](https://github.com/openlm-research/open_llama)
[^2]: [LLaMA-1-7B](https://arxiv.org/abs/2302.13971)
[^3]: [LLaMA-2-7B](https://ai.meta.com/llama/)
[^4]: [GPTQ](https://arxiv.org/abs/2210.17323)
[^5]: [OmniQuant](https://arxiv.org/pdf/2308.13137.pdf)
[^6]: [TinyLLaMA](https://github.com/jzhang38/TinyLlama)


## Zero-Shot Evaluation
| Task          | Metric   | LLaMA 3B q2g32 | LLaMA 3B q2g16 | LLaMA 3B q2g8 | LLaMA-1 7B q2g32 | LLaMA-2 7B q2g32 | LLaMA-2 7B q2g8 | LLaMA 3B FP16 | LLaMA-1 7B FP16 |
|---------------|----------|----------------|----------------|--------------|------------------|------------------|----------------|--------------|-----------------|
| Openbookqa    | acc      | 0.196          | 0.238          | 0.242        | 0.224            | 0.246            | 0.296          | 0.27         | 0.29            |
|               | ac_norm  | 0.332          | 0.358          | 0.362        | 0.388            | 0.376            | 0.4            | 0.4          | 0.41            |
| arc_challenge | acc      | 0.279          | 0.2978         | 0.3148       | 0.3422           | 0.3268           | 0.3618         | 0.34         | 0.39            |
|               | ac_norm  | 0.2944         | 0.3319         | 0.3345       | 0.3387           | 0.3387           | 0.372          | 0.37         | 0.41            |
| hellawswag    | acc      | 0.4238         | 0.444          | 0.462        | 0.4996           | 0.4961           | 0.5379         | 0.49         | 0.68            |
|               | ac_norm  | 0.5685         | 0.5988         | 0.6242       | 0.6447           | 0.6464           | 0.7014         | 0.67         | 0.73            |
| piqa          | acc      | 0.7024         | 0.716          | 0.7291       | 0.7476           | 0.7503           | 0.7715         | 0.75         | 0.78            |
|               | ac_norm  | 0.7116         | 0.7247         | 0.7312       | 0.7443           | 0.7421           | 0.7568         | 0.76         | 0.78            |
| arc_easy      | acc      | 0.5997         | 0.646          | 0.6528       | 0.6061           | 0.6174           | 0.6254         | 0.69         | 0.68            |
|               | ac_norm  | 0.5417         | 0.58           | 0.5972       | 0.4566           | 0.4781           | 0.4958         | 0.65         | 0.52            |
| Winogrande    | acc      | 0.5683         | 0.5888         | 0.6054       | 0.6283           | 0.6298           | 0.6582         | 0.62         | 0.68            |
| boolq         | acc      | 0.6281         | 0.6636         | 0.6327       | 0.6425           | 0.7061           | 0.7242         | 0.68         | 0.75            |
| truthfulqa_mc | mc1      | 0.2509         | 0.2118         | 0.2252       | 0.224            | 0.2313           | 0.2399         | 0.22         | 0.21            |
|               | mc2      | 0.3962         | 0.3501         | 0.3625       | 0.3702           | 0.3854           | 0.3795         | 0.35         | 0.34            |
| anli_r1       | acc      | 0.337          | 0.334          | 0.344        | 0.331            | 0.333            | 0.363          | 0.33         | 0.35            |
| anli_r2       | acc      | 0.335          | 0.332          | 0.331        | 0.326            | 0.349            | 0.347          | 0.32         | 0.34            |
| anli_r3       | acc      | 0.3358         | 0.3383         | 0.3425       | 0.3417           | 0.36             | 0.3733         | 0.35         | 0.37            |
| wic           | acc      | 0.4984         | 0.5094         | 0.4969       | 0.4984           | 0.4953           | 0.489          | 0.48         | 0.5             |
| rte           | acc      | 0.5596         | 0.5993         | 0.5632       | 0.639            | 0.6065           | 0.6426         | 0.58         | 0.56            |
| record        | f1       | 0.8502         | 0.8625         | 0.8687       | 0.8859           | 0.8872           | 0.9037         | 0.88         | 0.91            |
|               | em       | 0.8427         | 0.8545         | 0.8612       | 0.8781           | 0.8801           | 0.8959         | 0.89         | 0.91            |
| Average       |          | 0.4881         | 0.5037         | 0.5087       | 0.5122           | 0.5181           | 0.5391         | 0.528        | 0.5519          |

![Zero-Shot Harness Evaluation](https://github.com/GreenBitAI/low_bit_llama/assets/24189567/53190300-3cbc-4264-b847-dd8e0f2fac1e)

## Requirements

The inference currently requires a machine with CUDA installed.
Then you can simply run:

```bash
pip install -r requirements.txt
```

## Try the model

Use the environment variable `CUDA_VISIBLE_DEVICES` to select the correct GPU.
Multi-GPU is not supported, but the model is very compressed, so 1 GPU should be enough.
To use the instruction-tuned model, you can use the following commands.
Use the `-s` option to select the model size (default is 7B) and `-h` to check for options.

```bash
CUDA_VISIBLE_DEVICES=0 python llama_2b_sft_alpaca.py  # uses default prompt
CUDA_VISIBLE_DEVICES=0 python llama_2b_sft_alpaca.py -s 13B  # uses default prompt with 13B model
# custom prompts:
CUDA_VISIBLE_DEVICES=0 python llama_2b_sft_alpaca.py -p "Tell me the meaning of life"
CUDA_VISIBLE_DEVICES=0 python llama_2b_sft_alpaca.py -s 13B -p "Tell me the meaning of life"
```

Please note: the instruction-tuned model currently may keep generating tokens even after finishing the answer, which may result in strange output after the answer.

### Examples

Exemplary model output from prompting the instruction-tuned model:

- Prompt 1: "Give me an interesting traveling plan to Europe"

> One of the most popular and exciting destinations for travelers in Europe is London, England!
> The city has so much to offer with its iconic landmarks such as Big Ben, Tower Bridge, Buckingham Palace, Westminster Abbey, and more.
> You can also explore other cities like Paris, Rome, Berlin, Madrid or Barcelona on your trip.
> 
> To make it easier for you, I suggest booking flights through low-cost airlines like Ryanair, EasyJet, or Norwegian Air International.
> These carriers have great deals and are known for their punctuality.
> Additionally, consider renting a car if you want to see some of the countryside while exploring different parts of Europe. 
> 
> For accommodation options, there's plenty of hostels, hotels, and apartments available online at affordable prices.
> Remember to always check reviews before making any reservations. 
> Hope this helps you plan out your journey to Europe! Have fun and enjoy your adventure!"

- Prompt 2: "Give me a list of sights to visit in Europe"

> 1) Paris, France - The City of Lights with its iconic Eiffel Tower and Notre-Dame Cathedral.
> 2) London, England - A vibrant city known for its historic landmarks such as Buckingham Palace and Big Ben.
> 3) Rome, Italy - Home to ancient ruins like the Colosseum and St Peter's Basilica, along with modern architecture and cuisine.
> 4) Barcelona, Spain - Known for its stunning beaches, Gaudi’s Sagrada Familia cathedral, and lively nightlife scene.
> 5) Berlin, Germany - A dynamic capital city with diverse neighborhoods, museums, and historical monuments.
> 6) Madrid, Spain - A bustling metropolitan center with world-class art galleries, parks, and delicious food.
> 7) Vienna, Austria - An elegant imperial city famous for its palaces, gardens, and classical music culture.
> 8) Athens, Greece - A charming Mediterranean destination with ancient temples, beautiful beaches, and unique street markets.

- Prompt 3: "Tell me the meaning of life"

> The meaning of life, also known as the purpose or goal of existence, can be subjective and personal to each individual.
> However, some common themes include finding happiness, fulfillment, self-realization, love, connection with others, and making a positive impact on the world around us.
> Ultimately, it's about living in harmony with oneself and pursuing one’s passions while striving for growth and progress towards greater understanding and enlightenment.
> It's important not to focus solely on achieving success but rather on creating a fulfilled and meaningful life through experiences, relationships, and personal development.
> Overall, the search for meaning is a journey that is unique to everyone and may take different forms at various stages throughout our lives.
> The key is to find what brings joy and satisfaction, and make choices based on those values.
> 
> I hope this helps you understand the concept of the meaning of life better! Please let me know if there are any other questions I can answer. Thank you!

## Run inference or evaluation of the base model

If you want to evaluate or do inference on the base model (not tuned for instructions), you can use the following commands
(use the `-s` switch to select model size):

```bash
CUDA_VISIBLE_DEVICES=0 python llama_2b_evaluate.py
CUDA_VISIBLE_DEVICES=0 python llama_2b_evaluate.py -s 13b  # evaluate the 13B model
CUDA_VISIBLE_DEVICES=0 python llama_2b_inference.py
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
