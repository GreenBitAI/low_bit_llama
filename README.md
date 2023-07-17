# GreenBit LLaMA

This is GreenBitAI's research code for running **1-bit** and **2-bit** LLaMA models with extreme compression yet still strong performance.

This is meant to be a research demo for the quality of the model.
There is no speed-up implemented yet.

## Roadmap

Over the next few weeks, we will continue to offer both 2-bit and 1-bit versions of LLaMA models.
Additionally, we are considering the provision of low-bit versions for other open-source LLMs in the future.

## Results

The model size of LLaMA-7B shrinks from 12.5 GB (FP16) to 2.2 GiB (2 bits).

| [LLaMA-7B](https://arxiv.org/abs/2302.13971) | Bits | Wikitext2 | C4   | PTB   | checkpoint size (GiB) |
|----------------------------------------------|------|-----------|------|-------|-----------------------|
| FP16                                         | 16   | 5.67      | 7.07 | 8.80  | 12.5                  |
| [GPTQ](https://arxiv.org/abs/2210.17323)     | 4    | 5.85      | 7.21 | 9.00  | 3.7                   |
| **Ours**                                     | 2    | 7.59      | 8.96 | 11.33 | 2.2                   |

## Requirements

The inference currently requires a machine with CUDA installed.
Then you can simply run:

```bash
pip install -r requirements.txt
```

## Try the model

Use the environment variable `CUDA_VISIBLE_DEVICES` to select the correct GPU.
Multi-GPU is not supported, but the model is very compressed, so 1 GPU should be enough.
To use the instruction-tuned model, you can use the following commands:

```bash
CUDA_VISIBLE_DEVICES=0 python llama_2b_sft_alpaca.py  # uses default prompt
CUDA_VISIBLE_DEVICES=0 python llama_2b_sft_alpaca.py -p "Tell me the meaning of life"
```

Please note: the instruction-tuned model currently keeps generating tokens even after finishing the answer, which may result in strange output after the answer.

### Examples

Exemplary model output from prompting the instruction-tuned model:

- Prompt 1: "Give me an interesting traveling plan to Europe"

> One of the most popular and exciting destinations for travelers in Europe is London, England!
> The city has so much to offer with its iconic landmarks such as Big Ben, Tower Bridge, Buckingham Palace, Westminster Abbey, and more.
> You can also explore other cities like Paris, Rome, Berlin, Madrid or Barcelona on your trip.
> To make it easier for you, I suggest booking flights through low-cost airlines like Ryanair, EasyJet, or Norwegian Air International.
> These carriers have great deals and are known for their punctuality.
> Additionally, consider renting a car if you want to see some of the countryside while exploring different parts of Europe. 
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
> I hope this helps you understand the concept of the meaning of life better! Please let me know if there are any other questions I can answer. Thank you!

## Run inference or evaluation of the base model

If you want to evaluate or do inference on the base model (not tuned for instructions), you can use the following commands:

```bash
CUDA_VISIBLE_DEVICES=0 python llama_2b_evaluate.py
CUDA_VISIBLE_DEVICES=0 python llama_2b_inference.py
```

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
- `model.py`, `peft_tuners_lora.py` and `inference.py` (basis for `llama_2b_*.py` files):
[Alpaca_lora_4bit](https://github.com/johnsmith0031/alpaca_lora_4bit) released under MIT License

We release our changes and additions to these files under the [Apache 2.0 License](LICENSE).
