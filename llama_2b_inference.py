import argparse
import sys

import torch

from model import load_llama_model

if not torch.cuda.is_available():
    print("CUDA is needed to run the model.")
    sys.exit(0)

parser = argparse.ArgumentParser("Run inference with low-bit LLaMA models.")
parser.add_argument("-s", "--model-size", choices=["3b", "3B", "7b", "7B", "13b", "13B"], required=False, default="7B", type=str, help="Which model size to use.")
parser.add_argument("-v", "--llama-version", choices=[1, 2], required=False, default=1, type=int, help="which version to evaluate")
parser.add_argument("-g", "--groupsize", choices=[8, 16, 32], required=False, default=32, type=int, help="Specify quantization groups")

args = parser.parse_args()
args.model_size = args.model_size.upper()

if args.llama_version == 1:
    model_uri = f'GreenBitAI/LLaMA-{args.model_size}-2bit'

    if args.model_size in ["3b", "3B"]:
        model_uri = model_uri + f'-groupsize{args.groupsize}'

else:
    model_uri = f'GreenBitAI/LLaMA-2-{args.model_size}-2bit'

    if args.model_size in ["3b", "3B", "7b", "7B"]:
        model_uri = model_uri + f'-groupsize{args.groupsize}'


if args.groupsize == 32:
    asym = True
else:
    asym = False

cache_dir = './cache'

model, tokenizer = load_llama_model(model_uri, cache_dir=cache_dir, groupsize=args.groupsize, bits=2, half=True, asym=asym)
model.eval()

prompt = '''The meaning of life is'''
batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)

batch = {k: v.cuda() for k, v in batch.items()}
model.cuda()

for i in range(10):
    with torch.no_grad():
        generated = model.generate(
            inputs=batch["input_ids"],
            do_sample=True,
            use_cache=True,
            repetition_penalty=1.5,
            max_new_tokens=100,
            temperature=0.9,
            top_p=0.95,
            top_k=20,
            return_dict_in_generate=True,
            output_attentions=False,
            output_hidden_states=False,
            output_scores=False
        )
    result_text = tokenizer.decode(generated['sequences'].cpu().tolist()[0])
    print(result_text + "\n")
