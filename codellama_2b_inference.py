import argparse
import sys

import torch

from datautils import get_loaders
from evaluate import llama_eval
from model import load_llama_model, QuantLinear

if not torch.cuda.is_available():
    print("CUDA is needed to run the model.")
    sys.exit(0)

parser = argparse.ArgumentParser("Run inference with low-bit LLaMA models.")
parser.add_argument("-s", "--model-size", choices=["34B", "python-34B", "instruction-34B"], required=False, default="34B", type=str, help="Which model size to use.")
parser.add_argument("-v", "--llama-version", choices=[2], required=False, default=2, type=int, help="which version to evaluate")
parser.add_argument("-g", "--groupsize", choices=[8], required=False, default=8, type=int, help="Specify quantization groups")

args = parser.parse_args()
args.model_size = args.model_size#.upper()

model_uri = f'GreenBitAI/codellama-{args.model_size}-w2a16g{args.groupsize}'

asym = False
bits = 2
double_groupsize=32

v1 = (args.llama_version==1) and args.model_size in ["7b", "7B"]

cache_dir = './cache'

model, tokenizer = load_llama_model(model_uri, cache_dir=cache_dir, groupsize=args.groupsize, double_groupsize=double_groupsize, bits=2, half=True, v1=v1, asym=asym)
model.eval()


prompt = '''The difference between python and C++:'''

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
            max_new_tokens=512,
            temperature=0.8,
            top_p=0.95,
            top_k=20,
            return_dict_in_generate=True,
            output_attentions=False,
            output_hidden_states=False,
            output_scores=False
        )
    result_text = tokenizer.decode(generated['sequences'].cpu().tolist()[0])
    print(result_text + "\n")

