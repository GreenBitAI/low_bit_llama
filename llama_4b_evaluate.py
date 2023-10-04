import argparse
import sys

import torch

from datautils import get_loaders
from evaluate import llama_eval
from model import load_llama_model

if not torch.cuda.is_available():
    print("CUDA is needed to run the model.")
    sys.exit(0)

parser = argparse.ArgumentParser("Run inference with low-bit LLaMA models.")
parser.add_argument("-s", "--model-size", choices=["3b", "3B", "7b", "7B", "30b", "30B", "70b", "70B"], required=False, default="7B", type=str, help="Which model size to use.")
parser.add_argument("-v", "--llama-version", choices=[1, 2], required=False, default=2, type=int, help="which version to evaluate")
parser.add_argument("-g", "--groupsize", choices=[8, 6, 32], required=False, default=32, type=int, help="Specify quantization groups")

args = parser.parse_args()
args.model_size = args.model_size.upper()

if args.llama_version == 1: 
    model_uri = f'GreenBitAI/LLaMA-{args.model_size}-4bit' 
     
    if args.model_size in ["3b", "3B", "30b", "30B"]: 
        model_uri = model_uri + f'-groupsize{args.groupsize}' 
     
else: 
    model_uri = f'GreenBitAI/LLaMA-2-{args.model_size}-4bit' 
 
    if args.model_size in ["3b", "3B", "7b", "7B", "70b", "70B"]: 
        model_uri = model_uri + f'-groupsize{args.groupsize}' 
    else: 
        raise NotImplemented

if args.groupsize == 32: 
    asym = True 
else: 
    asym = False 

bits = 4

if bits == 2:
    if asym:
        double_groupsize = -1
    else:
        if args.groupsize == 32: 
            double_groupsize=32   
        else:   
            if args.llama_version == 1: 
                double_groupsize=64  
            else: 
                double_groupsize=32 
else:
    if args.model_size in ["3b", "3B"]:
        double_groupsize=64
    elif args.model_size in ["7b", "7B"]:
        double_groupsize=256

v1 = (args.llama_version==1) and args.model_size in ["7b", "7B"]

cache_dir = './cache'

model, tokenizer = load_llama_model(model_uri, cache_dir=cache_dir, groupsize=args.groupsize, double_groupsize=double_groupsize, v1=v1, bits=4, half=True, asym=asym)
model.eval()

print("Loading dataset 'c4' for evaluation...")
_, c4_testloader = get_loaders('c4', model=model_uri, cache_dir=cache_dir, nsamples=128, seed=0, seqlen=2048)
llama_eval(model, c4_testloader)

print("Loading dataset 'wikitext2' for evaluation...")
_, wikitext2_testloader = get_loaders('wikitext2', model=model_uri, cache_dir=cache_dir, nsamples=128, seed=0, seqlen=2048)
llama_eval(model, wikitext2_testloader)

print("Loading dataset 'ptb' for evaluation...")
_, ptb_testloader = get_loaders('ptb', model=model_uri, cache_dir=cache_dir, nsamples=128, seed=0, seqlen=2048)
llama_eval(model, ptb_testloader)
