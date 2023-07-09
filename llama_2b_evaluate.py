import sys

import torch

from datautils import get_loaders
from evaluate import llama_eval
from model import load_llama_model

if not torch.cuda.is_available():
    print("CUDA is needed to run the model.")
    sys.exit(0)

model_uri = 'GreenBitAI/LLaMA-7B-2bit'
cache_dir = './cache'

model, tokenizer = load_llama_model(model_uri, cache_dir=cache_dir, half=True, groupsize=32, bits=2)
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
