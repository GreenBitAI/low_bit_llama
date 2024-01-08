import argparse
import sys

import torch

from datautils import get_loaders
from evaluate import llama_eval
from model import load_llama_model, QuantLinear

if not torch.cuda.is_available():
    print("CUDA is needed to run the model.")
    sys.exit(0)

parser = argparse.ArgumentParser("Run evaluation with low-bit Yi models.")
parser.add_argument("-s", "--model-size", choices=["6b", "6B", "34b", "34B"], required=False, default="34B", type=str, help="Which model size to use.")
parser.add_argument("-b", "--wbits", choices=[2,4], required=False, default=2, type=int, help="which weight bit to evaluate")
parser.add_argument("-g", "--groupsize", choices=[8, 32], required=False, default=8, type=int, help="Specify quantization groups")
parser.add_argument("-c", "--chat", action="store_true", required=False, help="Specify chating model")

args = parser.parse_args()
args.model_size = args.model_size.lower()

if args.chat:
    model_uri = f'GreenBitAI/yi-{args.model_size}-chat-w{args.wbits}a16g{args.groupsize}'
else:
    model_uri = f'GreenBitAI/yi-{args.model_size}-w{args.wbits}a16g{args.groupsize}'

asym = False
bits = args.wbits
double_groupsize=32
kquant=True
v1 = False

cache_dir = './cache'

model, tokenizer = load_llama_model(model_uri, cache_dir=cache_dir, groupsize=args.groupsize, double_groupsize=double_groupsize, bits=bits, half=True, v1=v1, asym=asym, kquant=kquant)
model.eval()

print("Loading dataset 'wikitext2' for evaluation...")
_, wikitext2_testloader = get_loaders('wikitext2', model=model_uri, cache_dir=cache_dir, nsamples=128, seed=0, seqlen=2048, tokenizer=tokenizer)
llama_eval(model, wikitext2_testloader)

print("Loading dataset 'ptb' for evaluation...")
_, ptb_testloader = get_loaders('ptb', model=model_uri, cache_dir=cache_dir, nsamples=128, seed=0, seqlen=2048, tokenizer=tokenizer)
llama_eval(model, ptb_testloader)

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
            max_new_tokens=256,
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

