import argparse
import sys

import torch
from pprint import pprint
from datautils import get_loaders
from evaluate import llama_eval
from model import load_llama_model, QuantLinear
from LMClass import *
from utils.utils import create_logger
from pathlib import Path
from lm_eval import evaluator

if not torch.cuda.is_available():
    print("CUDA is needed to run the model.")
    sys.exit(0)

parser = argparse.ArgumentParser("Run inference with low-bit LLaMA models.")
parser.add_argument("-s", "--model-size", choices=["6b", "6B", "34b", "34B"], required=False, default="34B", type=str, help="Which model size to use.")
parser.add_argument("-b", "--wbits", choices=[2,4], required=False, default=2, type=int, help="which weight bit to evaluate")
parser.add_argument("-g", "--groupsize", choices=[8, 32], required=False, default=8, type=int, help="Specify quantization groups")
parser.add_argument("-t", "--tasks", default="openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,race,truthfulqa_mc,anli_r1,anli_r2,anli_r3,wic,rte,record", type=str, help="Specify harness tasks")
parser.add_argument("--limit", type=int, default=-1)
parser.add_argument("--num_fewshot", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=16)

args = parser.parse_args()
args.model_size = args.model_size.lower()

model_uri = f'GreenBitAI/yi-{args.model_size}-w{args.wbits}a16g{args.groupsize}'

asym = False
bits = args.wbits
double_groupsize=32
kquant = True if bits == 2 else False
v1 = False
return_config=True
cache_dir = './cache'

model, tokenizer, config = load_llama_model(model_uri, cache_dir=cache_dir, groupsize=args.groupsize, double_groupsize=double_groupsize, bits=bits, half=True, v1=v1, asym=asym, kquant=kquant, return_config=return_config)
model.eval()

lm = LMClass(model_uri, batch_size=args.batch_size, cache_dir=cache_dir)
lm.model=model
lm.tokenizer = tokenizer
lm.config = config
lm.reinitial()

print("start harness evaluation")
results={}
logger = create_logger(Path("../log/"))
t_results = evaluator.simple_evaluate(
            lm,
            tasks=args.tasks,
            num_fewshot=args.num_fewshot,
            limit=None if args.limit == -1 else args.limit,
        )
results.update(t_results)
logger.info(results)
pprint(results)
# for test of MMLU
if 'hendrycksTest' in args.tasks:
    all_cors = []
    all_cors_norm = []
    subcat_cors = {subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists}
    cat_cors = {cat: [] for cat in categories}
    cat_cors_norm = {cat: [] for cat in categories}
    for key in t_results['results'].keys():
        if not 'hendrycksTest' in key:
            continue
        subject = key.split('-')[-1]
        cors = t_results['results'][key]['acc']
        cors_norm = t_results['results'][key]['acc_norm']
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
                    cat_cors_norm[key].append(cors_norm)
            all_cors.append(cors)
            all_cors_norm.append(cors_norm)

    for cat in cat_cors:
        cat_acc = np.mean(cat_cors[cat])
        logger.info("Average accuracy {:.4f} - {}".format(cat_acc, cat))
    weighted_acc = np.mean(all_cors)
    logger.info("Average accuracy: {:.4f}".format(weighted_acc))
