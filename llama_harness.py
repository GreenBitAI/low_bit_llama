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

parser = argparse.ArgumentParser("Run harness evaluation with low-bit Yi models.")
parser.add_argument("-s", "--model-size", choices=["1.1b", "1.1B", "3b", "3B", "7b", "7B", "70b", "70B"], required=False, default="7B", type=str, help="Which model size to use.")
parser.add_argument("-b", "--wbits", choices=[2], required=False, default=2, type=int, help="which weight bit to evaluate")
parser.add_argument("-g", "--groupsize", choices=[8, 16, 32], required=False, default=32, type=int, help="Specify quantization groups")
parser.add_argument("-v", "--llama-version", choices=[1, 2], required=False, default=1, type=int, help="which version to evaluate")
parser.add_argument("-t", "--tasks", default="openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,race,truthfulqa_mc,anli_r1,anli_r2,anli_r3,wic,rte,record", type=str, help="Specify harness tasks")
parser.add_argument("--limit", type=int, default=-1)
parser.add_argument("--num_fewshot", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=16)

args = parser.parse_args()
args.model_size = args.model_size.upper()

if args.llama_version == 1:
    model_uri = f'GreenBitAI/LLaMA-{args.model_size}-2bit'

    if args.model_size in ["3b", "3B", "30b", "30B"]:
        model_uri = model_uri + f'-groupsize{args.groupsize}'

else:
    model_uri = f'GreenBitAI/LLaMA-2-{args.model_size}-2bit'

    if args.model_size in ["1.1b", "1.1B", "7b", "7B", "70b", "70B"]:
        model_uri = model_uri + f'-groupsize{args.groupsize}'
    else:
        raise NotImplemented

if args.groupsize == 32 and args.model_size not in ["1.1b", "1.1B"]:
    asym = True
else:
    asym = False

bits = 2

if bits == 2:
    if asym:
        double_groupsize = -1
    else:
        if args.groupsize == 32:
            double_groupsize=32
        else:
            if args.llama_version == 1 and args.model_size not in ["30b", "30B"]:
                double_groupsize=64
            else:
                double_groupsize=32
else:
    if args.model_size in ["3b", "3B"]:
        double_groupsize=64
    elif args.model_size in ["7b", "7B"]:
        double_groupsize=256

v1 = (args.llama_version==1) and args.model_size in ["7b", "7B"]

bits = args.wbits
kquant = False
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
