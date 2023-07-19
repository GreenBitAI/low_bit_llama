import sys
import argparse

import torch

from model import load_llama_model_lora
from peft_tuners_lora import replace_peft_model_with_lora_model

if not torch.cuda.is_available():
    print("CUDA is needed to run the model.")
    sys.exit(0)

DEFAULT_PROMPT = "Give me an interesting traveling plan to Europe"
parser = argparse.ArgumentParser("prompt the instruction-tuned model")
parser.add_argument("-p", "--prompt", default=DEFAULT_PROMPT, required=False, type=str, help="Enter your prompt here.")
parser.add_argument("-s", "--model-size", choices=["7b", "7B", "13b", "13B"], required=False, default="7B", type=str, help="Which model size to use.")
args = parser.parse_args()
args.model_size = args.model_size.upper()

if args.prompt == DEFAULT_PROMPT:
    print(f"Using default prompt: {args.prompt}")
else:
    print(f"Input prompt: {args.prompt}")

replace_peft_model_with_lora_model()

model_uri = f'GreenBitAI/LLaMA-{args.model_size}-2bit'
lora_uri = f'GreenBitAI/LLaMA-{args.model_size}-2bit-alpaca'
cache_dir = './cache'

model, tokenizer = load_llama_model_lora(model_uri, lora_uri, cache_dir=cache_dir, groupsize=32, bits=2)

model.eval()

prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{args.prompt}\n\n### Response:\n"

batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
batch = {k: v.cuda() for k, v in batch.items()}
model.cuda()

for i in range(10):
    print("Generating tokens, this may take a few seconds...")
    with torch.no_grad():
        generated = model.generate(
            inputs=batch["input_ids"],
            do_sample=True, use_cache=True,
            repetition_penalty=1.2,
            max_new_tokens=256,
            temperature=0.6,
            top_p=0.4,
            top_k=0,
            return_dict_in_generate=True,
            output_attentions=False,
            output_hidden_states=False,
            output_scores=False
        )
    result_text = tokenizer.decode(generated['sequences'].cpu().tolist()[0])
    # detect if model "stop" token
    if "```" in result_text:
        result_text = result_text.split("```")[0]
    print(result_text + "\n")
