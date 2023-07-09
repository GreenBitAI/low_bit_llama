import sys

import torch

from model import load_llama_model

if not torch.cuda.is_available():
    print("CUDA is needed to run the model.")
    sys.exit(0)

model_uri = 'GreenBitAI/LLaMA-7B-2bit'
cache_dir = './cache'

model, tokenizer = load_llama_model(model_uri, cache_dir=cache_dir, half=True, groupsize=32, bits=2)
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
