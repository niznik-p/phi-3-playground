from sys import platform

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

MAC_PHI_PATH = '/Users/niznik/GitHub/text-generation-webui/models/microsoft_Phi-3-mini-128k-instruct/'
LINUX_PHI_PATH = '/scratch/network/niznik/HF/Phi-3-mini-128k-instruct/'

# torch.random.manual_seed(0)

if platform == 'darwin':
    device = torch.device("cpu")
    model = AutoModelForCausalLM.from_pretrained(
        MAC_PHI_PATH,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MAC_PHI_PATH
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        LINUX_PHI_PATH,
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        LINUX_PHI_PATH
    )

messages = [
    {"role": "user", "content": "You are a happy mallard and like to tell people random facts about yourself."}
]

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

generation_args = {
    "max_new_tokens": 250,
    "return_full_text": False,
    "temperature": 1.5,
    "do_sample": True,
}

output = pipe(messages, **generation_args)
print(output[0]['generated_text'])
