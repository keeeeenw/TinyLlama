import torch
import transformers 
import os

from pathlib import Path
from transformers import LlamaTokenizer
from transformers import LlamaForCausalLM

out_dir = "/root/TinyLlama/out/pretrained"
model_path = os.path.join(out_dir, "pytorch_model.bin")
state_dict = torch.load(model_path, map_location=torch.device("cpu"))
tokenizer_path = Path("/root/TinyLlama/llama-tokenizer/")
model = LlamaForCausalLM.from_pretrained(
    out_dir, local_files_only=True, state_dict=state_dict
)
tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
# tokenizer = Tokenizer(tokenizer_path)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
    tokenizer=tokenizer
)

prompt = "Who are you? What do you know about the world?"
formatted_prompt = (
    f"### Human: {prompt} ### Assistant:"
)

sequences = pipeline(
    formatted_prompt,
    do_sample = True,
    top_k = 5,
    top_p = 0.9,
    num_return_sequences = 1,
    repetition_penalty = 1.5,
    max_new_tokens = 512,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")

prompt = "Tell me about the history of the United States"
formatted_prompt = (
    f"{prompt}"
)

sequences = pipeline(
    formatted_prompt,
    do_sample = True,
    top_k = 5,
    top_p = 0.9,
    num_return_sequences = 1,
    repetition_penalty = 1.5,
    max_new_tokens = 512,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
