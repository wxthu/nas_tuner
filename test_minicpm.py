from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
torch.manual_seed(0)

path = 'openbmb/MiniCPM4-0.5B'
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True)

# User can directly use the chat interface
promt = "".join(["Write an article about Artificial Intelligence." for _ in range(5000)])
responds, history = model.chat(tokenizer, promt, temperature=0.7, top_p=0.7, max_length=40000)
print(responds)

# User can also use the generate interface
# messages = [
#     {"role": "user", "content": "Write an article about Artificial Intelligence."},
# ]
# prompt_text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True,
# )
# model_inputs = tokenizer([prompt_text], return_tensors="pt").to(device)

# model_outputs = model.generate(
#     **model_inputs,
#     max_new_tokens=1024,
#     top_p=0.7,
#     temperature=0.7
# )
# output_token_ids = [
#     model_outputs[i][len(model_inputs[i]):] for i in range(len(model_inputs['input_ids']))
# ]

# responses = tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)[0]
# print(responses)
