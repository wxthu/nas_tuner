import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nsa_utils.dataloader import LongAlignForInferenceDataset

if __name__ == "__main__":
	torch.manual_seed(0)
	path = 'openbmb/MiniCPM4-8B'
	max_length = 30000
	device = "cuda"
	tokenizer = AutoTokenizer.from_pretrained(path)
	model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True)

	dataset = LongAlignForInferenceDataset(tokenizer=tokenizer, max_length=max_length)
	
	for prompt in dataset:
		prompt = prompt.to(device)
		outputs = model.generate(**prompt, max_new_tokens=100, do_sample=False, top_p=0.7, temperature=0.0)
		output_token_ids = [
			outputs[i][len(prompt[i]):] for i in range(len(prompt['input_ids']))
		]

		responses = tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)[0]
		print("==============================")
		print(responses)
		print("==============================")
		print(outputs.shape, len(output_token_ids))
		break