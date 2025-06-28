import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nsa_utils.dataloader import LongAlignForInferenceDataset
from modeling_minicpm import MiniCPMForCausalLM

from typing import Any, Dict, List, Optional, Tuple, Union
import os

class WrappedMiniCPM(MiniCPMForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ):
        return super().forward(
			input_ids=input_ids,
			attention_mask=attention_mask,
			position_ids=position_ids,
			past_key_values=past_key_values,
			inputs_embeds=inputs_embeds,
			labels=labels,
			use_cache=use_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
			logits_to_keep=logits_to_keep,
			**kwargs
		)
        
    

if __name__ == "__main__":
	torch.manual_seed(0)
	path = 'openbmb/MiniCPM4-8B'
	max_length = 165536
	device = "cuda"
	tokenizer = AutoTokenizer.from_pretrained(path)
	# model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True)
	model = WrappedMiniCPM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True)

	dataset = LongAlignForInferenceDataset(tokenizer=tokenizer, max_length=max_length)
	
	count = 0
	for prompt in dataset:
		prompt = prompt.to(device)
		print(f"xxxxxxxxxxxxxxxx {prompt['input_ids'].shape} xxxxxxxxxxxxxxxxxxxx")
		outputs = model.generate(**prompt, max_new_tokens=100, do_sample=False, top_p=0.7, temperature=0.0)
		output_token_ids = [
			outputs[i][len(prompt[i]):] for i in range(len(prompt['input_ids']))
		]

		responses = tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)[0]
		print("==============================")
		print(responses)
		print("==============================")
		print(outputs.shape, len(output_token_ids))

		os.mkdir(f"sample_{count}_seqlen_{prompt['input_ids'].shape[1]}")
		os.system(f"mv layer_* sample_{count}_seqlen_{prompt['input_ids'].shape[1]}/")
		count += 1
		if count >= 10: break