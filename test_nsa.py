import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaAttention

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'native_sparse_attention_pytorch')))
from native_sparse_attention_pytorch.native_sparse_attention import SparseAttention
# from native_sparse_attention_pytorch.native_sparse_attention_pytorch.native_sparse_attention import SparseAttention


def test_llama_model():
	# 加载模型和分词器
	print("正在加载Llama-3.1-8B-Instruct模型...")
	model_id = "meta-llama/Llama-3.1-8B-Instruct"

	tokenizer = AutoTokenizer.from_pretrained(model_id)

	model = AutoModelForCausalLM.from_pretrained(
		model_id,
		torch_dtype=torch.float16,  # 使用半精度加载以节省显存
		# device_map="auto"           # 自动决定模型加载到哪个设备
	)

	def replace_attention_with_nsa(model):
		attn = SparseAttention(
			dim = 4096,
			dim_head = 128,
			heads = 32,
			sliding_window_size = 512,
			compress_block_size = 32,
			compress_block_sliding_stride = 16,
			selection_block_size = 32,
			num_selected_blocks = 16,
			kv_heads = 8
		).to(model.device).half()
		"""替换模型中的所有注意力层为NSA"""
		for name, module in model.named_modules():
			if isinstance(module, LlamaAttention):
				parent_name = '.'.join(name.split('.')[:-1])
				child_name = name.split('.')[-1]
				
				if parent_name:
					parent = model.get_submodule(parent_name)
					layer_idx = None
					if hasattr(parent, 'layer_idx'):
						layer_idx = parent.layer_idx
					
					# 获取模块的配置
					config = module.config
					# 创建NSA层并替换
					setattr(parent, child_name, attn)
		
		return model

	replace_attention_with_nsa(model)
	# for name, module in model.named_modules():
	# 	print(f"{name}__{module}_##-->{isinstance(module, LlamaAttention)}")
	# return
	# 准备提示词
	prompt = "请简要解释人工智能是什么。"

	# 添加模型需要的格式（系统提示和用户提示）
	messages = [
		{"role": "system", "content": "你是一个有帮助的AI助手。"},
		{"role": "user", "content": prompt}
	]
    
	# 将消息转换为模型需要的格式
	prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
	print("\n输入提示词:")
	print(prompt)

	# 编码输入
	inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

	# 生成回答
	print("\n正在生成回答...\n")
	with torch.no_grad():
		outputs = model.generate(
			**inputs,
			max_new_tokens=1,
			temperature=0.7,
			top_p=0.9,
			do_sample=True
		)
    
	# 解码并打印输出
	response = tokenizer.decode(outputs[0], skip_special_tokens=True)

	# 提取助手的回答部分
	assistant_response = response.split("ASSISTANT: ")[-1].strip()
	print("模型回答:")
	print(assistant_response)

if __name__ == "__main__":
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("警告: 没有检测到GPU，将使用CPU运行（这会非常慢）")
    
    test_llama_model()