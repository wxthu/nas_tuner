import os
import deepspeed
import torch
import torch.distributed as dist
import torch.utils.data.dataloader

from llama_nsa import NsaLlamaForCausalLM, load_custom_weights_and_freeze, build_pipeline_llama
from transformers import LlamaForCausalLM, LlamaConfig, AutoTokenizer
from dataloader import LongAlignChatTemplateDataset

def setup_distributed():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    deepspeed.init_distributed()
    print(f"[Rank {dist.get_rank()}] World Size: {dist.get_world_size()}")

def load_model_and_tokenizer(model_id: str, num_stages: int):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    # Load model
    config = LlamaConfig.from_pretrained(model_id)
    pretrained_llama_model = LlamaForCausalLM.from_pretrained(model_id)
    pretrained_state_dict = pretrained_llama_model.state_dict()
    model = NsaLlamaForCausalLM(config)
    # for name, param in model.named_parameters():
    #     param.requires_grad = "self_attn" in name
    model = load_custom_weights_and_freeze(model, pretrained_state_dict)
    pipeline_model = build_pipeline_llama(model, num_stages=num_stages)
    return tokenizer, pipeline_model

def train(
    model_id="meta-llama/Llama-3.1-8B-Instruct",
    num_stages=4,
    max_length=8192,
    memory_fraction=1.0,
):
    
    # torch.cuda.set_per_process_memory_fraction(memory_fraction)
    setup_distributed()
    tokenizer, pipeline_model = load_model_and_tokenizer(model_id=model_id, num_stages=num_stages)

    # Load dataset
    dataset = LongAlignChatTemplateDataset(tokenizer, max_length=max_length)

    model_engine, _,  _,  _ = deepspeed.initialize(
        model=pipeline_model,
        model_parameters=[p for p in pipeline_model.parameters() if p.requires_grad],
        config="ds_config.json",
        training_data=dataset
    )
    
    # data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    print(f"[Rank {dist.get_rank()}] Start training loop...")
    loss = model_engine.train_batch()
    # for epoch in range(1):
    #     for step, batch in enumerate(data_loader):
    #         input_ids = batch["input_ids"].to(model_engine.device)
    #         attention_mask = batch["attention_mask"].to(model_engine.device)
    #         labels = batch["labels"].to(model_engine.device)
            
    #         loss = model_engine(input_ids, attention_mask=attention_mask, labels=labels)
    #         model_engine.backward(loss)
    #         model_engine.step()

    #         if step % 10 == 0:
    #             print(f"[Epoch {epoch} Step {step}] Loss: {loss.item():.4f}")
                

if __name__ == "__main__":
    train()