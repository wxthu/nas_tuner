
import csv
from tqdm import tqdm
import deepspeed
import torch
import torch.distributed as dist
import torch.utils.data.dataloader

from nsa_utils.qwen_nsa import NsaQwen2ForCausalLM, load_custom_weights_and_freeze, build_pipeline_qwen2
from transformers import Qwen2ForCausalLM, Qwen2Config, AutoTokenizer
from nsa_utils.dataloader import LongAlignChatTemplateDataset

def setup_distributed():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    deepspeed.init_distributed()
    print(f"[Rank {dist.get_rank()}] World Size: {dist.get_world_size()}")

def load_model_and_tokenizer(model_id: str, num_stages: int):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    # Load model
    config = Qwen2Config.from_pretrained(model_id)
    pretrained_llama_model = Qwen2ForCausalLM.from_pretrained(model_id)
    pretrained_state_dict = pretrained_llama_model.state_dict()
    model = NsaQwen2ForCausalLM(config)
    model = load_custom_weights_and_freeze(model, pretrained_state_dict)
    pipeline_model = build_pipeline_qwen2(model, num_stages=num_stages)
    return tokenizer, pipeline_model

def train(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    num_stages=4,
    max_length=30000,
):
    
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
    
    print(f"[Rank {dist.get_rank()}] Start training loop...")
    with open("training_loss.csv", mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Step", "Loss"])
        
        total_steps = 100000
        with tqdm(total=total_steps, desc="Training Progress", unit="step") as pbar:
            for step in range(1, total_steps + 1):
                loss = model_engine.train_batch()
                if step % 10 == 0:
                    print(f"======== loss : {loss:.8f} ======")
                    writer.writerow([step // 10, loss])
                    pbar.update(1)
                    
                if step % 200 == 0:
                    model_engine.save_checkpoint("./my_model")
                

if __name__ == "__main__":
    train()