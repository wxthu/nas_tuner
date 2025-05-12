from transformers import AutoTokenizer, TrainingArguments, Trainer, TrainerCallback
from llama_nsa import NsaLlamaForCausalLM, load_custom_weights_and_freeze
from transformers import LlamaForCausalLM, LlamaConfig
from dataloader import LongAlignChatTemplateDataset

import torch
import pandas as pd
import csv
import os

class LossLoggerCallback(TrainerCallback):
    def __init__(self, csv_path="loss_log.csv"):
        self.csv_path = csv_path
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "loss"])

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([state.global_step, logs["loss"]])


def train(
    model_id="meta-llama/Llama-3.1-8B-Instruct",
    max_length=8192,
    memory_fraction=0.95,
):
    
    torch.cuda.set_per_process_memory_fraction(memory_fraction)
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    config = LlamaConfig.from_pretrained(model_id)
    pretrained_llama_model = LlamaForCausalLM.from_pretrained(model_id)
    pretrained_state_dict = pretrained_llama_model.state_dict()
    model = NsaLlamaForCausalLM(config)
    model = load_custom_weights_and_freeze(model, pretrained_state_dict)

    # print(f"xxxxxxxxxx {sum(p.numel() for p in model.parameters() if p.requires_grad)} xxxxxxx")
    # Load dataset
    dataset = LongAlignChatTemplateDataset(tokenizer, max_length=max_length)

    training_args = TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=1,
        num_train_epochs=10,
        learning_rate=2e-5,
        save_strategy="epoch",
        logging_steps=10,
        save_total_limit=5,
        deepspeed="ds_config.json",  # 跟配置文件一致
        fp16=True,
        remove_unused_columns=False,
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        callbacks=[LossLoggerCallback()],
    )

    trainer.train()

    # log_history = trainer.state.log_history
    # df = pd.DataFrame(log_history)
    # df.to_csv("loss_log.csv", index=False)

if __name__ == "__main__":
    train()