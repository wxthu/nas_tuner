import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

class LongAlignChatTemplateDataset(Dataset):
    def __init__(self, tokenizer, max_length=16384):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.raw_data = load_dataset("THUDM/LongAlign-10k", split="train")
        
        self.tokenizer.pad_token = tokenizer.eos_token

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        example = self.raw_data[idx]
        messages = example["messages"]

        # 使用 chat template 自动构建 prompt + response 格式
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        encoded = self.tokenizer(encoded, return_tensors="pt", padding="max_length", max_length=self.max_length, truncation=True)
        input_ids = encoded["input_ids"]
        attn_mask = encoded["attention_mask"]
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100 # ignore the loss of last token
        return ((input_ids, attn_mask), labels)


class LongAlignForInferenceDataset(Dataset):
    def __init__(self, tokenizer, max_length=16384):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.raw_data = load_dataset("THUDM/LongAlign-10k", split="train")
        
        self.tokenizer.pad_token = tokenizer.eos_token

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        example = self.raw_data[idx]
        messages = example["messages"]

        # 使用 chat template 自动构建 prompt + response 格式
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        encoded = self.tokenizer(encoded, return_tensors="pt", padding="max_length", max_length=self.max_length, truncation=True)
        return encoded

