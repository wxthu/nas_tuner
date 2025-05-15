import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

class LongAlignChatTemplateDataset(Dataset):
    def __init__(self, tokenizer, max_length=8192):
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

        encoded = self.tokenizer(encoded, return_tensors="pt", padding="max_length", truncation=True)
        input_ids = encoded["input_ids"]
        attn_mask = encoded["attention_mask"]
        return ((input_ids, attn_mask), input_ids.clone()) # (input, label)


class LongAlignFlatDataset(Dataset):
    def __init__(self, tokenizer, max_length=8192):
        self.raw_data = load_dataset("THUDM/LongAlign-10k", split="train")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.preprocess()

    def preprocess(self):
        all_texts = []
        for example in self.raw_data:
            messages = example["messages"]
            # 拼接为一个长文本
            text = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "user":
                    text += f"\n### Instruction:\n{content}\n"
                elif role == "assistant":
                    text += f"\n### Response:\n{content}\n"
            all_texts.append(text.strip())
        return all_texts

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # return {
        #     "input_ids": input_ids,
        #     "attention_mask": attention_mask,
        #     "labels": input_ids.clone()
        # }
        return (input_ids, attention_mask, input_ids.clone())

if __name__ == '__main__':
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  # 必须设置pad_token

    dataset = LongAlignFlatDataset(tokenizer, max_length=8192)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

    # 查看一个样本
    for batch in dataloader:
        print(batch["input_ids"].shape)
        print(tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True)[:500])
        break
