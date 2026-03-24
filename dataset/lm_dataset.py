import json

from torch.utils.data import Dataset
import torch
import os
import random
from datasets import load_dataset

# 禁用 HuggingFace tokenizer 的多进程并行，避免在 DataLoader 多进程环境中产生死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"

## 以上内容不用管 直接复制粘贴

class PretrainDataset(Dataset):
    # init
    def __init__(self, data_path, tokenizer, max_length = 512):
        super().__init__()
        
        self.tokenizer = tokenizer
        # 单条样本的最大序列 (最多多少token)
        self.max_length = max_length
        # 用 HuggingFace datasets 的惰性加载，避免一次性读入大文件
        self.samples = load_dataset("json", data_files = data_path, split = "train")
        
    # __len__
    # 得到数据集有多少条记录（多少行）
    def __len__(self):
        return len(self.samples)
    # __getitem__
    # 我们拿到的是json的每一行
    # tokenizer会把文本转化为input_ids
    # 需要加上EOS, BOS, 以及PAD填充
    # 需要自行编写labels, 防止PAD参与loss运算
    # 需要编写attention_mask, 告诉模型哪些有效哪些是PAD
    # 要输出input_ids, attention_mask, labels
    def __getitem__(self, index):
        sample = self.samples[index]
        
        tokens = self.tokenizer(
            str(sample["text"]), # 把sample中text部分转化为字符串
            add_special_tokens = False, # 不自动添加特殊token
            max_length = self.max_length - 2, # 两个位置给BOS和EOS
            truncation = True, # 如果长度超过max 自动剪切
        ).input_ids
        
        # 前后添加BOS和EOS
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        # 补足PAD
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length-len(tokens))
        
        # 转化为张量
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        # labels: 让PAD不参与loss计算
        labels = input_ids.clone()
        # 将PAD位置的标签设为100
        # crossloss会自动忽略 -100的部分
        # 前面自动实现了自回归部分
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # attention_mask: 让PAD不参与attention计算
        # 非PAD设置为1 PAD设置为0
        # .long 把True/False转化为1/0
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        # 输出
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }