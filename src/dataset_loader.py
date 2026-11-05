import pandas as pd
import torch
from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len=128):
       
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_map = {"Sarcasm": 0, "Idiom": 1, "Irony": 2, "Metaphor": 3, "Hyperbole": 4, "Slang/Idiom": 1}
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        row = self.data.iloc[index]
        text = str(row["text"])
        context = str(row["cultural_context"])
        
      
        input_text = f"[CONTEXT: {context}] {text}"
        inputs = self.tokenizer(
            input_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
      
        torch.manual_seed(index)
        visual_features = torch.randn(512) 
        
        label = self.label_map.get(row["label"], 0)
        
        return {
            "input_ids": inputs["input_ids"].flatten(),
            "attention_mask": inputs["attention_mask"].flatten(),
            "visual_features": visual_features,
            "labels": torch.tensor(label, dtype=torch.long)
        }
