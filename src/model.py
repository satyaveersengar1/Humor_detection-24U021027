import torch
import torch.nn as nn
from transformers import AutoModel

class Model(nn.Module):
    def __init__(self, num_classes=5):
        super(Model, self).__init__()
        self.text_encoder = AutoModel.from_pretrained("google/muril-base-cased")
        
        self.fusion = nn.Linear(768 + 512, 512)
        self.relu = nn.ReLU()
        
        self.classifier = nn.Linear(512, num_classes)
        
    def forward(self, input_ids, attention_mask, visual_features):
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = text_outputs.last_hidden_state[:, 0, :] # [CLS] token
        
        fused = torch.cat((cls_token, visual_features), dim=1)
        fused = self.relu(self.fusion(fused))
        
        logits = self.classifier(fused)
        return logits
