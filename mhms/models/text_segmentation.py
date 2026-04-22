import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class HierarchicalBERT(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased', num_article_layers=12, hidden_size=768, num_heads=12):
        super(HierarchicalBERT, self).__init__()
        self.sentence_encoder = BertModel.from_pretrained(pretrained_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads,
            dim_feedforward=hidden_size * 4, dropout=0.1, batch_first=True
        )
        self.article_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_article_layers)
        self.segmentation_head = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask=None):
        B, S, W = input_ids.shape
        flat_input = input_ids.view(-1, W)
        flat_mask = attention_mask.view(-1, W) if attention_mask is not None else None

        outputs = self.sentence_encoder(input_ids=flat_input, attention_mask=flat_mask)
        sentence_features = outputs.pooler_output.view(B, S, -1)
        article_features = self.article_encoder(sentence_features)

        logits = self.segmentation_head(article_features).squeeze(-1)
        return self.sigmoid(logits), article_features
