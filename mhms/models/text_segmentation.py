import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class HierarchicalBERT(nn.Module):
    """
    Textual Segmentation Module as described in MHMS paper (Section 3.3).
    It uses a two-level Transformer encoder:
    1. Sentence-level encoder (BERT) outputs [CLS] tokens for each sentence.
    2. Article-level encoder (Transformer) relates different sentences through cross-attention
       and predicts a semantic segmentation boundary score.
    """
    def __init__(self, pretrained_model_name='bert-base-uncased', num_article_layers=12, hidden_size=768, num_heads=12):
        super(HierarchicalBERT, self).__init__()
        
        # 1. First-level Encoder (Sentence Level)
        # The paper mentions reducing to 64 word pieces per sentence and 128 sentences per doc.
        # It uses the BERT_BASE checkpoint (12 attention heads, 768 dim).
        self.sentence_encoder = BertModel.from_pretrained(pretrained_model_name)
        
        # We also create a tokenizer for ease of inference later
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        
        # 2. Second-level Encoder (Article Level)
        # An encoder layer to capture the representation of the sequence of sequence [CLS] tokens.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=num_heads, 
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True
        )
        self.article_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_article_layers)
        
        # Final linear layer to output sequence boundary prediction (1 for boundary, 0 for not)
        self.segmentation_head = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: (Batch Size, Num Sentences, Max Words per Seq)
            attention_mask: (Batch Size, Num Sentences, Max Words per Seq)
            
        Returns:
            segment_probs: (Batch Size, Num Sentences) Probability of boundary.
            article_features: (Batch Size, Num Sentences, Hidden Size) Cross-attended features.
        """
        B, S, W = input_ids.shape
        
        # Flatten batch and sentences to run through sentence encoder together
        # New shape: (B * S, W)
        flat_input = input_ids.view(-1, W)
        flat_mask = attention_mask.view(-1, W) if attention_mask is not None else None
        
        # Pass through sentence encoder (BERT)
        outputs = self.sentence_encoder(input_ids=flat_input, attention_mask=flat_mask)
        
        # Extract the [CLS] tokens from the pooler output
        # CLS dimension: (B * S, Hidden Size)
        cls_tokens = outputs.pooler_output
        
        # Reshape back to sequence of sentences: (B, S, Hidden Size)
        sentence_features = cls_tokens.view(B, S, -1)
        
        # Pass through article-level encoder
        article_features = self.article_encoder(sentence_features)
        
        # Predict segmentation boundaries
        logits = self.segmentation_head(article_features).squeeze(-1)  # (B, S)
        segment_probs = self.sigmoid(logits)
        
        return segment_probs, article_features
