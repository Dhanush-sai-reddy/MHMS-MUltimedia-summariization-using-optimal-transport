import os
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer

class CNNMultimodalDataset(Dataset):
    def __init__(self, data_dir="cnn_data", max_sentences=20, max_words=64, visual_dim=1024):
        """
        Args:
            data_dir: Path to cnn_data directory.
            max_sentences: Maximum sentences per article.
            max_words: Maximum words per sentence.
            visual_dim: The mock video feature dimension mapping to VTS.
        """
        self.data_dir = data_dir
        self.max_sentences = max_sentences
        self.max_words = max_words
        self.visual_dim = visual_dim
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Read the global labels
        label_file = os.path.join(data_dir, 'label.txt')
        self.labels = []
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip().replace('[', '').replace(']', '')
                    if not line:
                        self.labels.append([])
                    else:
                        nums = [float(x) for x in line.split()]
                        self.labels.append(nums)
        
        # Find all valid directories
        self.samples = []
        for d in os.listdir(data_dir):
            dir_path = os.path.join(data_dir, d)
            if os.path.isdir(dir_path) and d.isdigit():
                # We map folder ID as index to labels if possible 
                # (adjust logic based on actual index mapping if it differs)
                idx = int(d)
                self.samples.append({
                    "id": idx,
                    "path": dir_path
                })
        
        self.samples.sort(key=lambda x: x["id"])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample_path = sample["path"]
        sample_id = sample["id"]
        
        # 1. Read Text
        artitle_sec_path = os.path.join(sample_path, 'artitle_section.txt')
        sentences = []
        if os.path.exists(artitle_sec_path):
            with open(artitle_sec_path, 'r', encoding='utf-8') as f:
                sentences = [s.strip() for s in f.readlines() if s.strip()]
                
        # Truncate or Pad sentences
        actual_sents = len(sentences)
        sentences = sentences[:self.max_sentences]
        
        input_ids = torch.zeros((self.max_sentences, self.max_words), dtype=torch.long)
        attention_mask = torch.zeros((self.max_sentences, self.max_words), dtype=torch.long)
        
        for i, sent in enumerate(sentences):
            encoded = self.tokenizer(
                sent, 
                max_length=self.max_words, 
                truncation=True, 
                padding='max_length', 
                return_tensors='pt'
            )
            input_ids[i] = encoded['input_ids'].squeeze(0)
            attention_mask[i] = encoded['attention_mask'].squeeze(0)
            
        # 2. Read Labels (Text Extractive Labels)
        # Using sample_id cautiously. If label length is 444 and IDs go up to 262, 
        # let's assume index in labels array is (sample_id - 1).
        label_idx = min(sample_id - 1, len(self.labels) - 1)
        target_labels = self.labels[label_idx] if label_idx >= 0 else []
        
        # Pad target labels to max_sentences
        summ_labels = torch.zeros(self.max_sentences, dtype=torch.float)
        for i, val in enumerate(target_labels[:self.max_sentences]):
            summ_labels[i] = val
            
        # 3. Handle Video Features (Mocking extraction from .ts files)
        video_dir = os.path.join(sample_path, 'video')
        num_shots = 10 # default fallback
        if os.path.exists(video_dir):
            ts_files = [f for f in os.listdir(video_dir) if f.endswith('.ts')]
            num_shots = max(10, len(ts_files)) # Ensure at least enough shots for VTS window
            
        # We simulate CNN extracted temporal features for these shots
        # In a real pipeline, torchvision or ffmpeg would decode the .ts and pass through ResNet
        video_features = torch.randn((num_shots, self.visual_dim), dtype=torch.float)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "video_features": video_features,
            "summ_labels": summ_labels,
            "num_sentences": min(actual_sents, self.max_sentences),
            "num_shots": num_shots
        }

if __name__ == "__main__":
    # Test dataset instantiation
    ds = CNNMultimodalDataset(data_dir="../cnn_data")
    if len(ds) > 0:
        sample = ds[0]
        print(f"Loaded Sample Shapes:")
        print(f"- input_ids: {sample['input_ids'].shape}")
        print(f"- video_features: {sample['video_features'].shape}")
        print(f"- summ_labels: {sample['summ_labels'].shape}")
