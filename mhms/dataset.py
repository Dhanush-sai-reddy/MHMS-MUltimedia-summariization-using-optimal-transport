import os
import torch
import numpy as np
from torch.utils.data import Dataset

class CNNMultimodalDataset(Dataset):
    def __init__(self, data_dir="cnn_data", max_sentences=20, max_words=64, visual_dim=1024, max_shots=20):
        """
        Args:
            data_dir: Path to cnn_data directory.
            max_sentences: Maximum sentences per article.
            max_words: Maximum words per sentence.
            visual_dim: The mock video feature dimension mapping to VTS.
            max_shots: Maximum shots per video for padding.
        """
        self.data_dir = data_dir
        self.max_sentences = max_sentences
        self.max_words = max_words
        self.visual_dim = visual_dim
        self.max_shots = max_shots
        
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
        
        # Zero-dependency tokenizer since we already have the summaries!
        for i, sent in enumerate(sentences):
            words = sent.split()[:self.max_words]
            for j, w in enumerate(words):
                # Basic token index simulation to totally avoid heavy BERT tokenizers 
                input_ids[i, j] = (abs(hash(w)) % 30000) + 1
                attention_mask[i, j] = 1
            
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
        actual_shots = 10 # default fallback
        if os.path.exists(video_dir):
            ts_files = [f for f in os.listdir(video_dir) if f.endswith('.ts')]
            actual_shots = max(1, len(ts_files)) 
            
        # We simulate CNN extracted temporal features for these shots padded to max_shots
        video_features = torch.zeros((self.max_shots, self.visual_dim), dtype=torch.float)
        
        # Populate up to max_shots
        populated_shots = min(actual_shots, self.max_shots)
        mock_real_features = torch.randn((populated_shots, self.visual_dim), dtype=torch.float)
        video_features[:populated_shots, :] = mock_real_features

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "video_features": video_features,
            "summ_labels": summ_labels,
            "num_sentences": min(actual_sents, self.max_sentences),
            "num_shots": populated_shots
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
