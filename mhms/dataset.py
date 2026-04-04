import os
import torch
import numpy as np
from torch.utils.data import Dataset

class CNNMultimodalDataset(Dataset):
    def __init__(self, data_dir="cnn_data", embeddings_dir="embeddings", max_sentences=20, visual_dim=2048, text_dim=768, max_shots=20):
        """
        Loads pre-extracted BERT and ResNet embeddings.
        Args:
            data_dir: Path to cnn_data directory for labels.
            embeddings_dir: Path to the generated .npy embeddings.
            max_sentences: Maximum sentences per article (temporal steps).
            visual_dim: The video feature dimension (2048 for ResNet50).
            text_dim: Text feature dimension (768 for BERT).
            max_shots: Maximum shots per video for padding.
        """
        self.max_sentences = max_sentences
        self.visual_dim = visual_dim
        self.text_dim = text_dim
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
        
        self.samples = []
        
        text_dir = os.path.join(embeddings_dir, "text")
        visual_dir = os.path.join(embeddings_dir, "visual")
        
        if not os.path.exists(text_dir) or not os.path.exists(visual_dir):
            print("Warning: Embeddings directory not found. Please run embedding_pipeline.py first.")
            return

        # Find cases that have BOTH text and visual embeddings
        for f in os.listdir(text_dir):
            if f.endswith(".npy"):
                case_id_str = f.replace("case_", "").replace(".npy", "")
                if case_id_str.isdigit():
                    case_id = int(case_id_str)
                    vis_path = os.path.join(visual_dir, f"case_{case_id}.npy")
                    if os.path.exists(vis_path):
                        self.samples.append({
                            "id": case_id,
                            "text_path": os.path.join(text_dir, f),
                            "visual_path": vis_path
                        })
                        
        self.samples.sort(key=lambda x: x["id"])
        print(f"Dataset initialized with {len(self.samples)} valid multimodal cases.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample_id = sample["id"]
        
        # 1. Load Text Embeddings (BERT 768-dim)
        text_npy = np.load(sample["text_path"])
        actual_sents = text_npy.shape[0]
        
        text_features = torch.zeros((self.max_sentences, self.text_dim), dtype=torch.float)
        populated_sents = min(actual_sents, self.max_sentences)
        text_features[:populated_sents, :] = torch.from_numpy(text_npy[:populated_sents, :])
        
        text_mask = torch.zeros(self.max_sentences, dtype=torch.long)
        text_mask[:populated_sents] = 1
        
        # 2. Add Text Extractive Labels
        label_idx = min(sample_id - 1, len(self.labels) - 1)
        target_labels = self.labels[label_idx] if label_idx >= 0 else []
        
        summ_labels = torch.zeros(self.max_sentences, dtype=torch.float)
        for i, val in enumerate(target_labels[:self.max_sentences]):
            summ_labels[i] = val
            
        # 3. Load Visual Embeddings (ResNet 2048-dim)
        vis_npy = np.load(sample["visual_path"])
        actual_shots = vis_npy.shape[0]
        
        video_features = torch.zeros((self.max_shots, self.visual_dim), dtype=torch.float)
        populated_shots = min(actual_shots, self.max_shots)
        video_features[:populated_shots, :] = torch.from_numpy(vis_npy[:populated_shots, :])
        
        video_mask = torch.zeros(self.max_shots, dtype=torch.long)
        video_mask[:populated_shots] = 1

        return {
            "text_features": text_features,      # (Max_Sents, 768)
            "text_mask": text_mask,              # (Max_Sents)
            "video_features": video_features,    # (Max_Shots, 2048)
            "video_mask": video_mask,            # (Max_Shots)
            "summ_labels": summ_labels,          # (Max_Sents)
            "num_sentences": populated_sents,
            "num_shots": populated_shots
        }
