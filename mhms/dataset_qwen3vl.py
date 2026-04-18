"""
MHMS Dataset Loader for Qwen3 VL Unified Embeddings
====================================================
Loads pre-extracted Qwen3 VL embeddings for both text and video.
Both modalities are in the SAME embedding space (e.g., 3584-dim).

Key differences from original dataset.py:
- Unified embedding dimension for both text and visual
- No separate BERT tokenizer needed (Qwen handles tokenization internally)
- Better cross-modal alignment due to unified pretraining
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
import re


class CNNMultimodalDatasetQwen3VL(Dataset):
    def __init__(self, data_dir="cnn_data", embeddings_dir="embeddings_qwen3vl", 
                 max_sentences=20, max_shots=20, embedding_dim=3584):
        """
        Loads pre-extracted Qwen3 VL embeddings.
        
        Args:
            data_dir: Path to cnn_data directory for labels and raw text.
            embeddings_dir: Path to Qwen3 VL embeddings.
            max_sentences: Maximum sentences per article (temporal steps).
            max_shots: Maximum shots per video for padding.
            embedding_dim: Qwen3 VL hidden dimension (3584 for 8B, 2048 for 2B).
        """
        self.max_sentences = max_sentences
        self.max_shots = max_shots
        self.embedding_dim = embedding_dim
        
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
            raise ValueError(
                f"Qwen3 VL embeddings not found at '{embeddings_dir}'.\n"
                f"Please run: python embedding_pipeline_qwen3vl.py"
            )

        # Load manifest to get embedding dimension
        manifest_path = os.path.join(embeddings_dir, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
                saved_dim = manifest.get("embedding_dim", embedding_dim)
                if saved_dim != embedding_dim:
                    print(f"Warning: Using embedding_dim={saved_dim} from manifest (not {embedding_dim})")
                    self.embedding_dim = saved_dim

        # Find cases that have BOTH text and visual embeddings
        for f in os.listdir(text_dir):
            if f.endswith(".npy"):
                case_id_str = f.replace("case_", "").replace(".npy", "")
                if case_id_str.isdigit():
                    case_id = int(case_id_str)
                    vis_path = os.path.join(visual_dir, f"case_{case_id}.npy")
                    # Also check if raw text exists (for sentence-level labels)
                    raw_text_path = os.path.join(data_dir, str(case_id), "artitle_section.txt")
                    if os.path.exists(vis_path):
                        self.samples.append({
                            "id": case_id,
                            "text_path": os.path.join(text_dir, f),
                            "visual_path": vis_path,
                            "raw_text_path": raw_text_path
                        })
                        
        self.samples.sort(key=lambda x: x["id"])
        print(f"Qwen3VL Dataset initialized with {len(self.samples)} valid multimodal cases.")
        print(f"  Embedding dimension: {self.embedding_dim}")

    def __len__(self):
        return len(self.samples)

    def _split_into_sentences(self, text):
        """Split text into sentences using simple heuristics."""
        # Replace common abbreviations
        text = text.replace('U.S.', 'US')
        text = text.replace('Dr.', 'Dr')
        text = text.replace('Mr.', 'Mr')
        text = text.replace('Mrs.', 'Mrs')
        text = text.replace('Ms.', 'Ms')
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample_id = sample["id"]
        
        # 1. Load Text Embeddings (Qwen3 VL unified dim)
        text_npy = np.load(sample["text_path"])
        actual_sents = text_npy.shape[0]
        
        text_features = torch.zeros((self.max_sentences, self.embedding_dim), dtype=torch.float)
        populated_sents = min(actual_sents, self.max_sentences)
        text_features[:populated_sents, :] = torch.from_numpy(text_npy[:populated_sents, :])
        
        text_mask = torch.zeros(self.max_sentences, dtype=torch.long)
        text_mask[:populated_sents] = 1
        
        # 2. Load Visual Embeddings (Qwen3 VL - same dim as text!)
        vis_npy = np.load(sample["visual_path"])
        actual_shots = vis_npy.shape[0]
        
        video_features = torch.zeros((self.max_shots, self.embedding_dim), dtype=torch.float)
        populated_shots = min(actual_shots, self.max_shots)
        video_features[:populated_shots, :] = torch.from_numpy(vis_npy[:populated_shots, :])
        
        video_mask = torch.zeros(self.max_shots, dtype=torch.long)
        video_mask[:populated_shots] = 1

        # 3. Load Text Extractive Labels
        label_idx = min(sample_id - 1, len(self.labels) - 1)
        target_labels = self.labels[label_idx] if label_idx >= 0 else []
        
        summ_labels = torch.zeros(self.max_sentences, dtype=torch.float)
        for i, val in enumerate(target_labels[:self.max_sentences]):
            summ_labels[i] = val

        return {
            "text_features": text_features,      # (Max_Sents, EMBEDDING_DIM)
            "text_mask": text_mask,              # (Max_Sents)
            "video_features": video_features,    # (Max_Shots, EMBEDDING_DIM) - SAME DIM!
            "video_mask": video_mask,            # (Max_Shots)
            "summ_labels": summ_labels,          # (Max_Sents)
            "num_sentences": populated_sents,
            "num_shots": populated_shots
        }
