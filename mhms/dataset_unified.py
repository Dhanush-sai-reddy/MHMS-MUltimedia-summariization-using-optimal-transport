import os, json, torch, numpy as np
from torch.utils.data import Dataset

class CNNMultimodalDatasetUnified(Dataset):
    def __init__(self, data_dir="cnn_data", embeddings_dir="embeddings_qwen2vl",
                 max_sentences=20, max_shots=20, embedding_dim=1536):
        self.max_sentences = max_sentences
        self.max_shots = max_shots
        self.embedding_dim = embedding_dim

        label_file = os.path.join(data_dir, 'label.txt')
        self.labels = []
        if os.path.exists(label_file):
            with open(label_file) as f:
                for line in f:
                    line = line.strip().replace('[', '').replace(']', '')
                    self.labels.append([float(x) for x in line.split()] if line else [])

        self.samples = []
        text_dir = os.path.join(embeddings_dir, "text")
        visual_dir = os.path.join(embeddings_dir, "visual")
        if not os.path.exists(text_dir) or not os.path.exists(visual_dir):
            raise ValueError(f"Embeddings not found at '{embeddings_dir}'. Run embedding pipeline first.")

        manifest_path = os.path.join(embeddings_dir, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                saved_dim = json.load(f).get("embedding_dim", embedding_dim)
                if saved_dim != embedding_dim:
                    self.embedding_dim = saved_dim

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
        print(f"Unified dataset: {len(self.samples)} cases, dim={self.embedding_dim}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample_id = sample["id"]

        text_npy = np.load(sample["text_path"])
        populated_sents = min(text_npy.shape[0], self.max_sentences)
        text_features = torch.zeros((self.max_sentences, self.embedding_dim), dtype=torch.float)
        text_features[:populated_sents] = torch.from_numpy(text_npy[:populated_sents])
        text_mask = torch.zeros(self.max_sentences, dtype=torch.long)
        text_mask[:populated_sents] = 1

        vis_npy = np.load(sample["visual_path"])
        populated_shots = min(vis_npy.shape[0], self.max_shots)
        video_features = torch.zeros((self.max_shots, self.embedding_dim), dtype=torch.float)
        video_features[:populated_shots] = torch.from_numpy(vis_npy[:populated_shots])
        video_mask = torch.zeros(self.max_shots, dtype=torch.long)
        video_mask[:populated_shots] = 1

        label_idx = min(sample_id - 1, len(self.labels) - 1)
        summ_labels = torch.zeros(self.max_sentences, dtype=torch.float)
        for i, val in enumerate((self.labels[label_idx] if label_idx >= 0 else [])[:self.max_sentences]):
            summ_labels[i] = val

        return {
            "text_features": text_features, "text_mask": text_mask,
            "video_features": video_features, "video_mask": video_mask,
            "summ_labels": summ_labels,
            "num_sentences": populated_sents, "num_shots": populated_shots
        }
