import argparse, os, json, torch
import torch.nn as nn
from torch.utils.data import DataLoader
from mhms.dataset_unified import CNNMultimodalDatasetUnified
from mhms.models.mhms_framework_unified import MHMS_Unified

MODEL_CONFIGS = {
    "qwen2vl": {"embedding_dim": 1536, "embeddings_dir": "embeddings_qwen2vl"},
    "qwen3vl": {"embedding_dim": 2560, "embeddings_dir": "embeddings_qwen3vl"},
    "clip":    {"embedding_dim": 512,  "embeddings_dir": "embeddings_clip"},
}

def train(args):
    cfg = MODEL_CONFIGS[args.model]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = CNNMultimodalDatasetUnified(
        embeddings_dir=cfg["embeddings_dir"], embedding_dim=cfg["embedding_dim"])
    if len(dataset) == 0:
        return
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = MHMS_Unified(embedding_dim=cfg["embedding_dim"], video_hidden_dim=512, text_hidden_dim=512, video_omega_b=3)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            tf = batch['text_features'].to(device)
            vf = batch['video_features'].to(device)
            tm = batch['text_mask'].to(device)
            vm = batch['video_mask'].to(device)
            labels = batch['summ_labels'].to(device)

            optimizer.zero_grad()
            out = model(tf, vf, text_mask=tm, video_mask=vm)
            text_loss = nn.functional.binary_cross_entropy(
                out['text_summ_probs'] * tm.float(), labels * tm.float(),
                reduction='sum') / (tm.sum() + 1e-8)
            loss = text_loss + args.lambda_ot * out['ot_loss']
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"  E{epoch} B{batch_idx+1}/{len(dataloader)} L:{loss.item():.4f}")

        print(f"Epoch {epoch} avg loss: {total_loss/len(dataloader):.4f}")
        torch.save(model.state_dict(), f"mhms_{args.model}_epoch_{epoch}.pth")

    torch.save(model.state_dict(), f"mhms_{args.model}_final.pth")
    print(f"Saved mhms_{args.model}_final.pth")

if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODEL_CONFIGS.keys()), default="qwen2vl")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lambda-ot", type=float, default=0.1)
    train(p.parse_args())
