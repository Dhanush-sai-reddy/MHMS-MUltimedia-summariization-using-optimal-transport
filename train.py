import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from mhms.dataset import CNNMultimodalDataset
from mhms.models.mhms_framework import MHMS

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    dataset = CNNMultimodalDataset(data_dir="cnn_data", embeddings_dir="embeddings", max_sentences=20, visual_dim=2048, text_dim=768)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    print(f"Loaded {len(dataset)} samples.")
    if len(dataset) == 0:
        return

    model = MHMS(text_feature_dim=768, visual_feature_dim=2048, video_hidden_dim=256, video_omega_b=3)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    bce_loss = nn.BCELoss()
    alpha_ot = 0.5
    num_epochs = 3

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(dataloader):
            text_features = batch['text_features'].to(device)
            video_features = batch['video_features'].to(device)
            text_input_ids = batch['text_input_ids'].to(device)
            text_attention_mask = batch['text_attention_mask'].to(device)
            text_mask = batch['text_mask'].to(device)
            video_mask = batch['video_mask'].to(device)
            summ_labels = batch['summ_labels'].to(device)

            optimizer.zero_grad()
            outputs = model(text_features, video_features, text_input_ids, text_attention_mask,
                          text_mask=text_mask, video_mask=video_mask)

            l_text = bce_loss(outputs['text_summ_probs'], summ_labels)
            ot_loss = outputs['ot_loss']
            loss = l_text + alpha_ot * ot_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 2 == 0:
                print(f"E[{epoch+1}/{num_epochs}] B[{batch_idx+1}/{len(dataloader)}] "
                      f"Loss:{loss.item():.4f} (T:{l_text.item():.4f} OT:{ot_loss.item():.4f})")

        print(f"--- Epoch {epoch+1} Avg Loss: {total_loss/len(dataloader):.4f} ---")
        torch.save(model.state_dict(), f"mhms_epoch_{epoch+1}.pth")

    torch.save(model.state_dict(), "mhms_model_weights.pth")
    print("Saved mhms_model_weights.pth")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    train()
