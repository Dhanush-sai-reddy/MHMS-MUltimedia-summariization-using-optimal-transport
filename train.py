import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from mhms.dataset import CNNMultimodalDataset
from mhms.models.mhms_framework import MHMS

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Initialize Dataset and Dataloader
    print("Loading CNN Multimodal Dataset (BERT & ResNet Embeddings)...")
    dataset = CNNMultimodalDataset(data_dir="cnn_data", embeddings_dir="embeddings", max_sentences=20, visual_dim=2048, text_dim=768)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    print(f"Loaded {len(dataset)} samples.")
    
    if len(dataset) == 0:
        print("Dataset is empty. Ensure 'cnn_data' has the right folder structure.")
        return

    # 2. Initialize the MHMS Model
    print("Initializing MHMS Framework...")
    # Using small versions for demonstration purposes. 
    # Adjust omega_b based on expected sequence lengths in VTS.
    model = MHMS(
        text_feature_dim=768,
        visual_feature_dim=2048,
        video_hidden_dim=256,
        video_omega_b=3 
    )
    model.to(device)
    
    # 3. Setup Optimizer and Loss Functions
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    bce_loss = nn.BCELoss()
    
    # Weighting for the Optimal Transport Cross-modal Alignment Loss
    alpha_ot = 0.5 
    
    num_epochs = 3
    print("Starting Training Loop...")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            text_features = batch['text_features'].to(device)
            video_features = batch['video_features'].to(device)
            
            # Ground truth for Text Summarization
            summ_labels = batch['summ_labels'].to(device) # (B, max_sentences)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(text_features, video_features)
            
            # Retrieve components
            text_summ_probs = outputs['text_summ_probs']  # (B, Num_Sentences)
            ot_loss = outputs['ot_loss']                  # Scalar
            
            # Compute Supervised Text Summarization Loss
            # Target labels might be padded, ideally we'd mask out ignored segments
            l_text_summ = bce_loss(text_summ_probs, summ_labels)
            
            # The Beauty of MHMS:
            # We don't have video_summ_labels! We rely on the Optimal Transport Loss
            # to align textual and visual spaces, projecting the learning from text onto the video branches.
            loss = l_text_summ + (alpha_ot * ot_loss)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 2 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}] "
                      f"Loss: {loss.item():.4f} (Text: {l_text_summ.item():.4f}, OT: {ot_loss.item():.4f})")
                
        avg_loss = total_loss / len(dataloader)
        print(f"--- Epoch {epoch+1} Completed. Avg Total Loss: {avg_loss:.4f} ---")
        
        # Save intermediate model
        torch.save(model.state_dict(), f"mhms_epoch_{epoch+1}.pth")
        print(f"Intermediate weights saved to mhms_epoch_{epoch+1}.pth")

    print("\nTraining Complete! Saving model...")
    torch.save(model.state_dict(), "mhms_model_weights.pth")
    print("Model saved to 'mhms_model_weights.pth'")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    train()
