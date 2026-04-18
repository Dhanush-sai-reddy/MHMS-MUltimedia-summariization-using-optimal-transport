"""
Training Script for MHMS with Qwen3 VL Unified Embeddings
==========================================================
Trains the MHMS framework using Qwen3 VL embeddings.

Key differences from original train.py:
- Uses CNNMultimodalDatasetCLIP (unified embeddings)
- Uses MHMS_CLIP framework (optimized for unified dim)
- Better cross-modal alignment due to shared embedding space
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import os
import json

from mhms.dataset_clip import CNNMultimodalDatasetCLIP
from mhms.models.mhms_framework_clip import MHMS_CLIP


def train_epoch(model, dataloader, optimizer, device, lambda_ot=0.1):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_text_loss = 0
    total_ot_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        text_features = batch["text_features"].to(device)
        text_mask = batch["text_mask"].to(device)
        video_features = batch["video_features"].to(device)
        video_mask = batch["video_mask"].to(device)
        summ_labels = batch["summ_labels"].to(device)
        
        # Forward pass
        outputs = model(text_features, video_features)
        
        # Compute losses
        # Text summarization loss (supervised)
        text_summ_probs = outputs["text_summ_probs"]
        
        # Masked BCE loss for text summarization
        text_loss = nn.functional.binary_cross_entropy(
            text_summ_probs * text_mask.float(),
            summ_labels * text_mask.float(),
            reduction='sum'
        ) / (text_mask.sum() + 1e-8)
        
        # Optimal Transport loss (cross-modal alignment)
        ot_loss = outputs["ot_loss"]
        
        # Combined loss
        loss = text_loss + lambda_ot * ot_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track losses
        total_loss += loss.item()
        total_text_loss += text_loss.item()
        total_ot_loss += ot_loss.item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)} | "
                  f"Loss: {loss.item():.4f} (Text: {text_loss.item():.4f}, OT: {ot_loss.item():.4f})")
    
    avg_loss = total_loss / len(dataloader)
    avg_text_loss = total_text_loss / len(dataloader)
    avg_ot_loss = total_ot_loss / len(dataloader)
    
    return avg_loss, avg_text_loss, avg_ot_loss


def main():
    # Configuration
    DATA_DIR = "cnn_data"
    EMBEDDINGS_DIR = "embeddings_clip"
    CHECKPOINT_DIR = "checkpoints_clip"
    
    # Model hyperparameters
    EMBEDDING_DIM = 512  # Qwen3-VL-4B hidden size (4096 for 8B-Instruct)
    VIDEO_HIDDEN_DIM = 512
    TEXT_HIDDEN_DIM = 512
    VIDEO_OMEGA_B = 3
    
    # Training hyperparameters
    EPOCHS = 5
    BATCH_SIZE = 2
    LEARNING_RATE = 1e-4
    LAMBDA_OT = 0.1  # Weight for OT loss
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Save config
    config = {
        "embedding_dim": EMBEDDING_DIM,
        "video_hidden_dim": VIDEO_HIDDEN_DIM,
        "text_hidden_dim": TEXT_HIDDEN_DIM,
        "video_omega_b": VIDEO_OMEGA_B,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "lambda_ot": LAMBDA_OT,
        "embeddings_dir": EMBEDDINGS_DIR
    }
    with open(os.path.join(CHECKPOINT_DIR, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize dataset
    print("\nInitializing Qwen3 VL dataset...")
    try:
        dataset = CNNMultimodalDatasetCLIP(
            data_dir=DATA_DIR,
            embeddings_dir=EMBEDDINGS_DIR,
            max_sentences=20,
            max_shots=20,
            embedding_dim=EMBEDDING_DIM
        )
    except ValueError as e:
        print(f"Error: {e}")
        print("\nPlease generate Qwen3 VL embeddings first:")
        print("  python embedding_pipeline_clip.py")
        return
    
    if len(dataset) == 0:
        print("No samples found in dataset!")
        return
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize model
    print("\nInitializing MHMS model with Qwen3 VL embeddings...")
    model = MHMS_CLIP(
        embedding_dim=EMBEDDING_DIM,
        video_hidden_dim=VIDEO_HIDDEN_DIM,
        text_hidden_dim=TEXT_HIDDEN_DIM,
        video_omega_b=VIDEO_OMEGA_B,
        use_text_segmentation=False  # Disabled for speed
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print("\n" + "=" * 60)
    print("  Starting Training (Qwen3 VL Unified Embeddings)")
    print("=" * 60)
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        print("-" * 40)
        
        avg_loss, avg_text_loss, avg_ot_loss = train_epoch(
            model, dataloader, optimizer, device, lambda_ot=LAMBDA_OT
        )
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Text Loss:    {avg_text_loss:.4f}")
        print(f"  OT Loss:      {avg_ot_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"mhms_clip_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'config': config
        }, checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(CHECKPOINT_DIR, "mhms_clip_final.pth")
    torch.save(model.state_dict(), final_path)
    
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print(f"  Final model: {final_path}")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Evaluate model: python evaluate_model.py --checkpoint", final_path)
    print("  2. Generate summaries: python generate_summaries.py --checkpoint", final_path)


if __name__ == "__main__":
    main()
