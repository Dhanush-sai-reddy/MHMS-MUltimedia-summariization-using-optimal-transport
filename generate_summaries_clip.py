import os
import torch
import json
from mhms.dataset_clip import CNNMultimodalDatasetCLIP
from mhms.models.mhms_framework_clip import MHMS_CLIP

def generate_and_save_summaries():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Configuration
    DATA_DIR = "cnn_data"
    EMBEDDINGS_DIR = "embeddings_clip"
    CHECKPOINT_DIR = "checkpoints_clip"
    EMBEDDING_DIM = 512 # Qwen3-VL-4B hidden size
    VIDEO_HIDDEN_DIM = 512
    TEXT_HIDDEN_DIM = 512
    VIDEO_OMEGA_B = 3
    
    # 1. Initialize Dataset
    print(f"Loading CNN Dataset (Qwen3 VL Embeddings) for Inference...")
    dataset = CNNMultimodalDatasetCLIP(
        data_dir=DATA_DIR, 
        embeddings_dir=EMBEDDINGS_DIR, 
        max_sentences=20, 
        max_shots=20,
        embedding_dim=EMBEDDING_DIM
    )
    
    if len(dataset) == 0:
        print("Dataset is empty. Ensure embeddings_clip has matching features.")
        return

    # 2. Initialize the Model framework
    print("Initializing MHMS CLIP Framework...")
    model = MHMS_CLIP(
        embedding_dim=EMBEDDING_DIM,
        video_hidden_dim=VIDEO_HIDDEN_DIM,
        text_hidden_dim=TEXT_HIDDEN_DIM,
        video_omega_b=VIDEO_OMEGA_B,
        use_text_segmentation=False
    )
    
    final_path = os.path.join(CHECKPOINT_DIR, "mhms_clip_final.pth")
    if os.path.exists(final_path):
        # We need to just load the weights, wait, wait... train_clip.py saves `model.state_dict()` in mhms_clip_final.pth
        model.load_state_dict(torch.load(final_path, map_location=device))
        print("Pre-trained MHMS CLIP weights loaded successfully!")
    else:
        print("No trained weights found. Running generation with untrained initialized weights.")
        
    model.to(device)
    model.eval()

    print("\nStarting generation over cases...\n")
    
    # Process each case item individually
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            case_id = dataset.cases[i]["id"]
            sample_path = os.path.join(DATA_DIR, str(case_id))
            
            # Need to add batch dimension (1, ...)
            text_features = sample['text_features'].unsqueeze(0).to(device)
            video_features = sample['video_features'].unsqueeze(0).to(device)
            
            # Use forward pass to get text probs
            # In Qwen version, we directly use model(text, video) which returns text_summ_probs
            outputs = model(text_features, video_features)
            text_summ_probs = outputs["text_summ_probs"][0] # (1, max_text_nodes) -> (max_text_nodes)
            
            # Since MHMS_CLIP doesn't have an OT align inference function written yet,
            # we'll use ot_loss from outputs or do our own fast OT block or match by probs
            # Actually train_clip computes OT matrix. Here we just take top text nodes
            # To be simple and robust:
            probs = text_summ_probs.cpu().numpy()
            
            # Limit to actual sentences based on mask
            text_len = sample['text_mask'].sum().int().item()
            probs = probs[:text_len]
            
            # Top-3 indices
            top_indices = probs.argsort()[-3:][::-1]
            
            # Formulate the summary output result
            output_data = {
                "multimodal_summary": []
            }
            
            for idx in top_indices:
                output_data["multimodal_summary"].append({
                    "selected_text_sentence_index": int(idx),
                    "aligned_visual_segment_index": 0, # Simplified fallback when OT exact pairing is unsupported in generation
                    "score": float(probs[idx])
                })
                
            # Write JSON output directly inside the sample's case folder
            save_dest = os.path.join(sample_path, "multimodal_summary_output.json")
            with open(save_dest, "w") as f:
                json.dump(output_data, f, indent=4)
                
            if i % 10 == 0:
                print(f"Successfully processed and generated summary for target: {sample_path}")

    print("\nFinished generating and saving multimodal summaries inside each cnn_data folder!")

if __name__ == "__main__":
    generate_and_save_summaries()
