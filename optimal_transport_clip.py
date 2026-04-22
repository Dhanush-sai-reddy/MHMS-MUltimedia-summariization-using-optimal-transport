import numpy as np

def cosine_distance_matrix(E, V):
    """
    Compute the pairwise cosine distance matrix between two sets of embeddings.
    As per equation (10) in the MHMS paper:
    C_km = 1 - (e_k^T v_m) / (||e_k|| * ||v_m||)
    
    Args:
        E: (K, D) array of text embeddings (K tokens, D dim)
        V: (M, D) array of visual embeddings (M frames, D dim)
    Returns:
        C: (K, M) cost matrix
    """
    # Normalize the embeddings (with epsilon to handle zero-norm rows)
    E_norm = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    V_norm = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
    
    # Cosine similarity matrix
    similarity = np.dot(E_norm, V_norm.T)
    
    # Cosine distance
    C = 1 - similarity
    return C

def sinkhorn_algorithm(C, reg=0.1, num_iters=100):
    """
    Sinkhorn-Knopp algorithm for Optimal Transport problem.
    This solves the entropic regularized OT problem:
    min_T <C, T> + reg * H(T)
    
    Args:
        C: (K, M) cost matrix
        reg: Regularization parameter (beta or lambda in the paper)
        num_iters: Number of iterations for Sinkhorn
        
    Returns:
        T: (K, M) optimal transport matrix
        distance: The optimal transport distance
    """
    K, M = C.shape
    
    # We assume uniform marginal distributions as stated in the paper:
    # mu_k = 1/K, v_m = 1/M
    mu = np.ones(K) / K
    nu = np.ones(M) / M
    
    # Compute the Gibbs kernel
    K_matrix = np.exp(-C / reg)
    
    # Initialize scaling vectors
    u = np.ones(K) / K
    v = np.ones(M) / M
    
    # Iterative update
    for _ in range(num_iters):
        u = mu / np.dot(K_matrix, v)
        v = nu / np.dot(K_matrix.T, u)
        
    # The optimal transport plan
    T = np.diag(u) @ K_matrix @ np.diag(v)
    
    # The calculated cost
    distance = np.sum(T * C)
    
    return T, distance

def align_case(text_emb, visual_emb, reg=0.05, num_iters=100):
    """
    Align text and visual embeddings for a single case.
    Projects both to a common dimension, then runs Sinkhorn OT.
    
    Returns:
        T: transport plan matrix
        ot_distance: scalar alignment cost
        top_pairs: list of (text_idx, visual_idx, mass) sorted by strength
    """
    D_common = min(text_emb.shape[1], visual_emb.shape[1])
    E = text_emb[:, :D_common]
    V = visual_emb[:, :D_common]
    
    # Re-normalize after truncation
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
    
    C = cosine_distance_matrix(E, V)
    T, ot_distance = sinkhorn_algorithm(C, reg=reg, num_iters=num_iters)
    
    # Extract top alignment pairs from T
    K, M = T.shape
    pairs = []
    for i in range(K):
        for j in range(M):
            pairs.append((i, j, float(T[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    return T, ot_distance, pairs


if __name__ == "__main__":
    import os
    import json
    import glob
    
    emb_dir = "embeddings"
    text_dir = os.path.join(emb_dir, "text")
    visual_dir = os.path.join(emb_dir, "visual")
    out_dir = os.path.join(emb_dir, "alignments")
    os.makedirs(out_dir, exist_ok=True)
    
    print("=" * 60)
    print("  MHMS — Optimal Transport Cross-Modal Alignment")
    print("=" * 60)
    
    # Find cases that have BOTH text and visual embeddings
    text_files = {f.replace("case_", "").replace(".npy", ""): os.path.join(text_dir, f)
                  for f in os.listdir(text_dir) if f.endswith(".npy")}
    visual_files = {f.replace("case_", "").replace(".npy", ""): os.path.join(visual_dir, f)
                    for f in os.listdir(visual_dir) if f.endswith(".npy")}
    
    common_cases = sorted(set(text_files.keys()) & set(visual_files.keys()), key=int)
    print(f"\n  Cases with both modalities: {len(common_cases)}")
    print(f"  Regularization (lambda):    0.05")
    print(f"  Sinkhorn iterations:        100\n")
    
    results_summary = {}
    all_distances = []
    
    for idx, case_id in enumerate(common_cases):
        text_emb = np.load(text_files[case_id])
        visual_emb = np.load(visual_files[case_id])
        
        T, ot_dist, top_pairs = align_case(text_emb, visual_emb)
        all_distances.append(ot_dist)
        
        # Keep top-N pairs (N = min of the two modality sizes)
        top_n = min(text_emb.shape[0], visual_emb.shape[0])
        top_pairs_trimmed = top_pairs[:top_n]
        
        result = {
            "case_id": int(case_id),
            "num_sentences": text_emb.shape[0],
            "num_keyframes": visual_emb.shape[0],
            "ot_distance": round(ot_dist, 6),
            "transport_plan_shape": list(T.shape),
            "top_alignments": [
                {"text_idx": p[0], "visual_idx": p[1], "transport_mass": round(p[2], 6)}
                for p in top_pairs_trimmed
            ]
        }
        
        # Save per-case alignment
        case_out = os.path.join(out_dir, f"case_{case_id}.json")
        with open(case_out, 'w') as f:
            json.dump(result, f, indent=2)
        
        results_summary[case_id] = {
            "ot_distance": round(ot_dist, 6),
            "shape": f"{text_emb.shape[0]}×{visual_emb.shape[0]}",
            "strongest_pair": f"sent_{top_pairs[0][0]}↔frame_{top_pairs[0][1]}"
        }
        
        if idx < 5 or idx % 40 == 0:
            print(f"  Case {case_id:>3s}  |  {text_emb.shape[0]:>2d} sents × {visual_emb.shape[0]:>2d} frames  |  "
                  f"OT dist: {ot_dist:.4f}  |  Top: sent_{top_pairs[0][0]}↔frame_{top_pairs[0][1]} "
                  f"(mass={top_pairs[0][2]:.4f})")
    
    # Save global summary
    summary_path = os.path.join(out_dir, "alignment_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Print stats
    dists = np.array(all_distances)
    print(f"\n{'=' * 60}")
    print(f"  Alignment Complete!")
    print(f"{'=' * 60}")
    print(f"  Cases aligned:     {len(common_cases)}")
    print(f"  Mean OT distance:  {dists.mean():.4f}")
    print(f"  Min OT distance:   {dists.min():.4f} (strongest alignment)")
    print(f"  Max OT distance:   {dists.max():.4f} (weakest alignment)")
    print(f"  Std OT distance:   {dists.std():.4f}")
    print(f"\n  Results → {out_dir}/")
    print(f"  Summary → {summary_path}")
    print(f"{'=' * 60}")
