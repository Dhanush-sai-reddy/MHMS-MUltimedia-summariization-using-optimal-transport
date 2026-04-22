import numpy as np

def cosine_distance_matrix(E, V):
    E_norm = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    V_norm = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
    return 1 - np.dot(E_norm, V_norm.T)

def sinkhorn_algorithm(C, reg=0.1, num_iters=100):
    K, M = C.shape
    mu = np.ones(K) / K
    nu = np.ones(M) / M
    K_matrix = np.exp(-C / reg)
    u = np.ones(K) / K
    v = np.ones(M) / M

    for _ in range(num_iters):
        u = mu / np.dot(K_matrix, v)
        v = nu / np.dot(K_matrix.T, u)

    T = np.diag(u) @ K_matrix @ np.diag(v)
    distance = np.sum(T * C)
    return T, distance

def align_case(text_emb, visual_emb, reg=0.05, num_iters=100):
    D_common = min(text_emb.shape[1], visual_emb.shape[1])
    E = text_emb[:, :D_common]
    V = visual_emb[:, :D_common]
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)

    C = cosine_distance_matrix(E, V)
    T, ot_distance = sinkhorn_algorithm(C, reg=reg, num_iters=num_iters)

    K, M = T.shape
    pairs = []
    for i in range(K):
        for j in range(M):
            pairs.append((i, j, float(T[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return T, ot_distance, pairs


if __name__ == "__main__":
    import os, json

    emb_dir = "embeddings"
    text_dir = os.path.join(emb_dir, "text")
    visual_dir = os.path.join(emb_dir, "visual")
    out_dir = os.path.join(emb_dir, "alignments")
    os.makedirs(out_dir, exist_ok=True)

    text_files = {f.replace("case_", "").replace(".npy", ""): os.path.join(text_dir, f)
                  for f in os.listdir(text_dir) if f.endswith(".npy")}
    visual_files = {f.replace("case_", "").replace(".npy", ""): os.path.join(visual_dir, f)
                    for f in os.listdir(visual_dir) if f.endswith(".npy")}
    common_cases = sorted(set(text_files.keys()) & set(visual_files.keys()), key=int)

    print(f"Cases with both modalities: {len(common_cases)}")

    results_summary = {}
    all_distances = []

    for idx, case_id in enumerate(common_cases):
        text_emb = np.load(text_files[case_id])
        visual_emb = np.load(visual_files[case_id])
        T, ot_dist, top_pairs = align_case(text_emb, visual_emb)
        all_distances.append(ot_dist)

        top_n = min(text_emb.shape[0], visual_emb.shape[0])
        result = {
            "case_id": int(case_id),
            "num_sentences": text_emb.shape[0],
            "num_keyframes": visual_emb.shape[0],
            "ot_distance": round(ot_dist, 6),
            "top_alignments": [
                {"text_idx": p[0], "visual_idx": p[1], "transport_mass": round(p[2], 6)}
                for p in top_pairs[:top_n]
            ]
        }
        with open(os.path.join(out_dir, f"case_{case_id}.json"), 'w') as f:
            json.dump(result, f, indent=2)

        results_summary[case_id] = {"ot_distance": round(ot_dist, 6)}
        if idx < 5 or idx % 40 == 0:
            print(f"  Case {case_id:>3s} | {text_emb.shape[0]:>2d} sents x {visual_emb.shape[0]:>2d} frames | OT: {ot_dist:.4f}")

    with open(os.path.join(out_dir, "alignment_summary.json"), 'w') as f:
        json.dump(results_summary, f, indent=2)

    dists = np.array(all_distances)
    print(f"\nAligned {len(common_cases)} cases | Mean OT: {dists.mean():.4f} | Min: {dists.min():.4f} | Max: {dists.max():.4f}")
