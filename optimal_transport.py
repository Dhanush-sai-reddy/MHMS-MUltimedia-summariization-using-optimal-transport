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
    # Normalize the embeddings
    E_norm = E / np.linalg.norm(E, axis=1, keepdims=True)
    V_norm = V / np.linalg.norm(V, axis=1, keepdims=True)
    
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

if __name__ == "__main__":
    print("--- Multimodal Alignment using Optimal Transport (Sinkhorn) ---\n")
    
    # 1. Simulate some modality features (e.g. 5 text tokens, 4 video frames)
    # Dimension of embedding D = 128
    D = 128
    K_tokens = 5
    M_frames = 4
    
    np.random.seed(42)  # For reproducibility
    text_embeddings = np.random.randn(K_tokens, D)
    video_embeddings = np.random.randn(M_frames, D)
    
    print(f"Extracted {K_tokens} text token embeddings and {M_frames} video frame embeddings (Dim={D}).\n")
    
    # 2. Compute the Cost Matrix mapping text to video embeddings
    C = cosine_distance_matrix(text_embeddings, video_embeddings)
    print("Cost Matrix (Cosine Distance) C:")
    print(np.round(C, 3))
    print()
    
    # 3. Align representations using Optimal Transport
    # We apply Sinkhorn's algorithm to compute the transport plan T
    reg_param = 0.05
    T, ot_distance = sinkhorn_algorithm(C, reg=reg_param, num_iters=100)
    
    print("Optimal Transport Plan Matrix T:")
    print(np.round(T, 3))
    print()
    
    print(f"Calculated Multimodal Alignment OT Distance: {ot_distance:.4f}")
    
    # Interpretation: Large values in Matrix T indicate a strong multimodal alignment
    # between that specific text token (row) and video frame (col).
