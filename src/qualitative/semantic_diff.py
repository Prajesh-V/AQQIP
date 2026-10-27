import numpy as np
from typing import List, Tuple, Dict, Any

def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
    """
    Computes pairwise cosine similarity between two sets of embeddings.
    emb1: shape (N, D)
    emb2: shape (M, D)
    Returns: shape (N, M)
    """
    if len(emb1) == 0 or len(emb2) == 0:
        return np.array([])
        
    # Normalize
    norm1 = np.linalg.norm(emb1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(emb2, axis=1, keepdims=True)
    
    # Avoid division by zero
    norm1[norm1 == 0] = 1
    norm2[norm2 == 0] = 1
    
    emb1_norm = emb1 / norm1
    emb2_norm = emb2 / norm2
    
    return np.dot(emb1_norm, emb2_norm.T)

def compute_semantic_diff(
    sentences_n: List[str], 
    embeddings_n: np.ndarray,
    sentences_n_1: List[str],
    embeddings_n_1: np.ndarray,
    threshold_added: float = 0.8,
    threshold_identical: float = 0.95
) -> Dict[str, Any]:
    """
    Computes added, deleted, and shifted sentences using greedy one-to-one matching.
    """
    if len(sentences_n) == 0 and len(sentences_n_1) == 0:
        return {
            "added": [], "deleted": [], "shifted": [],
            "shift_score": 0.0, "cosine_similarity": 1.0
        }
        
    if len(sentences_n) == 0:
        return {
            "added": [], "deleted": sentences_n_1, "shifted": [],
            "shift_score": 1.0, "cosine_similarity": 0.0
        }
        
    if len(sentences_n_1) == 0:
        return {
            "added": sentences_n, "deleted": [], "shifted": [],
            "shift_score": 1.0, "cosine_similarity": 0.0
        }
        
    sim_matrix = compute_cosine_similarity(embeddings_n, embeddings_n_1)
    
    added = []
    shifted = []
    deleted = []
    
    # Greedy one-to-one matching
    matched_n = set()
    matched_n_1 = set()
    
    # Flatten and sort similarities descending
    # sim_matrix shape: (len(sentences_n), len(sentences_n_1))
    sim_flat = sim_matrix.flatten()
    indices = np.argsort(sim_flat)[::-1]
    
    num_n_1 = len(sentences_n_1)
    
    for idx in indices:
        i = idx // num_n_1
        j = idx % num_n_1
        
        if i in matched_n or j in matched_n_1:
            continue
            
        sim = sim_flat[idx]
        
        if sim >= threshold_identical:
            matched_n.add(i)
            matched_n_1.add(j)
        elif sim >= threshold_added:
            matched_n.add(i)
            matched_n_1.add(j)
            shifted.append(sentences_n[i])
        else:
            # All remaining matches are below threshold, so stop matching
            break
            
    # Any unmatched in N is added
    for i in range(len(sentences_n)):
        if i not in matched_n:
            added.append(sentences_n[i])
            
    # Any unmatched in N-1 is deleted
    for j in range(len(sentences_n_1)):
        if j not in matched_n_1:
            deleted.append(sentences_n_1[j])
            
    total_sentences = len(sentences_n) + len(sentences_n_1)
    changed = len(added) + len(deleted) + len(shifted)
    
    shift_score = changed / total_sentences if total_sentences > 0 else 0.0
    overall_sim = float(np.mean(sim_matrix))
    
    return {
        "added": added,
        "deleted": deleted,
        "shifted": shifted,
        "shift_score": float(shift_score),
        "cosine_similarity": overall_sim
    }
