"""
retrieval.py — the Retriever (Part B)
=====================================

Text query -> top-k images, in three stages:

  1. QUERY-ADAPTIVE FUSION WEIGHT (alpha)
     Decide how much to trust the fashion channel vs the environment channel for THIS
     query, by comparing the query to two anchor sentences.

  2. SCATTER-GATHER LATE FUSION  (fast, high-recall, bi-encoder)
     ANN top-N on each named vector independently, union the candidates, and score each by
     `alpha * fashion_cosine + (1 - alpha) * std_cosine`.

  3. CROSS-ENCODER RERANK  (precise, compositional, BLIP ITM)
     Re-score the top fused candidates with BLIP's image-text matching head, whose
     cross-attention can check which colour is on which garment, then re-sort by
     `fusion_score + beta * P(match)`.

The rerank POOL scales with k: we always rerank at least RERANK_MIN (15), but when k is
larger (e.g. top_k=50) the pool grows to a multiple of k so there are always enough
reranked candidates to return. Only the final FUSION ranking is returned/printed.
"""

import math
import numpy as np
from PIL import Image

from . import config
from . import models_store as ms


def _rerank_pool_size(k, n_candidates):
    """Pool must be >= k. Floor at RERANK_MIN, grow with k, cap at what we retrieved."""
    want = max(config.RERANK_MIN, math.ceil(k * config.RERANK_K_MULT))
    return min(want, n_candidates)


# ------------------------------------------------------------------ stage 1: adaptive alpha
def compute_dynamic_alpha(query: str) -> float:
    """Weight on the fashion channel for this query, in [ALPHA_MIN, ALPHA_MIN+ALPHA_SPAN].

    The query and two anchor sentences (one 'clothing', one 'environment') are embedded
    with Standard CLIP; a temperature-scaled softmax over the two cosine similarities gives
    the fashion weight. Garment-heavy queries lean fashion; scene-heavy queries lean std.
    """
    q = ms.embed_text(ms.std_model, ms.std_processor, query)
    anchors = np.stack([ms.embed_text(ms.std_model, ms.std_processor, a)
                        for a in config.ALPHA_ANCHORS])
    fashion_sim, env_sim = anchors @ q                      # cosine (all normalised)

    t = config.ALPHA_TEMPERATURE
    ef, ee = np.exp(fashion_sim * t), np.exp(env_sim * t)
    weight = ef / (ef + ee)
    return config.ALPHA_MIN + weight * config.ALPHA_SPAN


# ------------------------------------------------------------------ stage 2 + 3: search
def search(client, query: str, top_k: int = config.TOP_K,
           scatter_size: int = config.SCATTER_SIZE,
           rerank_pool_size: int = None,
           verbose: bool = True):
    """Return the top-k fusion results for a query as a list of dicts:
    {image_path, fusion_score, fclip_score, std_score, itm_prob, boosted_score}.

    The rerank pool auto-scales with k (>= RERANK_MIN, ~RERANK_K_MULT * k for large k);
    pass rerank_pool_size to override.
    """
    alpha = compute_dynamic_alpha(query)
    std_q = ms.embed_text(ms.std_model, ms.std_processor, query)
    f_q = ms.embed_text(ms.f_model, ms.f_processor, query)

    # ---- scatter: independent ANN per channel ----
    # pull enough per channel that the union comfortably covers the (k-scaled) rerank pool
    per_channel = max(scatter_size, math.ceil(top_k * config.RERANK_K_MULT))
    res_std = client.query_points(config.COLLECTION_NAME, query=std_q.tolist(),
                                  using=config.STD_VECTOR, limit=per_channel).points
    res_f = client.query_points(config.COLLECTION_NAME, query=f_q.tolist(),
                                using=config.FCLIP_VECTOR, limit=per_channel).points

    # ---- gather: pull both vectors for the union of candidates ----
    ids = list({h.id for h in res_std} | {h.id for h in res_f})
    candidates = client.retrieve(config.COLLECTION_NAME, ids=ids,
                                 with_vectors=True, with_payload=True)

    # ---- late fusion ----
    fused = []
    for p in candidates:
        s_std = float(np.dot(std_q, np.array(p.vector[config.STD_VECTOR])))
        s_f = float(np.dot(f_q, np.array(p.vector[config.FCLIP_VECTOR])))
        fused.append({
            "image_path": p.payload["image_path"],
            "fusion_score": alpha * s_f + (1 - alpha) * s_std,
            "fclip_score": s_f,
            "std_score": s_std,
        })
    fused.sort(key=lambda x: x["fusion_score"], reverse=True)

    # ---- k-scaled rerank pool (always >= top_k) ----
    pool_size = rerank_pool_size or _rerank_pool_size(top_k, len(fused))
    shortlist = fused[:pool_size]
    if verbose:
        print(f"[search] '{query}' | alpha={alpha:.2f} | reranking {pool_size} with BLIP ITM ...")

    # ---- cross-encoder rerank: additive boost ----
    for m in shortlist:
        img = Image.open(m["image_path"]).convert("RGB")
        p_match = ms.blip_match_probability(img, query)
        m["itm_prob"] = p_match
        m["boosted_score"] = m["fusion_score"] + config.RERANK_BETA * p_match

    shortlist.sort(key=lambda x: x["boosted_score"], reverse=True)
    return shortlist[:top_k]


# ------------------------------------------------------------------ display (fusion only)
def show_results(client, query: str, top_k: int = config.TOP_K):
    """Run a search and display ONLY the final fusion images inline (for notebooks)."""
    import matplotlib.pyplot as plt
    results = search(client, query, top_k=top_k)

    print(f"\n==================== {query} ====================")
    for rank, m in enumerate(results, 1):
        plt.figure(figsize=(4, 4))
        plt.imshow(Image.open(m["image_path"]))
        plt.title(f"#{rank}  boosted={m['boosted_score']:.3f}\n"
                  f"fusion={m['fusion_score']:.3f} | ITM={m['itm_prob']:.2f}", fontsize=9)
        plt.axis("off")
        plt.show()
        print(m["image_path"])
    return results


def run_prompts(client, prompts=None, top_k: int = config.TOP_K):
    """Display fusion results for every prompt in the list (defaults to config.TEST_PROMPTS)."""
    prompts = prompts or config.TEST_PROMPTS
    return {q: show_results(client, q, top_k=top_k) for q in prompts}


if __name__ == "__main__":
    client = ms.get_client()
    run_prompts(client)
