"""
evaluator.py — fair comparison of retrieval approaches
======================================================

Two complementary measurements, both reusing the same index and models:

  1. VLM-JUDGE PRECISION@k
     Each approach does its OWN end-to-end retrieval (not a re-sort of one candidate set),
     then a local Qwen2-VL judge marks each returned image match / no-match for the query.
     Approaches compared:
         Std CLIP only   -> pure environment channel (ANN on `std_clip`)
         Fashion only    -> pure garment channel     (ANN on `fashion_clip`)
         Fusion          -> adaptive late fusion, no rerank
         Fusion + BLIP   -> full pipeline (fusion then BLIP ITM rerank)   [the system]

  2. BINDING SWAP-PROBE  (no judge, no labels)
     For a colour-inverted query pair ("red tie / white shirt" vs "white tie / red shirt"),
     does the system rank the correct image above the swapped one? Chance = 50%. This is the
     judge-free measure of compositional binding — the axis the whole design targets.

Failures from the judge return None and are excluded from the average, never scored 0, so a
flaky judge cannot masquerade as "the retrieval was wrong".

Usage:
    from fashion_search import models_store as ms, evaluator
    client = ms.get_client()
    evaluator.run_precision(client, k=5)                    # needs Qwen2-VL
    evaluator.run_binding_probe(client, [("red tie and white shirt in a formal setting",
                                          "white tie and red shirt in a formal setting")])
"""

import numpy as np
from PIL import Image

from . import config
from . import models_store as ms
from . import retrieval


# ---------------------------------------------------------------- approaches (own retrieval)
def _channel_only(client, query, using, k):
    qv = (ms.embed_text(ms.f_model, ms.f_processor, query) if using == config.FCLIP_VECTOR
          else ms.embed_text(ms.std_model, ms.std_processor, query))
    hits = client.query_points(config.COLLECTION_NAME, query=qv.tolist(),
                               using=using, limit=k, with_payload=True).points
    return [h.payload["image_path"] for h in hits]


def _fusion_only(client, query, k):
    """Adaptive fusion WITHOUT the BLIP rerank (rerank_pool_size=k, then sort by fusion)."""
    alpha = retrieval.compute_dynamic_alpha(query)
    std_q = ms.embed_text(ms.std_model, ms.std_processor, query)
    f_q = ms.embed_text(ms.f_model, ms.f_processor, query)
    res_std = client.query_points(config.COLLECTION_NAME, query=std_q.tolist(),
                                  using=config.STD_VECTOR, limit=config.SCATTER_SIZE).points
    res_f = client.query_points(config.COLLECTION_NAME, query=f_q.tolist(),
                                using=config.FCLIP_VECTOR, limit=config.SCATTER_SIZE).points
    ids = list({h.id for h in res_std} | {h.id for h in res_f})
    pts = client.retrieve(config.COLLECTION_NAME, ids=ids, with_vectors=True, with_payload=True)
    scored = []
    for p in pts:
        s = (alpha * float(np.dot(f_q, np.array(p.vector[config.FCLIP_VECTOR])))
             + (1 - alpha) * float(np.dot(std_q, np.array(p.vector[config.STD_VECTOR]))))
        scored.append((s, p.payload["image_path"]))
    scored.sort(reverse=True)
    return [path for _, path in scored[:k]]


def _fusion_blip(client, query, k):
    return [m["image_path"] for m in retrieval.search(client, query, top_k=k, verbose=False)]


def approaches(client):
    return {
        "Std CLIP only":  lambda q, k: _channel_only(client, q, config.STD_VECTOR, k),
        "Fashion only":   lambda q, k: _channel_only(client, q, config.FCLIP_VECTOR, k),
        "Fusion":         lambda q, k: _fusion_only(client, q, k),
        "Fusion + BLIP":  lambda q, k: _fusion_blip(client, q, k),
    }


# ---------------------------------------------------------------- Qwen2-VL judge
_JUDGE = {"model": None, "proc": None, "vis": None}


def load_judge(model_id="Qwen/Qwen2-VL-2B-Instruct"):
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    print(f"[eval] loading judge {model_id} ...")
    _JUDGE["model"] = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto").eval()
    _JUDGE["proc"] = AutoProcessor.from_pretrained(model_id)
    _JUDGE["vis"] = process_vision_info


def judge(image_path, query):
    """1 / 0 / None. None = judge failed (excluded), never counted as a miss."""
    import torch
    if _JUDGE["model"] is None:
        load_judge()
    try:
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": (
                f"Look at this image. Does it accurately match the description: '{query}'? "
                "Check BOTH the clothing (garment types and colours, each colour on the "
                "correct garment) AND the background setting. Answer strictly 'Yes' or 'No'.")}]}]
        proc = _JUDGE["proc"]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, vids = _JUDGE["vis"](msgs)
        inputs = proc(text=[text], images=imgs, videos=vids,
                      padding=True, return_tensors="pt").to(_JUDGE["model"].device)
        with torch.no_grad():
            out = _JUDGE["model"].generate(**inputs, max_new_tokens=5, do_sample=False)
        ans = proc.batch_decode([o[len(i):] for i, o in zip(inputs.input_ids, out)],
                                skip_special_tokens=True)[0].strip().lower()
        return 1 if "yes" in ans else 0
    except Exception as e:
        print(f"  [judge fail] {image_path}: {e}")
        return None


# ---------------------------------------------------------------- 1. precision@k
def run_precision(client, prompts=None, k=None, judge_model="Qwen/Qwen2-VL-2B-Instruct"):
    import pandas as pd
    from collections import defaultdict
    prompts = prompts or config.TEST_PROMPTS
    k = k or config.TOP_K
    apps = approaches(client)
    load_judge(judge_model)

    # gather predictions, then judge each unique (query, image) once
    preds = {(n, q): fn(q, k) for n, fn in apps.items() for q in prompts}
    cache = {}
    for (n, q), paths in preds.items():
        for p in paths:
            cache.setdefault((q, p), judge(p, q))

    rows = []
    for n in apps:
        pk = []
        for q in prompts:
            vals = [cache[(q, p)] for p in preds[(n, q)] if cache[(q, p)] is not None]
            if vals:
                pk.append(sum(vals) / len(vals))
        rows.append({"Approach": n, f"Mean P@{k}": np.mean(pk) if pk else float("nan")})
    df = pd.DataFrame(rows)
    print("\n" + "=" * 50 + f"\n  QWEN2-VL EVALUATION — Mean Precision@{k}\n" + "=" * 50)
    print(df.to_string(index=False))
    return df


# ---------------------------------------------------------------- 2. binding swap-probe
def run_binding_probe(client, pairs, k=10):
    """For each (correct_query, swapped_query), check whether the correct query's top-k and
    the swapped query's top-k differ. High overlap => the system is colour-blind to binding.
    Reports Jaccard overlap per pair and the mean (lower is better)."""
    import pandas as pd
    rows = []
    for correct_q, swapped_q in pairs:
        a = {m["image_path"] for m in retrieval.search(client, correct_q, top_k=k, verbose=False)}
        b = {m["image_path"] for m in retrieval.search(client, swapped_q, top_k=k, verbose=False)}
        overlap = len(a & b) / len(a | b) if (a | b) else 0.0
        rows.append({"correct": correct_q, "swapped": swapped_q, f"Jaccard@{k}": round(overlap, 3)})
    df = pd.DataFrame(rows)
    print("\n" + "=" * 60 + f"\n  BINDING SWAP-PROBE (Jaccard@{k}; lower = binding-aware)\n" + "=" * 60)
    print(df.to_string(index=False))
    print(f"\nmean Jaccard@{k} = {df[f'Jaccard@{k}'].mean():.3f}   (1.0 = query ignores which colour goes where)")
    return df


# default colour-inversion pairs drawn from the test prompts
BINDING_PAIRS = [
    ("A red tie and a white shirt in a formal setting.",
     "A white tie and red shirt in a formal setting"),
]


if __name__ == "__main__":
    client = ms.get_client()
    run_binding_probe(client, BINDING_PAIRS)     # judge-free, always runs
    run_precision(client)                        # needs Qwen2-VL + qwen-vl-utils
