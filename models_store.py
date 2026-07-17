"""
models_store.py — encoders, reranker, and vector-store client
=============================================================

Loads the three models (Standard CLIP, Fashion-CLIP, BLIP ITM) and opens the Qdrant
client, then exposes thin embedding helpers used by both the indexing and retrieval
pipelines. Keeping model handles and the DB client in one module means the same loaded
weights are reused everywhere and nothing is re-instantiated per call.

Why these three models
----------------------
* Standard CLIP (ViT-B/32): a general-purpose image-text encoder. It keeps background and
  scene information, so it owns the "where" axis (office, park, street, home).
* Fashion-CLIP: CLIP fine-tuned on fashion product data. It is stronger on garment type,
  colour, and style, so it owns the "what they're wearing" axis. Its training images are
  single products on white backgrounds, which motivates the segmentation step in the
  indexer (see indexing.py).
* BLIP ITM (Image-Text Matching): a *fused* cross-encoder. Image patches and text tokens
  attend to each other before a match/no-match head fires, so it can verify which colour
  is on which garment — the compositional check a pooled bi-encoder cannot do.
"""

import torch
import torch.nn.functional as F
from transformers import (CLIPModel, CLIPProcessor,
                          BlipProcessor, BlipForImageTextRetrieval)
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from . import config


# ============================================================================ model loading
def _optional_hf_login():
    """Log in to the HF Hub if a token is available (Colab userdata or env). Optional:
    all three models are public, so this is a no-op when no token is present."""
    token = None
    try:
        from google.colab import userdata
        token = userdata.get("HF_Token")
    except Exception:
        import os
        token = os.environ.get("HF_TOKEN")
    if token:
        try:
            from huggingface_hub import login
            login(token=token)
            print("[models] HF Hub login ok")
        except Exception as e:
            print(f"[models] HF login skipped: {e}")
    return token


print(f"[models] loading encoders + reranker on {config.DEVICE} ...")
_HF_TOKEN = _optional_hf_login()

std_model = CLIPModel.from_pretrained(config.STD_MODEL_ID).to(config.DEVICE).eval()
std_processor = CLIPProcessor.from_pretrained(config.STD_MODEL_ID)

f_model = CLIPModel.from_pretrained(config.FCLIP_MODEL_ID).to(config.DEVICE).eval()
f_processor = CLIPProcessor.from_pretrained(config.FCLIP_MODEL_ID)

blip_processor = BlipProcessor.from_pretrained(config.BLIP_ITM_ID)
blip_model = BlipForImageTextRetrieval.from_pretrained(config.BLIP_ITM_ID).to(config.DEVICE).eval()

print("[models] all models ready")


# ============================================================================ Qdrant client
def get_client() -> QdrantClient:
    """Open the on-disk Qdrant client. Swap this one function for a Qdrant *server* URL
    (QdrantClient(url=...)) to scale out — no other code changes (see the report's
    scalability section)."""
    return QdrantClient(path=config.QDRANT_PATH)


def recreate_collection(client: QdrantClient):
    """(Re)create the collection with two named vector spaces for a clean state."""
    if client.collection_exists(config.COLLECTION_NAME):
        client.delete_collection(config.COLLECTION_NAME)
    client.create_collection(
        collection_name=config.COLLECTION_NAME,
        vectors_config={
            config.STD_VECTOR:   qmodels.VectorParams(size=config.VECTOR_SIZE,
                                                      distance=qmodels.Distance.COSINE),
            config.FCLIP_VECTOR: qmodels.VectorParams(size=config.VECTOR_SIZE,
                                                      distance=qmodels.Distance.COSINE),
        },
    )
    print(f"[store] collection '{config.COLLECTION_NAME}' created")


# ============================================================================ embedding helpers
def extract_safe_features(features):
    """Return the projected embedding tensor across transformers versions (4.x returns a
    Tensor; 5.x returns a dataclass with image_embeds / text_embeds)."""
    if isinstance(features, torch.Tensor):
        return features
    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        v = getattr(features, attr, None)
        if v is not None:
            return v
    return features[0]


@torch.no_grad()
def embed_image(model, processor, images):
    """L2-normalised image embeddings for a list of PIL images -> (N, 512) numpy."""
    inp = processor(images=images, return_tensors="pt").to(config.DEVICE)
    feat = extract_safe_features(model.get_image_features(**inp))
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy()


@torch.no_grad()
def embed_text(model, processor, text):
    """L2-normalised text embedding for a single query string -> (512,) numpy."""
    inp = processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(config.DEVICE)
    feat = extract_safe_features(model.get_text_features(**inp))
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy()[0]


@torch.no_grad()
def blip_match_probability(image, query: str) -> float:
    """P(match) in [0,1] from the BLIP ITM head for one (image, query) pair. This is the
    cross-attention judgement used to rerank."""
    inp = blip_processor(images=image, text=query, return_tensors="pt").to(config.DEVICE)
    logits = blip_model(**inp).itm_score               # [1, 2] -> [no-match, match]
    return F.softmax(logits, dim=1)[:, 1].item()
