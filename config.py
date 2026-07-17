"""
config.py — configuration, separated from logic and data
========================================================

Everything the pipeline might need to tune lives here: model IDs, paths, vector
dimensions, and the retrieval/fusion/rerank hyper-parameters. No business logic and no
data. Import this module wherever a constant is needed rather than hard-coding values in
the indexer or retriever — that separation is what lets the same logic run against a
different dataset, a different vector store, or a bigger machine without edits.
"""

import torch

# ----------------------------------------------------------------------------- devices
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------------------------------------------------------- models
STD_MODEL_ID = "openai/clip-vit-base-patch32"        # general encoder -> environment/context
FCLIP_MODEL_ID = "patrickjohncyh/fashion-clip"       # fashion encoder -> garments/colour/style
BLIP_ITM_ID = "Salesforce/blip-itm-base-coco"        # cross-encoder reranker -> composition

# ----------------------------------------------------------------------------- vector store
QDRANT_PATH = "/content/qdrant_storage"              # on-disk Qdrant (persists across runs)
COLLECTION_NAME = "fashion_search_engine"
STD_VECTOR = "std_clip"
FCLIP_VECTOR = "fashion_clip"
VECTOR_SIZE = 512                                    # ViT-B/32 embedding dim for both encoders

# ----------------------------------------------------------------------------- data
IMAGE_GLOB = "/content/fashionpedia/val/test/*.jpg"  # where the images live
INDEX_BATCH_SIZE = 32

# ----------------------------------------------------------------------------- retrieval
SCATTER_SIZE = 50            # ANN candidates pulled per channel (recall knob)
TOP_K = 5                    # default number of final results returned

# How many fused candidates the cross-encoder re-scores. The pool must always be >= k
# (you cannot return k reranked results from a smaller pool), so it scales with k:
#   pool = min(scatter_union, max(RERANK_MIN, ceil(k * RERANK_K_MULT)))
# For small k the floor RERANK_MIN applies (always rerank at least 15); for large k
# (e.g. top_k=50) the pool grows to a multiple of k.
RERANK_MIN = 15              # minimum candidates to always rerank
RERANK_K_MULT = 3.0         # for large k, rerank ~RERANK_K_MULT * k candidates

# ----------------------------------------------------------------------------- fusion / rerank
# Query-adaptive alpha: weight on the fashion channel, mapped into [ALPHA_MIN, ALPHA_MIN+ALPHA_SPAN].
ALPHA_MIN = 0.2
ALPHA_SPAN = 0.7
ALPHA_TEMPERATURE = 10.0
RERANK_BETA = 0.5           # weight of BLIP ITM P(match) added to the fusion score

# Anchor sentences that define the two ends of the fashion<->environment axis for alpha.
ALPHA_ANCHORS = [
    "A description of clothing, fashion garments, and specific outfit details.",
    "A description of a physical environment, background setting, or location.",
]

# ----------------------------------------------------------------------------- evaluation prompts
TEST_PROMPTS = [
    "A person in a bright yellow raincoat.",                                  # attribute
    "Professional business attire inside a modern office.",                   # context/place
    "Someone wearing a blue shirt sitting on a park bench.",                  # complex semantic
    "Casual weekend outfit for a city walk.",                                 # style inference
    "A red tie and a white shirt in a formal setting.",                       # compositional
    "A white tie and red shirt in a formal setting",                          # inverted trap
    "A person wearing a dark green hoodie relaxing inside a home setting.",   # env + casual
    "Bright summer clothing on someone walking down an urban street.",        # vibe + urban
    "An elegant black evening blazer worn outdoors in a public park.",        # formal hybrid
    "A pink t-shirt paired with blue denim jeans in a casual indoor room.",   # fine colour binding
]
