# Compositional Fashion Retrieval

Text-to-image search over a fashion dataset for queries that mix **garment + colour +
environment** ("a red tie and a white shirt in a formal setting"). Built to beat a vanilla
CLIP application on the hard case: **compositional binding** (red tie/white shirt vs its
inverse).

## Architecture

```
INDEX (once)                                   RETRIEVE (per query)
  raw image ─► Standard CLIP ─► std_clip vec     query ─► alpha = adaptive fashion weight
  rembg→white ─► Fashion-CLIP ─► fashion_clip     ─► ANN std_clip + ANN fashion_clip (scatter)
        both in one Qdrant collection            ─► fuse: alpha·fclip + (1-alpha)·std
                                                 ─► BLIP ITM rerank top-15 (cross-attention)
                                                 ─► top-k fusion images
```

Three models, each on the axis it owns: **Standard CLIP** (environment), **Fashion-CLIP**
(garments/colour, fed segmented white-bg crops to match its training distribution), and
**BLIP ITM** (a fused cross-encoder that verifies which colour is on which garment).

## Modules (logic separated from data & config)

| file | role |
|------|------|
| `config.py` | all paths, model IDs, and hyper-parameters — no logic |
| `models_store.py` | loads the 3 models + Qdrant client; embedding helpers |
| `indexing.py` | Part A: segmentation + dual-encoder indexing |
| `retrieval.py` | Part B: adaptive fusion + BLIP rerank (fusion output only) |
| `run.py` | entrypoint |

## Setup

```bash
pip install torch transformers qdrant-client rembg pillow matplotlib scikit-learn huggingface_hub
```

Point `config.IMAGE_GLOB` at your images and `config.QDRANT_PATH` at a writable dir.

## Run

```bash
python -m fashion_search.run --index     
python -m fashion_search.run             # retrieve only (index already built)
```

```python
# or from a notebook / script
from fashion_search import models_store as ms, indexing, retrieval
client = ms.get_client()
indexing.build_index(client)                        # once
retrieval.show_results(client, "a red tie and a white shirt in a formal setting")
```

## Notes

- **Scalable retrieval:** indexing is one-time; queries touch only vectors. Swap
  `get_client()` for a Qdrant server URL to go from a laptop to a sharded cluster with no
  logic change. Rerank cost is bounded (top-15), independent of corpus size.
- **Zero-shot:** every stage is open-vocabulary — no fashion labels are trained on. New
  descriptions work because CLIP/Fashion-CLIP/BLIP were pre-trained on open text.
- **Known limits:** base encoders are the weakest link (recall ceiling); fusion sums raw
  cosines; `alpha` varies less than its range suggests. See the accompanying report.
