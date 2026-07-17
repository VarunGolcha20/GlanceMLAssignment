"""
indexing.py — the Indexer (Part A)
==================================

Turns a folder of images into a searchable two-vector index. For each image:

  * Standard CLIP embeds the RAW image           -> `std_clip` vector  (keeps environment)
  * Fashion-CLIP embeds a SEGMENTED, white-bg     -> `fashion_clip` vector (garment focus)
    version of the image

Why segment for Fashion-CLIP only
---------------------------------
Fashion-CLIP was trained on catalogue product shots: one item, white background, no
people. A cluttered real-world photo is out of distribution for it. So we remove the
background with rembg (U-2-Net) and composite the foreground on white, reconstructing the
input Fashion-CLIP handles best — while Standard CLIP keeps the untouched image so scene
information survives. Two representations of one photo, each matched to its encoder.

The heavy per-image work (segmentation + two forward passes) happens once, here at index
time; retrieval then only touches vectors. That is the property that makes the system
scale (see the report).
"""

import glob
from PIL import Image
from rembg import remove
from qdrant_client.http import models as qmodels

from . import config
from . import models_store as ms


# ------------------------------------------------------------------ segmentation
def isolate_garments(image_pil: Image.Image) -> Image.Image:
    """Remove the background (U-2-Net via rembg) and composite the foreground on pure
    white, aligning the image with Fashion-CLIP's product-shot training distribution."""
    img_rgba = remove(image_pil)                                  # adds alpha channel
    white_bg = Image.new("RGBA", img_rgba.size, "WHITE")
    white_bg.paste(img_rgba, (0, 0), img_rgba)                    # alpha as mask
    return white_bg.convert("RGB")


# ------------------------------------------------------------------ indexing
def build_index(client, image_paths=None, batch_size=config.INDEX_BATCH_SIZE):
    """Embed every image with both encoders and upsert into Qdrant.

    Args:
        client: an open Qdrant client (from models_store.get_client()).
        image_paths: list of image file paths; defaults to config.IMAGE_GLOB.
        batch_size: images per forward pass.
    """
    if image_paths is None:
        image_paths = sorted(glob.glob(config.IMAGE_GLOB))
    if not image_paths:
        raise SystemExit(f"no images matched {config.IMAGE_GLOB!r} — check config.IMAGE_GLOB")

    ms.recreate_collection(client)
    print(f"[index] indexing {len(image_paths)} images ...")

    points = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        raw_images = [Image.open(p).convert("RGB") for p in batch_paths]
        segmented = [isolate_garments(img) for img in raw_images]

        std_feat = ms.embed_image(ms.std_model, ms.std_processor, raw_images)      # raw
        f_feat = ms.embed_image(ms.f_model, ms.f_processor, segmented)             # white-bg

        for j, path in enumerate(batch_paths):
            points.append(qmodels.PointStruct(
                id=i + j,
                payload={"image_path": path},
                vector={config.STD_VECTOR: std_feat[j].tolist(),
                        config.FCLIP_VECTOR: f_feat[j].tolist()},
            ))
        print(f"[index] processed {i + len(batch_paths)}/{len(image_paths)}")

    # upsert in chunks so a very large corpus does not build one giant request
    for k in range(0, len(points), 256):
        client.upsert(collection_name=config.COLLECTION_NAME, points=points[k:k + 256])
    print(f"[index] done: {len(points)} images indexed into '{config.COLLECTION_NAME}'")
    return len(points)


if __name__ == "__main__":
    client = ms.get_client()
    build_index(client)
