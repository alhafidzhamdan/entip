"""Step 1 of the AI glioma workflow: pre-training, where the foundation model is used.

Each H&E slide is cut into tissue tiles of a fixed physical size (112 µm, 224 px at 0.5 µm/px) and every tile
is embedded once with a frozen foundation model, H-optimus-0 (Apache-2.0,
https://huggingface.co/bioptimus/H-optimus-0). The result is one bag of tile embeddings per slide, saved as
``<scan_id>.npz`` (``wsi.bags.Bag``). Steps 2 to 5 train and predict from these bags only, so the foundation
model runs once per slide, on a GPU, and everything after it runs on a CPU.

The foundation model is not changed here: no fine-tuning, no further pre-training. Changing it would change
every embedding, so each bag records the encoder version and steps 2 to 5 refuse bags from a different one.

Commands: ``entip-train embed extract`` (slides already on disk) and ``entip-train embed build-bags`` (download
each IDC slide in a manifest, embed it, delete it). Both append one line per slide to ``index.jsonl`` in the
output folder. Research use only.
"""
