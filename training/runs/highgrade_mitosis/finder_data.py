"""Step 3c: fetch and prepare the public data for the first mitosis finder.

    python finder_data.py fetch --data mitosis-data
    python finder_data.py prepare --data mitosis-data --out mitosis-finder1

**fetch** downloads the stMIDOG++ and LUNG-MITO annotations (Zenodo 18661323, CC BY 4.0), the LUNG-MITO images, and the
MIDOG++ images named in ``finder_sources.csv`` (figshare, CC0). Every file is checked against its published MD5. Five
of the 503 MIDOG++ images have no CC0 record in the collection, so they are not in that list and are never used.

**prepare** writes ``index.json`` and ``summary.json`` and fixes the split once:

* Mitosis: prometaphase, metaphase, anaphase, telophase and atypical. Look-alike: the datasets' negative class.
* Prophase is ignored: neither a mitosis nor a look-alike, in training or scoring.
* ``unseen``: every human melanoma and human neuroendocrine tumour case, kept out of training and tuning.
* ``tune``: 15% of the other MIDOG++ cases, by tumour type.
* ``train``: the remaining MIDOG++ cases and all of LUNG-MITO. LUNG-MITO's tiles carry no patient identifiers, so it
  can never be split safely and is used for training only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import sys
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCES = HERE / "finder_sources.csv"
ZENODO = "https://zenodo.org/api/records/18661323/files/{}/content"
FIGSHARE = "https://ndownloader.figshare.com/files/{}"
ZENODO_FILES = {"MIDOGpp_subtyping.json": "08eab113df290ec40492981e3dbf0b46",
                "Lung_mito_extended.zip": "477ae480b36d2db3175ed5a02101d34d"}
ROLE = {1: "ignore", 2: "pos", 3: "pos", 4: "pos", 5: "pos", 6: "neg", 7: "pos"}
UNSEEN = {"human melanoma", "human neuroendocrine tumor"}
LUNG_MEMBER = re.compile(r"^(Lung_mito_extended/\d{6}\.png|Lung_mito_extended\.json)$")


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, path: Path, want_md5: str, tries: int = 5) -> str:
    """Download to ``path`` once, checked against its MD5. A finished file is marked by ``<name>.md5``."""
    mark = path.with_name(path.name + ".md5")
    if path.exists() and mark.exists() and mark.read_text().strip() == want_md5:
        return "kept"
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research-data-client"})
            with urllib.request.urlopen(req, timeout=120) as r, part.open("wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            got = md5(part)
            if got != want_md5:
                raise ValueError(f"MD5 {got}, expected {want_md5}")
            part.replace(path)
            mark.write_text(want_md5 + "\n")
            return "downloaded"
        except Exception as e:  # network errors and bad checksums are both retried
            last = e
            time.sleep(5 * (attempt + 1))
    part.unlink(missing_ok=True)
    raise SystemExit(f"could not download {path.name}: {last}")


def read_sources() -> list:
    rows = list(csv.DictReader(open(SOURCES, newline="")))
    if not rows or any(r["licence"] != "CC0" for r in rows):
        raise SystemExit("finder_sources.csv must list CC0 images only")
    return rows


def extract_lung(data: Path) -> int:
    """Only the expected PNG and JSON names are written, so an archive cannot place a file elsewhere."""
    out = data / "lung"
    done = out / ".complete"
    if done.exists():
        return len(list(out.glob("*.png")))
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(data / "Lung_mito_extended.zip") as z:
        for m in z.infolist():
            if m.is_dir():
                continue
            if not LUNG_MEMBER.match(m.filename):
                raise SystemExit(f"unexpected file in the LUNG-MITO archive: {m.filename}")
            with z.open(m) as src, (out / Path(m.filename).name).open("wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            n += m.filename.endswith(".png")
    done.write_text(f"{n}\n")
    return n


def fetch(a) -> int:
    data = Path(a.data)
    for name, want in ZENODO_FILES.items():
        print(name, download(ZENODO.format(name), data / name, want), flush=True)
    rows = read_sources()
    with ThreadPoolExecutor(a.workers) as pool:
        jobs = [pool.submit(download, FIGSHARE.format(r["figshare_file"]), data / "midog" / r["file_name"], r["md5"])
                for r in rows]
        for n, j in enumerate(as_completed(jobs), start=1):
            j.result()
            if n % 25 == 0 or n == len(jobs):
                print(f"MIDOG++ images {n}/{len(jobs)}", flush=True)
    print(json.dumps({"midog_images": len(rows), "lung_images": extract_lung(data)}, indent=2))
    return 0


def objects(anns: list) -> list:
    out = []
    for a in anns:
        x, y, w, h = (float(v) for v in a["bbox"])
        rings = [[round(float(v), 1) for v in ring] for ring in a["segmentation"] if len(ring) >= 6]
        if not rings:
            raise SystemExit(f"annotation {a['id']} has no usable mask")
        out.append({"role": ROLE[int(a["category_id"])], "x": round(x + w / 2, 1), "y": round(y + h / 2, 1),
                    "rings": rings})
    return out


def by_image(coco: dict) -> dict:
    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[str(a["image_id"])].append(a)
    return anns


def split_midog(names_by_tumour: dict, tune: float, seed: int) -> dict:
    split = {}
    for tumour in sorted(names_by_tumour):
        names = sorted(names_by_tumour[tumour])
        if tumour in UNSEEN:
            split.update({n: "unseen" for n in names})
            continue
        random.Random(f"{seed}-{tumour}").shuffle(names)
        k = max(1, round(tune * len(names)))
        split.update({n: "tune" if i < k else "train" for i, n in enumerate(names)})
    return split


def decode_tiff(path: str) -> str:
    """Write ``<name>.rgb.npy`` once, so training can read small crops without decoding a whole compressed TIFF."""
    import numpy as np
    from PIL import Image
    cached = Path(path + ".rgb.npy")
    if cached.exists():
        return str(cached)
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"))
    tmp = cached.with_name(f"{cached.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as f:
        np.save(f, arr)
    os.replace(tmp, cached)
    return str(cached)


def prepare(a) -> int:
    from PIL import Image
    data, out = Path(a.data), Path(a.out)
    if (out / "index.json").exists():
        raise SystemExit(f"{out / 'index.json'} already exists: the split is fixed once")
    allowed = {r["file_name"] for r in read_sources()}
    coco = json.load(open(data / "MIDOGpp_subtyping.json"))
    anns = by_image(coco)
    cases = [i for i in coco["images"] if i["file_name"] in allowed]
    missing = [i["file_name"] for i in cases if not (data / "midog" / i["file_name"]).exists()]
    if missing and not a.allow_missing:
        raise SystemExit(f"{len(missing)} MIDOG++ images have not been fetched")
    cases = [i for i in cases if i["file_name"] not in set(missing)]
    groups = defaultdict(list)
    for i in cases:
        groups[i["tumor_type"]].append(i["file_name"])
    split = split_midog(groups, a.tune, a.seed)
    images = []
    for i in cases:
        images.append({"id": f"midog:{i['file_name']}", "path": f"midog/{i['file_name']}", "source": "midog",
                       "tumour": i["tumor_type"], "split": split[i["file_name"]], "w": int(i["width"]),
                       "h": int(i["height"]), "objects": objects(anns.get(str(i["id"]), []))})
    lung = json.load(open(data / "lung" / "Lung_mito_extended.json"))
    lung_anns = by_image(lung)
    for i in lung["images"]:
        p = data / "lung" / i["file_name"]
        if not p.exists():
            if a.allow_missing:
                continue
            raise SystemExit(f"missing LUNG-MITO image {i['file_name']}")
        with Image.open(p) as im:          # about 40% of the records give a size of 0, so read it from the file
            w, h = im.size
        images.append({"id": f"lung:{i['file_name']}", "path": f"lung/{i['file_name']}", "source": "lung",
                       "tumour": "human lung adenocarcinoma", "split": "train", "w": w, "h": h,
                       "objects": objects(lung_anns.get(str(i["id"]), []))})
    if not images:
        raise SystemExit("no images to prepare")
    tiffs = [str(data / im["path"]) for im in images if im["source"] == "midog"]
    with ProcessPoolExecutor(max(1, a.workers)) as pool:
        for n, _ in enumerate(pool.map(decode_tiff, tiffs), start=1):
            if n % 50 == 0 or n == len(tiffs):
                print(f"decoded {n}/{len(tiffs)} images", file=sys.stderr, flush=True)
    summary = defaultdict(lambda: {"images": 0, "mitoses": 0, "look_alikes": 0, "prophase_ignored": 0})
    for im in images:
        for key in (f"split:{im['split']}", f"{im['split']}:{im['tumour']}"):
            s = summary[key]
            s["images"] += 1
            c = Counter(o["role"] for o in im["objects"])
            s["mitoses"] += c["pos"]
            s["look_alikes"] += c["neg"]
            s["prophase_ignored"] += c["ignore"]
    out.mkdir(parents=True, exist_ok=True)
    with (out / "index.json").open("w") as f:
        json.dump({"data": str(data.resolve()), "seed": a.seed, "images": images}, f, separators=(",", ":"))
    (out / "summary.json").write_text(json.dumps(dict(sorted(summary.items())), indent=2) + "\n")
    print(json.dumps({k: v for k, v in sorted(summary.items()) if k.startswith("split:")}, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download and checksum the public data")
    f.add_argument("--data", required=True)
    f.add_argument("--workers", type=int, default=4)
    f.set_defaults(fn=fetch)
    p = sub.add_parser("prepare", help="write the labels and fix the split")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tune", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=20261126)
    p.add_argument("--workers", type=int, default=int(os.environ.get("NSLOTS", 1)))
    p.add_argument("--allow-missing", action="store_true", help="tests only: prepare from a partial download")
    p.set_defaults(fn=prepare)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
