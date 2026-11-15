"""Fork one WSI into a clinical copy and a research copy, joined by a linkage code.

Layout (two roots that must not overlap):

    <nhs_root>/                      NHS side, never shared
        keystore.sqlite              CHI / accession <-> research IDs, audit log
        clinical/<link_code>/<original file name>
        clinical/<link_code>/entip_link.json   linkage code for the clinical system

    <research_root>/                 research side
        slides/<link_code>.tiff      pseudonymised pyramid
        outbox/p5.jsonl              objects for the central database (pillar 5)
        actions.jsonl                logged actions, no identifiers

Steps: hash the source, mint IDs, copy the clinical file unchanged, write the
research copy, scan it for identifiers (delete it and stop on any finding),
then register Patient..Scan with pillar 5 and log the action.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from . import deid, registry
from .keystore import KeyStore


class IdentifierLeak(RuntimeError):
    pass


def sha256_file(path: str | Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while b := fh.read(chunk):
            h.update(b)
    return h.hexdigest()


def _overlaps(a: Path, b: Path) -> bool:
    a, b = a.resolve(), b.resolve()
    return a == b or a in b.parents or b in a.parents


@dataclass
class Stores:
    nhs_root: Path
    research_root: Path
    allow_real_identifiers: bool = False

    def __post_init__(self):
        self.nhs_root = Path(self.nhs_root)
        self.research_root = Path(self.research_root)
        if _overlaps(self.nhs_root, self.research_root):
            raise ValueError("NHS store and research store must be separate folders, neither inside the other")

    @property
    def keystore_path(self) -> Path:
        return self.nhs_root / "keystore.sqlite"

    def keystore(self) -> KeyStore:
        return KeyStore(self.keystore_path, allow_real_identifiers=self.allow_real_identifiers)

    def registry(self) -> registry.OutboxRegistry:
        return registry.OutboxRegistry(self.research_root / "outbox" / "p5.jsonl")

    def log_action(self, action: dict) -> None:
        registry.assert_no_identifiers(action)
        self.research_root.mkdir(parents=True, exist_ok=True)
        with open(self.research_root / "actions.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(action, sort_keys=True) + "\n")


@dataclass
class ForkResult:
    link_code: str
    study_id: str
    research_path: Path
    clinical_path: Path
    reused: bool
    dropped: list[str] = field(default_factory=list)


def fork_slide(
    stores: Stores,
    slide: str | Path,
    *,
    chi: str,
    accession: str,
    block: str,
    section: str,
    stain: str,
    actor: str,
    patient_name: str | None = None,
    extra_identifiers: list[str] | None = None,
    compression: str = "zlib",
) -> ForkResult:
    slide = Path(slide)
    digest = sha256_file(slide)
    with stores.keystore() as ks:
        clinical_dir = stores.nhs_root / "clinical"
        minted = ks.mint(
            chi=chi, accession=accession, block=block, section=section,
            clinical_sha256=digest, clinical_uri="pending", original_name=slide.name, actor=actor,
        )
        code = minted.link_code
        clinical_path = clinical_dir / code / slide.name
        research_path = stores.research_root / "slides" / f"{code}.tiff"
        if not minted.new_scan:
            row = ks.scan(code)
            if row["status"] == "forked" and research_path.exists():
                return ForkResult(code, minted.study_id, research_path, Path(row["clinical_uri"]), True)

        # clinical copy: unchanged bytes, plus the linkage code beside it
        clinical_path.parent.mkdir(parents=True, exist_ok=True)
        if not clinical_path.exists():
            shutil.copy2(slide, clinical_path)
        if sha256_file(clinical_path) != digest:
            raise RuntimeError("clinical copy does not match the source")
        (clinical_path.parent / "entip_link.json").write_text(
            json.dumps({"link_code": code, "study_id": minted.study_id, "accession": accession,
                        "block": block, "section": section, "sha256": digest}, indent=2)
        )
        ks.db.execute("UPDATE scans SET clinical_uri = ? WHERE link_code = ?", (str(clinical_path), code))
        ks.db.commit()

        # research copy
        meta = {"link_code": code, "study_id": minted.study_id, "section_id": minted.section_id, "stain": stain}
        info = deid.write_research_copy(slide, research_path, research_meta=meta, compression=compression)
        identifiers = [chi, accession, slide.name, slide.stem, *(extra_identifiers or [])]
        if patient_name:
            identifiers.append(patient_name)
        findings = deid.scan_for_identifiers(research_path, identifiers)
        if findings:
            research_path.unlink(missing_ok=True)
            ks.set_status(code, "refused", actor, {"findings": len(findings)})
            stores.log_action(registry.build_action(
                "fork_slide.refused", actor, inputs={"link_code": code}, outputs={"findings": len(findings)}))
            raise IdentifierLeak("; ".join(findings))
        os.chmod(research_path, 0o644)
        research_sha = sha256_file(research_path)
        ks.set_research_copy(code, str(research_path), research_sha, actor)

    scan = {
        "uri": str(research_path.relative_to(stores.research_root)),
        "sha256": research_sha,
        "width": info.width,
        "height": info.height,
        "levels": [list(lv) for lv in info.levels],
        "mpp": info.mpp,
        "magnification": info.magnification,
        "source_format": info.vendor,
        "frame": "scan_px_level0",
    }
    objects = registry.build_objects(minted=minted, block=block, section=section, stain=stain, scan=scan, actor=actor)
    action = registry.build_action(
        "fork_slide", actor,
        inputs={"link_code": code},
        outputs={"research_sha256": research_sha, "dropped_images": info.dropped},
    )
    stores.registry().register(objects, action)
    stores.log_action(action)
    return ForkResult(code, minted.study_id, research_path, clinical_path, False, info.dropped)
