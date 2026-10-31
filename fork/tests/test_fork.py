import json

import numpy as np
import pytest
import tifffile

from conftest import fork_case
from entip_fork import cli, deid, synth
from entip_fork.fork import IdentifierLeak, Stores


def _all_bytes(root):
    return b"".join(p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file())


def test_fork_end_to_end(stores, case):
    r = fork_case(stores, case)
    # research copy: one series, same pixels, no label/macro
    with tifffile.TiffFile(case.path) as a, tifffile.TiffFile(r.research_path) as b:
        assert len(b.series) == 1
        assert len(b.series[0].levels) == len(a.series[0].levels)
        for la, lb in zip(a.series[0].levels, b.series[0].levels):
            assert np.array_equal(la.asarray(), lb.asarray())
        meta = json.loads(b.pages[0].description)
        assert meta["link_code"] == r.link_code and meta["mpp"] == pytest.approx(0.25)
    assert {d.split(" ")[0] for d in r.dropped} == {"Thumbnail", "Label", "Macro"}
    # clinical copy is byte-identical and carries the same linkage code
    assert r.clinical_path.read_bytes() == case.path.read_bytes()
    link = json.loads((r.clinical_path.parent / "entip_link.json").read_text())
    assert link["link_code"] == r.link_code
    # nothing anywhere in the research store carries a direct identifier
    blob = _all_bytes(stores.research_root)
    for ident in (case.chi, case.accession, case.patient_name, case.user, "case.svs"):
        assert ident.encode() not in blob
    assert not list(stores.research_root.rglob("*.sqlite"))
    # pillar 5 hand-off
    rec = stores.registry().records()
    types = [o["type"] for o in rec[0]["objects"]]
    assert types == ["Patient", "Specimen", "Block", "Section", "Scan"]
    assert rec[0]["objects"][-1]["link_code"] == r.link_code


def test_fork_is_idempotent(stores, case):
    a = fork_case(stores, case)
    b = fork_case(stores, case)
    assert a.link_code == b.link_code and b.reused
    assert len(stores.registry().records()) == 1


def test_leak_is_refused_and_copy_deleted(stores, case):
    # put the accession into a research-side field: the scan must catch it
    with pytest.raises(IdentifierLeak):
        fork_case(stores, case, stain=f"H&E {case.accession}")
    assert not list((stores.research_root / "slides").glob("*.tiff"))
    assert stores.registry().records() == []
    with stores.keystore() as ks:
        assert ks.audit_log()[-1]["action"] == "status.refused"


def test_overlapping_stores_refused(tmp_path):
    with pytest.raises(ValueError):
        Stores(tmp_path / "a", tmp_path / "a" / "research")
    with pytest.raises(ValueError):
        Stores(tmp_path / "a", tmp_path / "a")


def test_retile_from_512_tiles_and_strips(tmp_path, stores):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (700, 900, 3), np.uint8)
    small = img[::4, ::4].copy()
    p = tmp_path / "in" / "generic.tif"
    p.parent.mkdir()
    with tifffile.TiffWriter(p) as t:
        t.write(img, tile=(512, 512), photometric="rgb", subifds=1, metadata=None,
                resolution=(40000, 40000), resolutionunit="CENTIMETER")
        t.write(small, photometric="rgb", subfiletype=1, metadata=None)  # stripped level
    out = tmp_path / "out.tiff"
    info = deid.write_research_copy(p, out, research_meta={"link_code": "L-TEST"})
    assert info.mpp == pytest.approx(0.25)
    with tifffile.TiffFile(out) as b:
        levels = b.series[0].levels
        assert np.array_equal(levels[0].asarray(), img)
        assert np.array_equal(levels[1].asarray(), small)


def test_cli_roundtrip(tmp_path, capsys):
    slide = tmp_path / "in" / "s.svs"
    assert cli.main(["synth", str(slide), "--seed", "5"]) == 0
    c = json.loads(capsys.readouterr().out)
    args = ["fork", str(slide), "--nhs-root", str(tmp_path / "nhs"), "--research-root", str(tmp_path / "res"),
            "--chi", c["chi"], "--accession", c["accession"], "--block", "A1", "--section", "L1",
            "--stain", "H&E", "--patient-name", c["patient_name"], "--actor", "t"]
    assert cli.main(args) == 0
    out = json.loads(capsys.readouterr().out)
    assert cli.main(["check", out["research_copy"], "--identifier", c["chi"], "--identifier", c["accession"]]) == 0
    capsys.readouterr()
    assert cli.main(["reidentify", out["link_code"], "--nhs-root", str(tmp_path / "nhs"),
                     "--reason", "test", "--actor", "t"]) == 0
    assert c["chi"] in capsys.readouterr().out
    assert cli.main(["check", str(slide), "--identifier", c["chi"]]) == 2


def test_batch_manifest(tmp_path, capsys):
    rows = []
    for i in range(3):
        c = synth.make_slide(tmp_path / "in" / f"s{i}.svs", seed=10 + i)
        rows.append(f"s{i}.svs,{c.chi},{c.accession},A1,L1,H&E,{c.patient_name},")
    man = tmp_path / "in" / "manifest.csv"
    man.write_text("slide,chi,accession,block,section,stain,patient_name,extra_identifiers\n" + "\n".join(rows))
    rc = cli.main(["batch", str(man), "--nhs-root", str(tmp_path / "nhs"), "--research-root", str(tmp_path / "res"), "--actor", "t"])
    assert rc == 0
    assert len(list((tmp_path / "res" / "slides").glob("*.tiff"))) == 3
