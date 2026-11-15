import pytest

from entip_fork.keystore import KeyStore, RealIdentifierRefused


def _mint(ks, chi, acc="SYN-1", block="A1", section="L1", sha="a" * 64):
    return ks.mint(chi=chi, accession=acc, block=block, section=section, clinical_sha256=sha,
                   clinical_uri="x", original_name="f.svs", actor="t")


def test_same_patient_same_study_id(tmp_path, case):
    with KeyStore(tmp_path / "k.sqlite") as ks:
        a = _mint(ks, case.chi, sha="a" * 64)
        b = _mint(ks, case.chi, acc="SYN-2", sha="b" * 64)
        assert a.study_id == b.study_id
        assert a.specimen_id != b.specimen_id
        assert a.link_code != b.link_code


def test_same_file_reuses_link_code(tmp_path, case):
    with KeyStore(tmp_path / "k.sqlite") as ks:
        a = _mint(ks, case.chi)
        b = _mint(ks, case.chi)
        assert a.link_code == b.link_code and not b.new_scan


def test_same_file_other_section_refused(tmp_path, case):
    with KeyStore(tmp_path / "k.sqlite") as ks:
        _mint(ks, case.chi)
        with pytest.raises(ValueError):
            _mint(ks, case.chi, section="L2")


def test_accession_cannot_move_patient(tmp_path, case):
    import random
    from entip_fork import chi
    other = chi.synthetic(random.Random(99))
    with KeyStore(tmp_path / "k.sqlite") as ks:
        _mint(ks, case.chi)
        with pytest.raises(ValueError):
            _mint(ks, other, sha="c" * 64)


def test_real_chi_refused_by_default(tmp_path):
    from entip_fork.chi import mod11_check_digit
    first9 = "010180123"  # 1 Jan 1980: a real date, so could be a real person
    c = first9 + str(mod11_check_digit(first9) or 0)
    with KeyStore(tmp_path / "k.sqlite") as ks:
        if mod11_check_digit(first9) is not None:
            with pytest.raises(RealIdentifierRefused):
                _mint(ks, c)


def test_reidentify_is_audited(tmp_path, case):
    with KeyStore(tmp_path / "k.sqlite") as ks:
        m = _mint(ks, case.chi)
        with pytest.raises(ValueError):
            ks.reidentify(m.link_code, actor="t", reason=" ")
        out = ks.reidentify(m.link_code, actor="dr", reason="MDT query")
        assert out["rows"][0]["chi"] == case.chi
        log = ks.audit_log()
        assert log[-1]["action"] == "reidentify" and log[-1]["detail"]["reason"] == "MDT query"


def test_keystore_file_private(tmp_path, case):
    p = tmp_path / "k.sqlite"
    KeyStore(p).close()
    assert (p.stat().st_mode & 0o077) == 0
