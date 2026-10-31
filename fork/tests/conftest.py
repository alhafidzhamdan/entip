import pytest

from entip_fork import synth
from entip_fork.fork import Stores


@pytest.fixture
def case(tmp_path):
    return synth.make_slide(tmp_path / "incoming" / "case.svs", seed=1)


@pytest.fixture
def stores(tmp_path):
    return Stores(tmp_path / "nhs", tmp_path / "research")


def fork_case(stores, case, **kw):
    from entip_fork.fork import fork_slide

    args = dict(chi=case.chi, accession=case.accession, block=case.block, section=case.section,
                stain=case.stain, actor="test", patient_name=case.patient_name)
    args.update(kw)
    return fork_slide(stores, case.path, **args)
