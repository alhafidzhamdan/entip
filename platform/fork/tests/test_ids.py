import random

from entip_fork import chi, ids


def test_mod11_known_value():
    # weights 10..2 over 000126123: 0+0+0+7*1? computed by hand below
    first9 = "000126123"
    total = sum(int(d) * w for d, w in zip(first9, range(10, 1, -1)))
    expect = (11 - total % 11) % 11
    assert chi.mod11_check_digit(first9) == (None if expect == 10 else expect)


def test_synthetic_chi_valid_and_not_real():
    rng = random.Random(3)
    for _ in range(200):
        c = chi.synthetic(rng)
        assert chi.is_valid(c)
        assert not chi.looks_real(c)


def test_bad_checksum_rejected():
    c = chi.synthetic(random.Random(1))
    bad = c[:9] + str((int(c[9]) + 1) % 10)
    # may still pass Luhn by chance; check that at least most corruptions fail
    fails = sum(not chi.is_valid(c[:9] + str(d)) for d in range(10) if str(d) != c[9])
    assert fails >= 8
    assert not chi.is_valid("12345")


def test_luhn_accepted():
    # a 10-digit number valid under Luhn
    base = "000126123"
    for d in range(10):
        if chi.luhn_valid(base + str(d)):
            assert chi.is_valid(base + str(d))


def test_find_candidates():
    c = chi.synthetic(random.Random(2))
    assert chi.find_candidates(f"x|Barcode = A^{c}|y") == [c]
    assert chi.find_candidates(f"{c[:6]} {c[6:]}") == [c]
    assert chi.find_candidates("1536, 2048") == []


def test_ids_roundtrip_and_typo():
    for kind in ids.KINDS:
        code = ids.new_id(kind)
        assert ids.is_valid(code)
        assert ids.is_valid(code.lower())
        body = code[2:10]
        swapped = code[:2] + body[1] + body[0] + body[2:] + code[10]
        if body[0] != body[1]:
            assert not ids.is_valid(swapped)
        wrong = code[:2] + ("0" if body[0] != "0" else "1") + code[3:]
        assert not ids.is_valid(wrong)
