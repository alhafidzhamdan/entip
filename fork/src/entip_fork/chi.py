"""CHI (Community Health Index) number checks.

Structure and check digits follow Public Health Scotland's phsmethods notes
(https://code.publichealthscotland.scot/phsmethods/articles/chi-operations.html):
10 digits, DDMMYY date of birth, two digits, a sex digit, and a check digit.
The check digit is Modulus-11 (weights 10..2), and from August 2026 a new CHI may
instead carry a Modulus-10 (Luhn) check digit. We accept either.

Synthetic CHIs made here use day "00", which is not a real date of birth, so they
cannot belong to a living person. By default the key store refuses any CHI whose
first six digits form a real date (see ``looks_real``).
"""

from __future__ import annotations

import datetime as _dt
import random as _random
import re

_CHI_RE = re.compile(r"^\d{10}$")


def normalise(chi: str) -> str:
    """Strip spaces and dashes, e.g. '000126 1234' -> '0001261234'."""
    return re.sub(r"[\s-]", "", str(chi))


def mod11_check_digit(first9: str) -> int | None:
    """Modulus-11 check digit, or None when the result is 10 (no valid digit)."""
    total = sum(int(d) * w for d, w in zip(first9, range(10, 1, -1)))
    result = 11 - (total % 11)
    if result == 11:
        return 0
    if result == 10:
        return None
    return result


def luhn_valid(number: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(number)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def is_valid(chi: str) -> bool:
    """Format plus check digit (Modulus-11 or Modulus-10/Luhn)."""
    chi = normalise(chi)
    if not _CHI_RE.match(chi):
        return False
    check = mod11_check_digit(chi[:9])
    if check is not None and check == int(chi[9]):
        return True
    return luhn_valid(chi)


def looks_real(chi: str) -> bool:
    """True when the first six digits are a real calendar date (DDMMYY)."""
    chi = normalise(chi)
    try:
        _dt.datetime.strptime(chi[:6], "%d%m%y")
    except ValueError:
        return False
    return True


def synthetic(rng: _random.Random | None = None) -> str:
    """A CHI-shaped number with a valid Modulus-11 digit and day '00'."""
    rng = rng or _random.Random()
    while True:
        first9 = "00" + f"{rng.randint(1, 12):02d}" + f"{rng.randint(0, 99):02d}" + f"{rng.randint(0, 999):03d}"
        check = mod11_check_digit(first9)
        if check is not None:
            return first9 + str(check)


def find_candidates(text: str) -> list[str]:
    """All 10-digit runs in ``text`` (allowing one space or dash after DDMMYY) that pass ``is_valid``."""
    hits = []
    for m in re.finditer(r"(?<!\d)(\d{6})[ -]?(\d{4})(?!\d)", text):
        cand = m.group(1) + m.group(2)
        if is_valid(cand):
            hits.append(cand)
    return hits
