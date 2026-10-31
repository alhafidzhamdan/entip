"""Random identifiers for the research side.

IDs are random (not derived from CHI or accession), so they cannot be reversed
without the NHS-side key store. They use Crockford base32 (no I, L, O, U) and end
with a Luhn mod-32 check character, so a mistyped code is caught.

Format: ``<kind>-<8 chars><check>``, e.g. ``L-7K3M9Q2XH``. 40 random bits per ID.
"""

from __future__ import annotations

import secrets

ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_N = len(ALPHABET)

KINDS = {
    "P": "patient (study_id)",
    "S": "specimen",
    "B": "block",
    "T": "section",
    "L": "scan linkage code",
}


def _luhn_mod_n(body: str) -> str:
    factor = 2
    total = 0
    for ch in reversed(body):
        addend = factor * ALPHABET.index(ch)
        factor = 1 if factor == 2 else 2
        total += addend // _N + addend % _N
    return ALPHABET[(_N - total % _N) % _N]


def new_id(kind: str) -> str:
    if kind not in KINDS:
        raise ValueError(f"unknown id kind {kind!r}")
    body = "".join(secrets.choice(ALPHABET) for _ in range(8))
    return f"{kind}-{body}{_luhn_mod_n(body)}"


def normalise(code: str) -> str:
    """Upper-case and map look-alike letters (O->0, I/L->1) as Crockford allows."""
    code = code.strip().upper().replace("O", "0").replace("I", "1").replace("L", "1")
    # the kind prefix is a letter, so restore it after mapping
    if len(code) > 1 and code[1] == "-" and code[0] == "1":
        code = "L" + code[1:]
    return code


def is_valid(code: str) -> bool:
    code = normalise(code)
    if len(code) != 11 or code[1] != "-" or code[0] not in KINDS:
        return False
    body, check = code[2:10], code[10]
    if any(ch not in ALPHABET for ch in body + check):
        return False
    return _luhn_mod_n(body) == check
