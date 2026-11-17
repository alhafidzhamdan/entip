"""Analyse a blinded attention review: are the model's attended tiles diagnostic more often than random ones?

    python analyse.py key.csv scores-AH.csv [scores-XY.csv ...] [--json summary.json]

For each rater and pooled over raters:

* the share of attended tiles rated **D** (diagnostic), and **D or T** (any tumour), against the control tiles
  from the same slides, with exact (Clopper-Pearson) 95% intervals and the difference;
* the same for the attended tiles by triage class, and by whether the model's reference call was right;
* **cases where attention missed**: fewer than half of a case's attended tiles rated D or T (for a tumour case)
  or D (for a non-neoplastic case), with the model's call - the cases to look at first.

With two or more raters: Cohen's kappa for each pair, on the four scores and on diagnostic (D) or not.
Tiles left unrated, or rated A (artefact / not assessable), are counted as not diagnostic; the A share is
reported separately.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations

from scipy.stats import beta

CATS = ("D", "T", "N", "A")
NON_TUMOUR = ("normal_brain", "reactive")


def ci(k: int, n: int) -> tuple:
    if n == 0:
        return (None, None)
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return (round(lo, 3), round(hi, 3))


def share(scores: list, ok) -> dict:
    n, k = len(scores), sum(ok(s) for s in scores)
    return {"n": n, "k": k, "share": round(k / n, 3) if n else None, "ci95": ci(k, n)}


def kappa(a: list, b: list) -> float | None:
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[c] * cb[c] for c in set(a) | set(b)) / n ** 2
    return round((po - pe) / (1 - pe), 3) if pe < 1 else None


def summarise(key: dict, sc: dict) -> dict:
    """``sc``: {tile_code: score}. Tumour cases count D or T as on target; non-neoplastic cases D only."""
    att = [t for t in key.values() if t["set"] == "attended"]
    ctl = [t for t in key.values() if t["set"] == "control"]
    s = lambda ts: [sc.get(t["tile_code"], "") for t in ts]
    on = lambda t: sc.get(t["tile_code"]) in (("D",) if t["label"] in NON_TUMOUR else ("D", "T"))
    out = {"rated": sum(1 for t in key if sc.get(t)), "tiles": len(key),
           "diagnostic": {"attended": share(s(att), lambda v: v == "D"), "control": share(s(ctl), lambda v: v == "D")},
           "on_target": {"attended": share(att, on), "control": share(ctl, on)},
           "artefact": {"attended": share(s(att), lambda v: v == "A"), "control": share(s(ctl), lambda v: v == "A")}}
    for m in ("diagnostic", "on_target"):
        x, y = out[m]["attended"]["share"], out[m]["control"]["share"]
        out[m]["difference"] = round(x - y, 3) if x is not None and y is not None else None
    out["by_class"] = {c: share([t for t in att if t["label"] == c], on) for c in sorted({t["label"] for t in att})}
    out["by_reference_call"] = {k: share([t for t in att if t["reference_correct"] == v], on)
                                for k, v in (("right", "True"), ("wrong", "False"))}
    cases = defaultdict(list)
    for t in att:
        cases[t["case_code"]].append(t)
    missed = []
    for c, ts in sorted(cases.items()):
        f = sum(on(t) for t in ts) / len(ts)
        if f < 0.5 and all(sc.get(t["tile_code"]) for t in ts):
            t0 = ts[0]
            missed.append({"case": c, "patient": t0["patient"], "diagnosis": t0["diagnosis"], "on_target": round(f, 2),
                           "model_call": t0["model_call"], "reference_call": t0["reference_call"],
                           "reference_correct": t0["reference_correct"],
                           "scores": dict(Counter(sc.get(t["tile_code"]) for t in ts))})
    out["cases_where_attention_missed"] = missed
    return out


def show(name: str, s: dict) -> None:
    f = lambda d: f"{d['share']:.0%} ({d['k']}/{d['n']}, 95% CI {d['ci95'][0]:.0%}-{d['ci95'][1]:.0%})" \
        if d["n"] else "no tiles"
    print(f"== {name}: {s['rated']} of {s['tiles']} tiles rated")
    for m, label in (("diagnostic", "rated D (diagnostic)"), ("on_target", "on target (D or T; D for non-tumour)")):
        print(f"  {label}")
        print(f"    attended  {f(s[m]['attended'])}")
        print(f"    control   {f(s[m]['control'])}")
        if s[m]["difference"] is not None:
            print(f"    difference {s[m]['difference']:+.0%}")
    print(f"  artefact (A): attended {f(s['artefact']['attended'])}; control {f(s['artefact']['control'])}")
    print("  attended on target, by class: " + "; ".join(f"{c} {f(d)}" for c, d in s["by_class"].items()))
    print("  attended on target, by reference call: " + "; ".join(f"{c} {f(d)}" for c, d in s["by_reference_call"].items()))
    if s["cases_where_attention_missed"]:
        print("  cases where attention missed (under half the attended tiles on target):")
        for m in s["cases_where_attention_missed"]:
            print(f"    {m['case']} {m['patient']:<16} {m['diagnosis']}: {m['on_target']:.0%} on target, "
                  f"model called {m['model_call']}, reference call {m['reference_call']} "
                  f"({'right' if m['reference_correct'] == 'True' else 'wrong'}), scores {m['scores']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("key")
    ap.add_argument("scores", nargs="+")
    ap.add_argument("--json")
    a = ap.parse_args()
    key = {r["tile_code"]: r for r in csv.DictReader(open(a.key, newline=""))}
    raters = {}
    for p in a.scores:
        for r in csv.DictReader(open(p, newline="")):
            s = (r.get("score") or "").strip().upper()
            if r.get("tile_code") in key and s in CATS:
                raters.setdefault(r.get("rater") or p, {})[r["tile_code"]] = s
    if not raters:
        raise SystemExit("no scores matched the key")
    result = {"raters": {}}
    for name, sc in raters.items():
        result["raters"][name] = summarise(key, sc)
        show(name, result["raters"][name])
    if len(raters) > 1:
        pooled = {}
        for name, sc in raters.items():                  # every rating counts once: tile codes made unique per rater
            pooled.update({f"{t}|{name}": v for t, v in sc.items()})
        pkey = {f"{t}|{name}": {**key[t], "tile_code": f"{t}|{name}"} for name, sc in raters.items() for t in sc}
        result["pooled"] = summarise(pkey, pooled)
        show("pooled", result["pooled"])
        result["agreement"] = {}
        print("== agreement between raters (Cohen's kappa, tiles both rated)")
        for (n1, s1), (n2, s2) in combinations(raters.items(), 2):
            both = sorted(set(s1) & set(s2))
            k4 = kappa([s1[t] for t in both], [s2[t] for t in both])
            k2 = kappa([s1[t] == "D" for t in both], [s2[t] == "D" for t in both])
            result["agreement"][f"{n1} vs {n2}"] = {"tiles": len(both), "kappa_4": k4, "kappa_diagnostic": k2}
            print(f"  {n1} vs {n2}: {len(both)} tiles, kappa {k4} on the four scores, {k2} on diagnostic or not")
    if a.json:
        json.dump(result, open(a.json, "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
