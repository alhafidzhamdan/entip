"""``entip-glioma``: the AI glioma workflow commands. The same commands are added to ``entip-train`` when this
package is installed: ``embed`` (step 1), ``triage`` (step 2), ``highgrade*`` (step 3), ``methyl`` (step 5),
and ``distill`` (a small student encoder distilled from H-optimus-0).
Step 4 has its own command, ``entip-panel``."""
from __future__ import annotations

import argparse
import sys


def add_parsers(sub) -> None:
    from .distill.cli import add_parser as add_distill
    from .embed.cli import add_parser as add_embed
    from .highgrade.cli import add_parser as add_highgrade
    from .methylation.cli import add_parser as add_methyl
    from .triage.cli import add_parser as add_triage
    add_embed(sub)
    add_triage(sub)
    add_methyl(sub)
    add_highgrade(sub)
    add_distill(sub)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("entip-glioma")
    add_parsers(ap.add_subparsers(dest="cmd", required=True))
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
