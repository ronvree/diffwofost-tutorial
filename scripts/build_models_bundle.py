"""Build the models_bundle.zip committed under data/ for the notebook to fetch.

Only the pretrained model weights live in this bundle (~96 KB). The field-trial
files (also under data/) are mirrored verbatim from Harvard Dataverse
(doi:10.7910/DVN/1LC6W7); PCSE stock configs are fetched from upstream at
runtime.

Usage:
    python scripts/build_models_bundle.py [--src PATH] [--out PATH]

Defaults --src to ../diffWOFOST (sibling checkout) and --out to
<repo>/data/models_bundle.zip.
"""
import argparse
import zipfile
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent

FILES = [
    ("docs/notebooks/data_temp/trained_models/stress_nn_random.pt", "stress_nn_random.pt"),
    ("docs/notebooks/data_temp/trained_models/pure_lstm_random.pt", "pure_lstm_random.pt"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=REPO_ROOT.parent / "diffWOFOST",
                    help="Path to a diffwofost checkout (default: ../diffWOFOST)")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "models_bundle.zip",
                    help="Output zip path (default: <repo>/data/models_bundle.zip)")
    args = ap.parse_args()

    if not args.src.is_dir():
        sys.exit(f"diffwofost checkout not found at {args.src}")

    missing = [rel for rel, _ in FILES if not (args.src / rel).is_file()]
    if missing:
        sys.exit("Missing source files:\n  " + "\n  ".join(missing))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel, dest in FILES:
            zf.write(args.src / rel, dest)
            print(f"  + {dest}")

    size_kb = args.out.stat().st_size / 1024
    print(f"\nWrote {args.out}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
