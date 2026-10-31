"""Stage compact model packages locally; never contacts the Hub or uploads files."""

import argparse
from pathlib import Path

from jais_guard.publication.models import finalize_validation, prepare

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True, help="New staging directory")
parser.add_argument("--base-license-review", type=Path, required=True)
parser.add_argument("--validation-reviews", type=Path)
parser.add_argument("--validated-packages", type=Path)
args = parser.parse_args()
if bool(args.validation_reviews) != bool(args.validated_packages):
    parser.error("Supply both --validation-reviews and --validated-packages")
root = Path(__file__).resolve().parents[1]
print(prepare(root, args.output, args.base_license_review))
if args.validation_reviews:
    finalize_validation(root, args.output, args.validation_reviews, args.validated_packages)
