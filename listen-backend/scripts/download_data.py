"""Instructions and helpers for obtaining LISTEN datasets."""

from __future__ import annotations

import argparse

from app.data.loaders import download_qmsum_instructions, load_ami_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="LISTEN dataset utilities")
    parser.add_argument(
        "--dataset",
        choices=["ami", "qmsum-info"],
        required=True,
        help="Which dataset action to run",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-meetings", type=int, default=1)
    args = parser.parse_args()

    if args.dataset == "qmsum-info":
        print(download_qmsum_instructions())
    elif args.dataset == "ami":
        records = load_ami_corpus(split=args.split, max_meetings=args.max_meetings)
        print(f"Loaded {len(records)} AMI records from split '{args.split}'.")


if __name__ == "__main__":
    main()
