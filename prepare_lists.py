"""Create paired STFT/DSCG lists for one PRACH transfer without copying data.

Expected layout: DATA_ROOT/D1/STFT/UE1/*.png and
DATA_ROOT/D1/DSCG/UE1/*.png (likewise for the other domains and UEs).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FEATURES = ("stft", "dscg")


def natural_key(value: str):
    return (0, int(value)) if value.isdigit() else (1, value)


def paired_images(root: Path, domain: str, ue: int, limit: int):
    views = {}
    for feature in FEATURES:
        folder = root / domain / feature.upper() / f"UE{ue}"
        files = {path.stem: path.resolve() for path in folder.glob("*.png")}
        if not files:
            raise FileNotFoundError(f"No PNG images in {folder}")
        views[feature] = files
    if set(views["stft"]) != set(views["dscg"]):
        raise ValueError(f"STFT/DSCG sample IDs differ for {domain}/UE{ue}")
    ids = sorted(views["stft"], key=natural_key)
    if limit:
        ids = ids[:limit]
    return [{feature: str(views[feature][sample_id]) for feature in FEATURES}
            for sample_id in ids]


def write_list(path: Path, records: list[tuple[str, int]]) -> None:
    path.write_text("".join(f"{image}\t{label}\n" for image, label in records))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-domain", required=True)
    parser.add_argument("--target-domain", required=True)
    parser.add_argument("--known-ues", nargs="+", type=int, default=list(range(1, 8)))
    parser.add_argument("--unknown-ues", nargs="+", type=int, default=[8])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples-per-ue", type=int, default=0,
                        help="0 uses all paired images; smaller values are for smoke runs")
    args = parser.parse_args()
    root = args.data_root.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Data root does not exist: {root}")
    if args.source_domain == args.target_domain:
        parser.error("Source and target domains must differ")
    if len(args.known_ues) != 7 or len(set(args.known_ues)) != 7:
        parser.error("Exactly seven distinct known UEs are required")
    if not args.unknown_ues or len(set(args.unknown_ues)) != len(args.unknown_ues):
        parser.error("Provide distinct unknown UEs")
    if set(args.known_ues) & set(args.unknown_ues):
        parser.error("Known and unknown UEs must be disjoint")
    if args.max_samples_per_ue < 0:
        parser.error("--max-samples-per-ue must be nonnegative")
    if out.exists() and any(out.iterdir()):
        parser.error(f"Refusing to overwrite nonempty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)

    source = {feature: [] for feature in FEATURES}
    target_train = {feature: [] for feature in FEATURES}
    target_eval = {feature: [] for feature in FEATURES}
    counts = {}
    for label, ue in enumerate(args.known_ues):
        for item in paired_images(root, args.source_domain, ue, args.max_samples_per_ue):
            for feature in FEATURES:
                source[feature].append((item[feature], label))
        target_items = paired_images(root, args.target_domain, ue, args.max_samples_per_ue)
        counts[f"target_UE{ue}"] = len(target_items)
        for item in target_items:
            for feature in FEATURES:
                target_train[feature].append((item[feature], -1))
                target_eval[feature].append((item[feature], label))
    for offset, ue in enumerate(args.unknown_ues):
        target_items = paired_images(root, args.target_domain, ue, args.max_samples_per_ue)
        counts[f"target_UE{ue}"] = len(target_items)
        for item in target_items:
            for feature in FEATURES:
                target_train[feature].append((item[feature], -1))
                target_eval[feature].append((item[feature], 7 + offset))

    for split, values in (("source", source), ("target_train", target_train),
                          ("target_eval", target_eval)):
        for feature, records in values.items():
            write_list(out / f"{split}_{feature}.txt", records)
    file_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in out.glob("*.txt")}
    metadata = {"data_root": str(root), "source_domain": args.source_domain,
                "target_domain": args.target_domain, "known_ues": args.known_ues,
                "unknown_ues": args.unknown_ues,
                "source_samples": len(source["stft"]),
                "target_samples": len(target_eval["stft"]),
                "target_adaptation_labels_masked": True,
                "target_adaptation_and_evaluation_samples_identical": True,
                "per_ue_counts": counts, "list_sha256": file_hashes}
    (out / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote paired lists to {out}")
    print(f"Source {metadata['source_samples']} images; target {metadata['target_samples']} images")


if __name__ == "__main__":
    main()
