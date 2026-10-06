"""Validate the explicit source, target-adaptation, and target-evaluation lists."""

from __future__ import annotations

from pathlib import Path


KNOWN_CLASSES = 7


def read_list(filename: str) -> list[tuple[str, int]]:
    path = Path(filename).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        fields = line.rsplit(maxsplit=1)
        if len(fields) != 2:
            raise ValueError(f"{path}:{line_number}: expected image path and integer label")
        image = Path(fields[0])
        if not image.is_absolute():
            raise ValueError(f"{path}:{line_number}: image paths must be absolute")
        if not image.is_file():
            raise FileNotFoundError(f"{path}:{line_number}: {image}")
        records.append((str(image), int(fields[1])))
    if not records:
        raise ValueError(f"Empty image list: {path}")
    return records


def validate_lists(source_lists: list[str], target_train_lists: list[str],
                   target_eval_lists: list[str], feature: str):
    views = 2 if "+" in feature else 1
    if any(not item or len(item) != views
           for item in (source_lists, target_train_lists, target_eval_lists)):
        raise ValueError(f"{feature} requires {views} list(s) for each split")
    source = [read_list(path) for path in source_lists]
    target_train = [read_list(path) for path in target_train_lists]
    target_eval = [read_list(path) for path in target_eval_lists]

    for split_name, groups in (("source", source), ("target_train", target_train),
                               ("target_eval", target_eval)):
        if any(len(group) != len(groups[0]) for group in groups):
            raise ValueError(f"{split_name} feature views have different lengths")
        if any([label for _, label in group] != [label for _, label in groups[0]]
               for group in groups[1:]):
            raise ValueError(f"{split_name} feature views have different labels")
        if views == 2:
            for pair in zip(*groups):
                first, second = (Path(item[0]) for item in pair)
                if first.stem != second.stem or first.parent.name != second.parent.name:
                    raise ValueError(f"Unpaired {split_name} STFT/DSCG images: {pair}")

    source_labels = [label for _, label in source[0]]
    train_labels = [label for _, label in target_train[0]]
    eval_labels = [label for _, label in target_eval[0]]
    if set(source_labels) != set(range(KNOWN_CLASSES)):
        raise ValueError("Source lists must contain labels 0 through 6")
    if set(train_labels) != {-1}:
        raise ValueError("Target adaptation lists must mask all labels as -1")
    if not set(range(KNOWN_CLASSES)).issubset(eval_labels):
        raise ValueError("Target evaluation is missing a known class")
    if any(label < 0 for label in eval_labels) or not any(
            label >= KNOWN_CLASSES for label in eval_labels):
        raise ValueError("Target evaluation needs labeled known and unknown samples")
    for train_view, eval_view in zip(target_train, target_eval):
        if [image for image, _ in train_view] != [image for image, _ in eval_view]:
            raise ValueError("Target adaptation and evaluation image lists differ")
    for source_view, target_view in zip(source, target_eval):
        if {image for image, _ in source_view} & {image for image, _ in target_view}:
            raise ValueError("Source and target image paths overlap")

    def input_path(paths: list[str]):
        resolved = [str(Path(path).expanduser().resolve()) for path in paths]
        return tuple(resolved) if views == 2 else resolved[0]

    audit = {"feature": feature, "source_samples": len(source[0]),
             "target_samples": len(target_eval[0]),
             "known_classes": KNOWN_CLASSES,
             "unknown_eval_labels": sorted(set(eval_labels) - set(range(KNOWN_CLASSES))),
             "target_train_labels_masked": True,
             "target_adaptation_and_evaluation_samples_identical": True}
    return (input_path(source_lists), input_path(target_train_lists),
            input_path(target_eval_lists), KNOWN_CLASSES + 1, audit)
