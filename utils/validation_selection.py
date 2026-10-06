"""Explicit held-out target splits and validation-only HOS checkpoint selection."""
import csv
import hashlib
import json
import math
from pathlib import Path

METRICS = ('OS', 'OS_star', 'UNK', 'HOS', 'AUROC', 'AUPR', 'OSCR')


def read_target_splits(manifest, known_classes, feature_count):
    """Require absolute image paths and recording-group IDs; never invent groups."""
    path = Path(manifest).resolve()
    spec = json.loads(path.read_text())
    result, audit, seen_paths, seen_groups, seen_inodes = {}, {}, set(), set(), set()
    for split in ('train', 'validation', 'test'):
        entry = spec[split]
        lists = [(path.parent / p).resolve() for p in entry['lists']]
        if len(lists) != feature_count:
            raise ValueError(f'{split}: expected {feature_count} feature lists')
        group_file = (path.parent / entry['groups']).resolve()
        groups = group_file.read_text().splitlines()
        labels, paths, sample_ids = None, set(), None
        inodes = set()
        for file in lists:
            values = []
            identities = []
            modality_paths = set()
            for line in file.read_text().splitlines():
                image, label = line.rsplit('\t', 1)
                image_path = Path(image)
                if not image_path.is_absolute() or not image_path.is_file():
                    raise ValueError(f'{split}: image paths must be absolute existing files: {image}')
                image_path = image_path.resolve()
                if image_path in modality_paths:
                    raise ValueError(f'{split}: duplicate sample {image_path}')
                modality_paths.add(image_path)
                stat = image_path.stat()
                inodes.add((stat.st_dev, stat.st_ino))
                # Existing STFT/DSCG lists share device directory and sample filename.
                identities.append((image_path.parent.name, image_path.name))
                values.append(int(label))
            if labels is not None and (labels != values or identities != sample_ids):
                raise ValueError(f'{split}: feature lists are not sample-aligned')
            labels, sample_ids = values, identities
            paths.update(modality_paths)
        if not labels or len(groups) != len(labels) or any(not g.strip() for g in groups):
            raise ValueError(f'{split}: need one nonempty recording-group ID per sample')
        group_set = set(groups)
        if paths & seen_paths or inodes & seen_inodes or group_set & seen_groups:
            raise ValueError(f'{split}: samples or acquisition groups overlap across splits')
        if min(labels) < 0 or not set(range(known_classes)).issubset(labels) or not any(y >= known_classes for y in labels):
            raise ValueError(f'{split}: require every known class and at least one unknown class')
        seen_paths.update(paths); seen_inodes.update(inodes); seen_groups.update(group_set)
        result[split] = tuple(map(str, lists)) if feature_count > 1 else str(lists[0])
        audit[split] = dict(samples=len(labels),groups=len(group_set),
                           sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in lists + [group_file]})
    audit['manifest'] = str(path)
    audit['manifest_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    audit['protocol'] = 'disjoint target adaptation/labelled validation/test; recording-group disjoint'
    return result, audit


def audit_current_split(source_lists, target_lists):
    """Describe the original shared-target protocol without claiming holdout data."""
    def describe(lists):
        if isinstance(lists, str):
            lists = [lists]
        contents = [Path(p).read_bytes() for p in lists]
        labels = [[line.rsplit('\t', 1)[1] for line in data.decode().splitlines()] for data in contents]
        if not labels[0] or any(x != labels[0] for x in labels[1:]):
            raise ValueError('Current feature lists are empty or labels are not aligned')
        return dict(samples=len(labels[0]), lists=[Path(p).name for p in lists],
                    sha256={Path(p).name: hashlib.sha256(data).hexdigest() for p,data in zip(lists,contents)})
    return dict(protocol='development_hos', independent_validation=False,
                target_train_validation_test_overlap=True,
                source=describe(source_lists), target=describe(target_lists))


class HOSSelector:
    """Monitor every metric; only validation HOS determines the saved checkpoint."""
    def __init__(self, root, protocol='validation_hos'):
        if protocol not in {'validation_hos', 'development_hos'}:
            raise ValueError('Unknown selection protocol')
        self.role = 'development' if protocol == 'development_hos' else 'validation'
        self.selection_metric = self.role + '_HOS'
        self.checkpoint_name = 'best_' + self.role + '_model.pt'
        self.history_name = self.role + '_metrics.csv'
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if (self.root / 'selection.json').exists() or (self.root / self.history_name).exists():
            raise ValueError('Validation selection needs a fresh result directory')
        self.history = []
        self.best = None

    def observe(self, epoch, values, modules):
        import torch
        if len(values) != len(METRICS) or any(v is None or not math.isfinite(float(v)) or not 0 <= v <= 1 for v in values):
            raise ValueError('Validation metrics must all be finite and in [0, 1]')
        if self.history and epoch <= self.history[-1]['epoch']:
            raise ValueError('Validation epochs must be strictly increasing')
        record = dict(epoch=int(epoch), **dict(zip(METRICS, map(float, values))))
        self.history.append(record)
        # Strict > keeps the earliest epoch on an exact HOS tie. No test-based tie break.
        if self.best is None or record['HOS'] > self.best['HOS']:
            temp = self.root / ('best_' + self.role + '_model.tmp')
            torch.save({name: module.state_dict() for name, module in modules.items()}, temp)
            temp.replace(self.root / self.checkpoint_name)
            self.best = record.copy()
        with (self.root / self.history_name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch', *METRICS])
            writer.writeheader(); writer.writerows(self.history)
        (self.root / 'selection.json').write_text(json.dumps(dict(
            selection_metric=self.selection_metric, independent_validation=self.role == 'validation',
            target_train_evaluation_overlap=self.role == 'development', tie_rule='earliest_epoch', best=self.best,
            evaluated_epochs=[r['epoch'] for r in self.history], test_evaluated=False), indent=2))
        return record

    def load_best(self, modules, device):
        import torch
        if self.best is None:
            raise RuntimeError('No validation checkpoint selected')
        state = torch.load(self.root / self.checkpoint_name, map_location=device, weights_only=True)
        for name, module in modules.items():
            module.load_state_dict(state[name])


def unknown_loss_coefficient(base, epoch, ramp_epochs=0):
    """Optional linear warmup; 0 preserves the original fixed coefficient."""
    if base < 0 or not math.isfinite(base) or ramp_epochs < 0:
        raise ValueError('Unknown-loss coefficient and ramp duration must be nonnegative')
    return base * min(1.0, epoch / ramp_epochs) if ramp_epochs else base
