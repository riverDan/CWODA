from __future__ import print_function
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import random
import datetime
import numpy as np
import time
import datetime
import warnings
import sys
warnings.filterwarnings("ignore")


def main(args):
    from utils import utils as utils
    utils.setGPU(args.set_gpu)

    import torch
    from data_loader.base import UDADataset
    from data_protocol import validate_lists

    t1 = time.time()
    args.eval_interval = max(1, int(args.eval_interval))
    args.prefetch_factor = max(1, int(args.prefetch_factor))
    args.openmax_refresh_interval = max(1, int(args.openmax_refresh_interval))
    args.summary_windows = utils.parse_summary_windows(args.summary_windows)
    torch.manual_seed(args.seed)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.set_device(0)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = args.deterministic
    torch.backends.cudnn.benchmark = args.cuda and not args.deterministic
    args.device = torch.device("cuda:0" if args.cuda else "cpu")

    torch.set_num_threads(1)

    source_data, target_data, target_eval_data, num_class, data_audit = validate_lists(
        args.source_lists, args.target_train_lists, args.target_eval_lists, args.feature)
    args.unks = f"unk{len(data_audit['unknown_eval_labels'])}"
    sum_str = ''.join(
        '{:>20} : {:<20} \n'.format(name, str(getattr(args, name)))
        for name in sorted(vars(args))
    )
    validation_selection = getattr(args, 'selection_protocol', 'legacy') in {'validation_hos', 'development_hos'}
    split_audit = None
    if validation_selection:
        if args.model != 'CWODA':
            raise ValueError('HOS model selection requires CWODA')
        if args.max_batches:
            raise ValueError('Partial-batch evaluation is not allowed for validation model selection')
        import math
        if args.training_iter < 1 or args.unknown_loss_ramp_epochs < 0 or not math.isfinite(args.lamda3) or args.lamda3 < 0:
            raise ValueError('Positive epoch budget and finite nonnegative unknown-loss settings required')
        if args.selection_protocol == 'validation_hos':
            if not args.target_split_manifest:
                raise ValueError('validation_hos requires --target_split_manifest')
            from utils.validation_selection import read_target_splits
            split_paths, split_audit = read_target_splits(
                args.target_split_manifest, num_class - 1, 2 if '+' in args.feature else 1)
        else:
            if args.target_split_manifest:
                raise ValueError('development_hos uses the existing lists, not a split manifest')
            from utils.validation_selection import audit_current_split
            split_paths = dict.fromkeys(('train', 'validation', 'test'), target_data)
            split_audit = audit_current_split(source_data, target_data)
    elif getattr(args, 'target_split_manifest', ''):
        raise ValueError('A target split manifest requires --selection_protocol validation_hos')
    train_transforms, test_transforms = utils.bring_data_transformation()

    source_dset = UDADataset(source_data, source_data, num_class, train_transforms, test_transforms, is_target=False, batch_size=args.batch_size)
    # src_train_dset, _, _ = src_dset.get_dsets()
    target_dset = UDADataset(
        split_paths['train'] if validation_selection else target_data,
        split_paths['test'] if validation_selection else target_eval_data,
        num_class, train_transforms, test_transforms, is_target=True, batch_size=args.batch_size,
        val_path=split_paths['validation'] if validation_selection else None)
    target_dset.get_dsets()
    if validation_selection and len(target_dset.train_dataset) < args.batch_size:
        raise ValueError('Target adaptation split must contain at least one complete training batch')

    dataset_info = utils.build_dataset_info(
        args.dataset,
        feature=args.feature,
        source_domain=None if args.dataset == 'visda' else args.source_domain,
        target_domain=None if args.dataset == 'visda' else args.target_domain,
        unks=getattr(args, 'unks', None),
        ablation_mode=getattr(args, 'ablation_mode', None),
    )
    save_info = utils.build_save_info(args.exp_code, args.model, args.net, dataset_info, args.seed)
    args.save_info = save_info

    result_model_dir = utils.get_save_dir(args, save_info)
    from pathlib import Path
    if Path(result_model_dir).exists() or Path(os.path.splitext(result_model_dir)[0]).exists():
        raise FileExistsError(f'Refusing to overwrite existing run: {result_model_dir}')
    args.result_log_path = result_model_dir
    args.result_artifact_dir = utils.get_result_artifact_dir(result_model_dir)
    import json
    audit_path = Path(args.result_artifact_dir) / 'data_audit.json'
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(data_audit, indent=2) + '\n')
    if split_audit is not None:
        import json
        from pathlib import Path
        audit_path = Path(args.result_artifact_dir) / 'target_split_audit.json'
        if audit_path.exists():
            raise ValueError('Use a fresh result directory for validation selection')
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps(split_audit, indent=2))
    if validation_selection:
        serializable_args = {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, list, type(None)))}
        (Path(args.result_artifact_dir) / 'run_args.json').write_text(json.dumps(serializable_args, indent=2))
    args.class_names = utils.get_dataset_class_names(args.dataset, num_class)

    logger = utils.bring_logger(result_model_dir)
    args.logger = logger
    args.logger.info('=' * 90)
    args.logger.info(sum_str)
    # args.logger.info('=' * 30)

    if args.model == 'CWODA':
        from models.cwoda import CWODA
        model = CWODA(args, num_class, source_dset, target_dset)

    model.train_init()
    if not validation_selection:
        model.test(0)
    model.build_model()
    model.train()

    # DataLoader workers can hang during interpreter shutdown after early-ended
    # iterators (zip loaders, cached fitting, smoke tests). All artifacts are now
    # written; explicitly reap this training process's remaining workers.
    import multiprocessing
    children = multiprocessing.active_children()
    for child in children:
        child.terminate()
    for child in children:
        child.join(timeout=5)
        if child.is_alive():
            child.kill()
            child.join(timeout=5)

if __name__ == '__main__':
    from config import args
    main(args)
