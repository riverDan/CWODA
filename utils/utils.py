import csv
import json
import math
import torch
import torch.optim as opt
from models.basenet import *
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import sys
import os
from torch.utils.data import DataLoader
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
# import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
# import seaborn as sn
from collections import Counter
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"


## Some functions from https://github.com/ksaito-ut/OPDA_BP/blob/master/utils/utils.py

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)
def inverseDecayScheduler(step, initial_lr, gamma=10, power=0.75, max_iter=1000):
    return initial_lr * ((1 + gamma * min(1.0, step / float(max_iter))) ** (- power))
def StepwiseLRscheduler(step, initial_lr, gamma=10, decay_rate=0.75, max_iter=1000):
    return initial_lr * (1 + gamma * step) ** (-decay_rate)
    # return initial_lr * ((1 + gamma * min(1.0, step / float(max_iter))) ** (- power))
def ConstantScheduler(step, initial_lr, gamma=10, power=0.75, max_iter=1000):
    return initial_lr
def CosineScheduler(step, initial_lr, gamma=10, power=0.75, max_iter=1000):
    cos = (1 + np.cos((step / max_iter) * np.pi)) / 2
    lr = initial_lr * cos
    return lr
def StepScheduler(step, initial_lr, gamma=500, power=0.2, max_iter=1000):
    divide = step // 500
    lr = initial_lr * (0.2 ** divide)
    return lr
def setGPU(i):
    global os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "%s"%(i)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7,8"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    gpus = [x.strip() for x in (str(i)).split(',')]
    NGPU = len(gpus)
    return NGPU


def maybe_data_parallel(model, args):
    if not getattr(args, 'cuda', True) or not torch.cuda.is_available():
        return model

    model = model.cuda()
    use_data_parallel = getattr(args, 'force_data_parallel', False) or torch.cuda.device_count() > 1
    if use_data_parallel:
        model = torch.nn.DataParallel(model)
    return model


def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def get_model_init(args, known_num_class, all_num_class):
    net = args.net
    input_channels = get_input_channels(args)
    if net == 'vgg':
        model_g = VGGBase(input_channels=input_channels)
    elif 'resnet' in net:
        model_g = ResNetFc(model_name=args.net, input_channels=input_channels)
    elif 'effi' in net:
        model_g = EfficientNetB0(model_name='efficientnet', input_channels=input_channels)
    elif 'densenet' in net:
        model_g = DenseNet(model_name='densenet', input_channels=input_channels)
    else:
        print('Please specify the backbone network')
        sys.exit()

    # model_g = torch.nn.DataParallel(model_g).cuda()

    model_e = Net_CLS(in_dim=model_g.output_num(), out_dim=known_num_class, bias=False)
    model_c = Net_CLS_C(in_dim=model_g.output_num(), out_dim=all_num_class, bottle_neck_dim=args.bottle_neck_dim)

    model_g = maybe_data_parallel(model_g, args)
    model_e = maybe_data_parallel(model_e, args)
    model_c = maybe_data_parallel(model_c, args)

    return model_g, model_e, model_c

def get_model(args, known_num_class, all_num_class, domain_dim=3, dc_out_dim=None):
    net = args.net
    input_channels = get_input_channels(args)

    if net == 'vgg':
        model_g = VGGBase(input_channels=input_channels)
    elif 'resnet' in net:
        model_g = ResNetFc(model_name=args.net, input_channels=input_channels)
    elif 'effi' in net:
        model_g = EfficientNetB0(model_name='efficientnet', input_channels=input_channels)
    elif 'densenet' in net:
        model_g = DenseNet(model_name='densenet', input_channels=input_channels)
    else:
        print('Please specify the backbone network')
        sys.exit()

    if dc_out_dim is None:
        dc_dim = model_g.output_num()
    else:
        dc_dim = dc_out_dim

    model_e = Net_CLS(in_dim=model_g.output_num(), out_dim=known_num_class, bias=False)
    model_c = Net_CLS_C(in_dim=model_g.output_num(), out_dim=all_num_class, bottle_neck_dim=args.bottle_neck_dim)
    model_dc = Net_CLS_DC(dc_dim, out_dim=domain_dim, bottle_neck_dim=args.bottle_neck_dim2)

    model_g = maybe_data_parallel(model_g, args)
    model_e = maybe_data_parallel(model_e, args)
    model_c = maybe_data_parallel(model_c, args)
    model_dc = maybe_data_parallel(model_dc, args)

    return model_g, model_c, model_e, model_dc

class OptimWithSheduler:
    def __init__(self, optimizer, scheduler_func):
        self.optimizer = optimizer
        self.scheduler_func = scheduler_func
        self.global_step = 0.0
        for g in self.optimizer.param_groups:
            g['initial_lr'] = g['lr']
    def zero_grad(self):
        self.optimizer.zero_grad()
    def step(self):
        for g in self.optimizer.param_groups:
            g['lr'] = self.scheduler_func(step=self.global_step, initial_lr = g['initial_lr'])
        self.optimizer.step()
        self.global_step += 1

def bring_data_transformation():

    normalize_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize_transform
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize_transform
    ])

    return train_transforms, test_transforms


def extended_confusion_matrix(y_true, y_pred, true_labels=None, pred_labels=None):
    if not true_labels:
        true_labels = sorted(list(set(list(y_true))))
    true_label_to_id = {x: i for (i, x) in enumerate(true_labels)}
    if not pred_labels:
        pred_labels = true_labels
    pred_label_to_id = {x: i for (i, x) in enumerate(pred_labels)}
    confusion_matrix = np.zeros([len(true_labels), len(pred_labels)])
    for (true, pred) in zip(y_true, y_pred):
        confusion_matrix[true_label_to_id[true]][pred_label_to_id[pred]] += 1.0
    return confusion_matrix


def normalize_confusion_matrix(confusion_matrix):
    confusion_matrix = np.asarray(confusion_matrix, dtype=np.float32)
    if confusion_matrix.size == 0:
        return confusion_matrix
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return confusion_matrix / row_sums


def get_dataset_class_names(dataset, num_class):
    dataset = dataset.lower()
    dataset_class_names = {
        'office': ['back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle', 'calculator', 'desk_chair',
                   'desk_lamp', 'desktop_computer', 'file_cabinet', 'unk'],
        'officehome': ['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket', 'Calculator',
                       'Calendar', 'Candles', 'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk_Lamp',
                       'Drill', 'Eraser', 'Exit_Sign', 'Fan', 'File_Cabinet', 'Flipflops', 'Flowers', 'Folder',
                       'Fork', 'unk'],
        'visda': ['bicycle', 'bus', 'car', 'motorcycle', 'train', 'truck', 'unk'],
    }

    if dataset in {'radio_nr', 'radio_nr_new', 'radio_nr_14devices', 'radio_lte'}:
        if num_class <= 0:
            return []
        if num_class == 1:
            return ['unk']
        return [f'UE{i + 1}' for i in range(num_class - 1)] + ['unk']

    if dataset in dataset_class_names:
        return dataset_class_names[dataset][:num_class]

    return [f'class_{i}' for i in range(num_class)]


def get_result_artifact_dir(result_log_path):
    artifact_dir = os.path.splitext(result_log_path)[0]
    os.makedirs(artifact_dir, exist_ok=True)
    return artifact_dir


def build_dataset_info(dataset, feature=None, source_domain=None, target_domain=None,
                       unks=None, ablation_mode=None):
    parts = [str(dataset)]
    if feature:
        parts.append(str(feature))
    if source_domain:
        parts.append(str(source_domain))
    if target_domain:
        parts.append(str(target_domain))
    if unks:
        parts.append(str(unks))
    if ablation_mode:
        parts.append(str(ablation_mode))
    return '_'.join(parts)


def build_save_info(exp_code, model, net, dataset_info, seed):
    return f'{exp_code}#{model}#{net}#{dataset_info}#{seed}'


def _write_confusion_matrix_csv(csv_path, confusion_matrix, row_labels, col_labels):
    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['true/pred'] + list(col_labels))
        for row_label, row_values in zip(row_labels, confusion_matrix):
            writer.writerow([row_label] + [f'{value:.6f}' for value in row_values])


def render_confusion_matrix_text(confusion_matrix, row_labels, col_labels, precision=3):
    confusion_matrix = np.asarray(confusion_matrix)
    if confusion_matrix.size == 0:
        return 'empty confusion matrix'

    formatted_rows = [[f'{value:.{precision}f}' for value in row] for row in confusion_matrix]
    row_header_width = max([len('true/pred')] + [len(label) for label in row_labels])
    col_widths = []
    for col_idx, col_label in enumerate(col_labels):
        col_widths.append(max([len(col_label)] + [len(row[col_idx]) for row in formatted_rows]))

    lines = []
    header_cells = ['true/pred'.ljust(row_header_width)]
    header_cells.extend(label.ljust(width) for label, width in zip(col_labels, col_widths))
    lines.append(' '.join(header_cells))

    for row_label, row_values in zip(row_labels, formatted_rows):
        row_cells = [row_label.ljust(row_header_width)]
        row_cells.extend(value.ljust(width) for value, width in zip(row_values, col_widths))
        lines.append(' '.join(row_cells))

    return '\n'.join(lines)


def _get_song_font_name():
    candidates = [
        'SimSun',
        'Songti SC',
        'STSong',
        'Noto Serif CJK SC',
        'Source Han Serif SC',
        'AR PL UMing CN',
        'Noto Sans CJK SC',
    ]
    installed_fonts = {font.name for font in fm.fontManager.ttflist}
    for font_name in candidates:
        if font_name in installed_fonts:
            return font_name
    return None


def save_confusion_matrix_artifacts(save_dir, file_stem, confusion_matrix, row_labels, col_labels, title, normalize=True):
    raw_confusion = np.asarray(confusion_matrix, dtype=np.float32)
    display_confusion = normalize_confusion_matrix(raw_confusion) if normalize else raw_confusion
    song_font_name = _get_song_font_name()
    font_kwargs = {'fontname': song_font_name} if song_font_name else {}

    os.makedirs(save_dir, exist_ok=True)
    figure_path = os.path.join(save_dir, file_stem + '.png')
    counts_csv_path = os.path.join(save_dir, file_stem + '_counts.csv')
    normalized_csv_path = os.path.join(save_dir, file_stem + '_normalized.csv')

    _write_confusion_matrix_csv(counts_csv_path, raw_confusion, row_labels, col_labels)
    _write_confusion_matrix_csv(normalized_csv_path, display_confusion, row_labels, col_labels)

    fig_width = max(6.0, 1.1 * max(1, len(col_labels)) + 1.5)
    fig_height = max(3.5, 0.8 * max(1, len(row_labels)) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(display_confusion, cmap='Blues', aspect='auto', vmin=0.0 if normalize else None,
                   vmax=1.0 if normalize else None)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha='right', **font_kwargs)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, **font_kwargs)
    ax.set_xlabel('Predict Label', **font_kwargs)
    ax.set_ylabel('True Label', **font_kwargs)

    threshold = 0.5 if normalize else (display_confusion.max() / 2.0 if display_confusion.size else 0.0)
    for row_idx in range(display_confusion.shape[0]):
        for col_idx in range(display_confusion.shape[1]):
            if normalize:
                text_value = f'{display_confusion[row_idx, col_idx]:.2f}'
            else:
                text_value = f'{int(raw_confusion[row_idx, col_idx])}'
            color = 'white' if display_confusion[row_idx, col_idx] > threshold else 'black'
            ax.text(col_idx, row_idx, text_value, ha='center', va='center', color=color, fontsize=9, **font_kwargs)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return {
        'raw': raw_confusion,
        'normalized': display_confusion,
        'figure_path': figure_path,
        'counts_csv_path': counts_csv_path,
        'normalized_csv_path': normalized_csv_path,
        'row_labels': list(row_labels),
        'col_labels': list(col_labels),
    }


def get_target_confusion_row_labels(class_names, known_num_class, max_target_label):
    row_labels = []
    for label_idx in range(max_target_label):
        if label_idx < len(class_names):
            row_labels.append(class_names[label_idx])
        elif label_idx == known_num_class:
            row_labels.append('unk')
        else:
            row_labels.append(f'unk{label_idx - known_num_class + 1}')
    return row_labels


def export_target_confusion_matrices(args, epoch, total_label_t, total_pred_t, known_num_class, all_num_class):
    result_artifact_dir = getattr(args, 'result_artifact_dir', None)
    if result_artifact_dir is None:
        return {}

    total_label_t = np.asarray(total_label_t, dtype=np.int64)
    total_pred_t = np.asarray(total_pred_t, dtype=np.int64)
    class_names = list(getattr(args, 'class_names', get_dataset_class_names(args.dataset, all_num_class)))
    if len(class_names) < all_num_class:
        class_names.extend([f'class_{i}' for i in range(len(class_names), all_num_class)])

    all_class_names = class_names[:all_num_class]
    target_domain = getattr(args, 'target_domain', 'target')
    epoch_tag = f'epoch_{int(epoch):03d}'

    max_target_label = int(np.max(total_label_t) + 1)
    target_confusion = extended_confusion_matrix(
        total_label_t,
        total_pred_t,
        true_labels=list(range(max_target_label)),
        pred_labels=list(range(all_num_class))
    )
    target_row_labels = get_target_confusion_row_labels(all_class_names, known_num_class, max_target_label)

    target_artifacts = save_confusion_matrix_artifacts(
        save_dir=result_artifact_dir,
        file_stem=f'{epoch_tag}_{target_domain}_target_confusion',
        confusion_matrix=target_confusion,
        row_labels=target_row_labels,
        col_labels=all_class_names,
        title=f'Target {target_domain} Full Confusion',
        normalize=True,
    )

    return {
        'target': target_artifacts,
    }


def parse_summary_windows(summary_windows):
    if isinstance(summary_windows, (list, tuple)):
        raw_windows = [int(window) for window in summary_windows]
    else:
        raw_windows = []
        for token in str(summary_windows).split(','):
            token = token.strip()
            if not token:
                continue
            raw_windows.append(int(token))

    windows = []
    for window in raw_windows:
        if window <= 0:
            continue
        if window not in windows:
            windows.append(window)

    if not windows:
        raise ValueError('summary_windows must contain at least one positive integer.')

    return windows


def _metric_to_float(value):
    if value is None:
        return None

    if isinstance(value, torch.Tensor):
        value = value.item() if value.numel() == 1 else value.mean().item()
    elif isinstance(value, np.generic):
        value = value.item()

    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def build_metric_record(epoch, os_value, os_star, unk, hos, auroc=None, aupr=None, oscr=None):
    return {
        'epoch': int(epoch),
        'OS': _metric_to_float(os_value),
        'OS_star': _metric_to_float(os_star),
        'UNK': _metric_to_float(unk),
        'HOS': _metric_to_float(hos),
        'AUROC': _metric_to_float(auroc),
        'AUPR': _metric_to_float(aupr),
        'OSCR': _metric_to_float(oscr),
    }


def _average_metric_records(metric_records):
    averaged_metrics = {}
    for metric_name in ['OS', 'OS_star', 'UNK', 'HOS', 'AUROC', 'AUPR', 'OSCR']:
        values = [record[metric_name] for record in metric_records if record.get(metric_name) is not None]
        averaged_metrics[metric_name] = float(np.mean(values)) if values else None
    return averaged_metrics


def summarize_metric_history(metric_history, windows):
    if not metric_history:
        return {}

    history = sorted(metric_history, key=lambda record: int(record['epoch']))
    summary = {
        'history_length': len(history),
        'evaluated_epochs': [int(record['epoch']) for record in history],
        'final': dict(history[-1], count=1),
    }

    best_record = max(
        history,
        key=lambda record: (
            float('-inf') if record.get('HOS') is None else record['HOS'],
            float('-inf') if record.get('OS_star') is None else record['OS_star'],
            int(record['epoch']),
        ),
    )
    summary['best_hos'] = dict(best_record, count=1)

    for window in windows:
        window_size = min(int(window), len(history))
        window_records = history[-window_size:]
        averaged_metrics = _average_metric_records(window_records)
        averaged_metrics.update({
            'count': len(window_records),
            'requested_window': int(window),
            'start_epoch': int(window_records[0]['epoch']),
            'end_epoch': int(window_records[-1]['epoch']),
        })
        summary[f'last_{int(window)}_avg'] = averaged_metrics

    return summary


def save_metric_history_csv(result_artifact_dir, metric_history):
    if result_artifact_dir is None or not metric_history:
        return None

    os.makedirs(result_artifact_dir, exist_ok=True)
    csv_path = os.path.join(result_artifact_dir, 'epoch_metrics.csv')
    fieldnames = ['epoch', 'OS', 'OS_star', 'UNK', 'HOS', 'AUROC', 'AUPR', 'OSCR']
    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted(metric_history, key=lambda item: int(item['epoch'])):
            row = {'epoch': int(record['epoch'])}
            for fieldname in fieldnames[1:]:
                value = record.get(fieldname)
                row[fieldname] = '' if value is None else f'{value:.6f}'
            writer.writerow(row)
    return csv_path


def _build_metric_summary_metadata(args):
    return {
        'exp_code': getattr(args, 'exp_code', None),
        'model': getattr(args, 'model', None),
        'net': getattr(args, 'net', None),
        'dataset': getattr(args, 'dataset', None),
        'feature': getattr(args, 'feature', None),
        'source_domain': getattr(args, 'source_domain', None),
        'target_domain': getattr(args, 'target_domain', None),
        'unks': getattr(args, 'unks', None),
        'eval_unks': getattr(args, 'eval_unks', None),
        'ablation_mode': getattr(args, 'ablation_mode', None),
        'calibration_mode': getattr(args, 'calibration_mode', 'legacy_clustered'),
        'confidence_estimator': getattr(args, 'confidence_estimator', 'weibull'),
        'unknown_loss_weight_mode': getattr(args, 'unknown_loss_weight_mode', 'weibull'),
        'alignment_confidence_mode': getattr(args, 'alignment_confidence_mode', 'adaptive'),
        'seed': getattr(args, 'seed', None),
        'training_iter': getattr(args, 'training_iter', None),
        'eval_interval': getattr(args, 'eval_interval', None),
        'summary_windows': list(getattr(args, 'summary_windows', [])),
        'save_info': getattr(args, 'save_info', None),
        'result_log_path': getattr(args, 'result_log_path', None),
        'result_artifact_dir': getattr(args, 'result_artifact_dir', None),
    }


def save_metric_history_summary(args, metric_history):
    result_artifact_dir = getattr(args, 'result_artifact_dir', None)
    if result_artifact_dir is None or not metric_history:
        return {}

    history_csv_path = save_metric_history_csv(result_artifact_dir, metric_history)
    summary = summarize_metric_history(metric_history, getattr(args, 'summary_windows', [5, 10]))
    summary_payload = {
        'metadata': _build_metric_summary_metadata(args),
        'history_csv_path': history_csv_path,
    }
    summary_payload.update(summary)

    summary_path = os.path.join(result_artifact_dir, 'summary.json')
    with open(summary_path, 'w') as summary_file:
        json.dump(summary_payload, summary_file, indent=2, sort_keys=True)

    return {
        'history_csv_path': history_csv_path,
        'summary_path': summary_path,
        'summary': summary,
    }


def _format_metric_value(value):
    return 'NA' if value is None else f'{value:.3f}'


def render_metric_summary_lines(summary):
    if not summary:
        return []

    ordered_keys = ['final', 'best_hos']
    ordered_keys.extend(sorted(key for key in summary.keys() if key.startswith('last_')))
    lines = []
    for key in ordered_keys:
        if key not in summary:
            continue
        block = summary[key]
        if key.startswith('last_'):
            prefix = '{}_epochs_{:03d}-{:03d}_count_{}'.format(
                key.upper(),
                int(block['start_epoch']),
                int(block['end_epoch']),
                int(block['count']),
            )
        else:
            prefix = '{}_epoch_{:03d}'.format(key.upper(), int(block['epoch']))

        line = '{}_OS_{}_OS*_{}_UNK_{}_HOS_{}'.format(
            prefix,
            _format_metric_value(block.get('OS')),
            _format_metric_value(block.get('OS_star')),
            _format_metric_value(block.get('UNK')),
            _format_metric_value(block.get('HOS')),
        )
        if block.get('AUROC') is not None:
            line += '_AUROC_{}'.format(_format_metric_value(block.get('AUROC')))
        if block.get('AUPR') is not None:
            line += '_AUPR_{}'.format(_format_metric_value(block.get('AUPR')))
        if block.get('OSCR') is not None:
            line += '_OSCR_{}'.format(_format_metric_value(block.get('OSCR')))
        lines.append(line)

    return lines


def bce_loss(output, target):
    output_neg = 1 - output
    target_neg = 1 - target
    result = torch.mean(target * torch.log(output + 1e-6))
    result += torch.mean(target_neg * torch.log(output_neg + 1e-6))
    return -torch.mean(result)


def get_save_dir(args, save_info):
    result_dir = args.result_dir
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    result_folder_dir = os.path.join(result_dir, args.exp_code)
    if not os.path.exists(result_folder_dir):
        os.mkdir(result_folder_dir)
    result_saveinfo_dir = os.path.join(result_folder_dir, '%s.txt'%save_info)
    return result_saveinfo_dir


def bring_logger(results_log_dir, level='info'):
    import logging
    log1 = logging.getLogger('model specific logger')
    streamH = logging.StreamHandler()
    log1.addHandler(streamH)
    fileH = logging.FileHandler(results_log_dir)
    log1.addHandler(fileH)
    if level =='debug':
        log1.setLevel(level=logging.DEBUG)
    else:
        log1.setLevel(level=logging.INFO)
    return log1

from PIL import Image
def default_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def IQ_loader(path):
    seq = np.fromfile(path, dtype=np.float32).reshape(2, 30720)

    complex_seq = seq[0, :] + 1j * seq[1, :]
    fft_result = np.fft.fft(complex_seq, axis=0)
    fft_seq = np.stack((fft_result.real, fft_result.imag), axis=0).astype(np.float32)
    return fft_seq

    # # 重塑为 [2, 256, 120]
    # reshaped_fft = fft_seq.reshape(2, 256, 120)

    # # 创建目标数组并填充数据
    # target_array = np.zeros((2, 256, 256), dtype=np.float32)
    # target_array[:, :, :120] = reshaped_fft

    # return target_array


import os.path as osp
def plot_features(feature_dir, features, labels, num_classes):
    """Plot features on 2D plane.

    Args:
        features: (num_instances, num_features).
        labels: (num_instances).
    """
    # colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    for label_idx in range(num_classes):
        plt.scatter(
            features[labels==label_idx, 0],
            features[labels==label_idx, 1],
            c=colors[label_idx],
            s=1,
        )
    plt.legend(['UE1', 'UE2', 'UE3', 'UE4', 'UE5', 'UE6', 'UE7', 'UE8'], loc='upper right')
    # dirname = osp.join(feature_dir, prefix)
    # if not osp.exists(dirname):
    #     os.mkdir(dirname)
    # save_name = osp.join(dirname, 'epoch_' + str(epoch+1) + '.png')
    save_name = osp.join(feature_dir + '.png')
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()
