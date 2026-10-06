
from __future__ import print_function
import argparse
import os


default_num_workers = min(8, os.cpu_count() or 1)


parser = argparse.ArgumentParser(description='CWODA training for PRACH open-set domain adaptation')

# Data Level
parser.add_argument('--dataset', type=str, default='radio_nr_14devices',
                    choices=['radio_nr_14devices'],
                    help='seven source-known PRACH UEs plus target-unknown UEs')
parser.add_argument('--feature', type=str, default='stft+dscg',
                    choices=['dscg', 'stft', 'stft+dscg'],
                    help='input feature: dscg, stft, or stft+dscg')
parser.add_argument('--source_domain', type=str, default='D1',
                    help='source acquisition domain (for result naming)')
parser.add_argument('--target_domain', type=str, default='D2',
                    help='target acquisition domain (for result naming)')
parser.add_argument('--source_lists', nargs='+', default=None,
                    help='source training image list(s), one per input feature')
parser.add_argument('--target_train_lists', nargs='+', default=None,
                    help='unlabeled target adaptation list(s); labels must be -1')
parser.add_argument('--target_eval_lists', nargs='+', default=None,
                    help='labeled target evaluation list(s), matched to target_train_lists')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')

## Model Level
parser.add_argument('--model', type=str, default='CWODA', choices=['CWODA'],
                    help='model identifier used in output paths')
parser.add_argument(
    '--ablation_mode',
    type=str,
    default='full_osda',
    choices=['full_osda', 'da_wo_os', 'os_wo_da', 'source_only'],
    help=(
        'training mode: full_osda (default); da_wo_os uses binary source/target '
        'alignment without target open-set objectives; os_wo_da removes domain '
        'alignment; source_only disables target adaptation'
    ),
)
parser.add_argument('--net', type=str, default='resnet50', metavar='B',
                    help='resnet50, efficientnet, densenet, vgg')
parser.add_argument('--unknown_loss_weight_mode', choices=['weibull', 'uniform'], default='uniform')
parser.add_argument('--alignment_confidence_mode', choices=['adaptive', 'constant'], default='adaptive')
parser.add_argument('--confidence_estimator', choices=['weibull', 'msp'], default='msp')
parser.add_argument('--calibration_mode', choices=['source_class', 'legacy_clustered'], default='source_class')
parser.add_argument('--export_predictions', action='store_true')
parser.add_argument('--selection_protocol', choices=['legacy', 'validation_hos', 'development_hos'], default='legacy')
parser.add_argument('--target_split_manifest', default='', help='JSON with disjoint train/validation/test lists and recording-group files')
parser.add_argument('--unknown_loss_ramp_epochs', type=int, default=3, help='linear ramp of lamda3; 0 keeps it constant')
parser.add_argument('--max_batches', type=int, default=0, help='smoke test only: limit train/evaluation batches, 0 disables')
parser.add_argument('--bottle_neck_dim', type=int, default=256, metavar='B',
                    help='bottle_neck_dim for the classifier network.')
parser.add_argument('--bottle_neck_dim2', type=int, default=500, metavar='B',
                    help='bottle_neck_dim for the classifier network.')

## Iteration Level
parser.add_argument('--warmup_iter', type=int, default=2000, metavar='S',
                    help='warmup iteration for posterior inference')
parser.add_argument('--training_iter', type=int, default=10, metavar='S',
                    help='training_iter')
parser.add_argument('--update_term', type=int, default=10, metavar='S',
                    help='update term for posterior inference')
parser.add_argument('--eval_interval', type=int, default=1, metavar='S',
                    help='epoch interval for evaluation and metric logging')
parser.add_argument('--summary_windows', type=str, default='5,10', metavar='S',
                    help='comma separated evaluation windows used for averaged summaries')

## Loss Level
parser.add_argument('--threshold', type=float, default=0.85, metavar='fixmatch',
                    help='threshold for fixmatch')
parser.add_argument('--ls_eps', type=float, default=0.1, metavar='LR',
                    help='label smoothing for classification')

## Optimization Level
parser.add_argument('--update_freq_D', type=int, default=1, metavar='S',
                    help='freq for D in optimization.')
parser.add_argument('--update_freq_G', type=int, default=1, metavar='S',
                    help='freq for G in optimization.')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--num_workers', type=int, default=default_num_workers, metavar='N',
                    help='number of DataLoader workers')
parser.add_argument('--prefetch_factor', type=int, default=4, metavar='N',
                    help='number of batches prefetched by each DataLoader worker')
parser.add_argument('--no_persistent_workers', action='store_true', default=False,
                    help='disable persistent DataLoader workers')
parser.add_argument('--scheduler', type=str, default='cos',
                    help='learning rate scheduler')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='label smoothing for classification')
parser.add_argument('--e_lr', type=float, default=0.002, metavar='LR',
                    help='label smoothing for classification')
parser.add_argument('--g_lr', type=float, default=0.1, metavar='LR',
                    help='label smoothing for classification')

parser.add_argument('--opt_clip', type=float, default=0.1, metavar='LR',
                    help='label smoothing for classification')
parser.add_argument('--lamda1', type=float, default=0.5, metavar='LR',
                    help='label smoothing for classification')
parser.add_argument('--lamda2', type=float, default=0.6, metavar='LR',
                    help='label smoothing for classification')
parser.add_argument('--lamda3', type=float, default=0.3, metavar='LR',
                    help='label smoothing for classification')

parser.add_argument('--unks', type=str, default='unk1', metavar='LR',
                    help='legacy argument; result naming infers the unknown-UE count from lists')
parser.add_argument('--eval_unks', type=str, default='',
                    help='optional evaluation-only unknown split, e.g. unk1; empty uses the training split')
parser.add_argument('--feature_vis_interval', type=int, default=0, metavar='S',
                    help='epoch interval for feature visualization, 0 disables it')
parser.add_argument('--feature_vis_max_samples', type=int, default=2048, metavar='N',
                    help='max source or target samples used in each feature visualization')
parser.add_argument('--openmax_refresh_interval', type=int, default=1, metavar='N',
                    help='epoch interval for refreshing cached OpenMax parameters')
parser.add_argument('--openmax_max_samples', type=int, default=4096, metavar='N',
                    help='max target samples used to refresh OpenMax parameters, 0 uses all samples')

## etc:
parser.add_argument('--exp_code', type=str, default='Test', metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--result_dir', type=str, default='results', metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--set_gpu', type=str, default='0',
                    help='gpu setting, e.g. 0 or 0,1')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disable cuda')
parser.add_argument('--deterministic', action='store_true', default=False,
                    help='enable deterministic CUDA kernels for reproducibility')
parser.add_argument('--force_data_parallel', action='store_true', default=False,
                    help='wrap models with DataParallel even when only one GPU is visible')

args = parser.parse_args()
