# RFFI_CWODA

CWODA (confidence-weighted open-set domain alignment) for specific emitter
identification using 5G PRACH signals. The training code uses paired short-time
Fourier transform (STFT) and differential subcarrier graph (DSCG) images. It
trains on labeled source UEs and adapts to an unlabeled target domain that may
contain previously unseen UEs.

## Method and protocol

The closed-set branch is trained on the seven source-known UEs and then frozen.
Its maximum softmax probability (MSP) supplies complementary known and unknown
weights for a three-output domain discriminator. A separate open-set head is
trained with target consistency and unknown-rejection objectives and provides
the final predictions. The model and result paths are named `CWODA`; inherited
upstream code is credited in [THIRD_PARTY.md](THIRD_PARTY.md).

The default 5G experiment uses seven known UEs (`UE1`–`UE7`) and one or more
target-unknown UEs (starting at `UE8`). Known class labels are `0`–`6`; unknown
UEs receive distinct evaluation labels `7`, `8`, ... and are merged into one
unknown prediction class by the model. The target adaptation list has `-1` in
every label column. The target evaluation list references the same images but
contains labels used only to report metrics. This is **transductive evaluation**:
the target images are seen without labels during adaptation and then evaluated
with labels.

## Installation

Python 3.12 and PyTorch 2.2.2 / PyTorchVision 0.17.2 were used for local checks.
Install a PyTorch build suitable for your CPU or CUDA version, then install the
remaining packages:

```bash
python -m venv .venv
source .venv/bin/activate
# Install the matching torch and torchvision wheels for your platform first.
python -m pip install -r requirements.txt
```

The ResNet-50 backbone requests ImageNet weights through PyTorchVision on the
first run; download access or a populated PyTorch cache is required. No weights
are bundled here.

## Dataset download

The 5G PRACH dataset is shared as [RFFI_NR.zip](https://pan.baidu.com/s/1J2fqbXWZ-wjw525s5EPCCw)
on Baidu Netdisk. Access code (提取码): `aqf1`.

After downloading and extracting the archive, set `--data-root` to the directory
containing the D1, D2, D3, P1, P2, and P3 domain folders. The paired STFT and
DSCG images must follow the layout described below.

## Input layout and list generation

Supply paired PNG files with matching basenames in both feature folders:

```text
DATA_ROOT/
  D1/
    STFT/UE1/0.png
    DSCG/UE1/0.png
    ...
  D2/
    STFT/UE1/0.png
    DSCG/UE1/0.png
    ...
```

The generator checks that each UE has matching STFT/DSCG sample IDs. Example
for `D1` to `D2`, with `UE8` unknown:

```bash
python prepare_lists.py \
  --data-root /path/to/RFFI_NR \
  --source-domain D1 --target-domain D2 \
  --known-ues 1 2 3 4 5 6 7 --unknown-ues 8 \
  --output-dir lists/D1_D2_UE8
```

The command writes source, unlabeled target-adaptation, and labeled
target-evaluation lists for each feature, plus a manifest with counts and
SHA-256 hashes. Lists use absolute local paths and are ignored by Git. To
reproduce an openness level, pass all included unknown UEs to `--unknown-ues`.
The `--max-samples-per-ue` option is intended only for quick checks.

## Train one directed transfer

Run from the repository root. The training flags below reproduce the default
CWODA configuration in this release; the essential hyperparameters are also
the defaults in `config.py`.

```bash
python train.py \
  --feature stft+dscg \
  --source_domain D1 --target_domain D2 \
  --source_lists lists/D1_D2_UE8/source_stft.txt lists/D1_D2_UE8/source_dscg.txt \
  --target_train_lists lists/D1_D2_UE8/target_train_stft.txt lists/D1_D2_UE8/target_train_dscg.txt \
  --target_eval_lists lists/D1_D2_UE8/target_eval_stft.txt lists/D1_D2_UE8/target_eval_dscg.txt \
  --warmup_iter 2000 --training_iter 10 --batch_size 32 \
  --lamda1 0.5 --lamda2 0.6 --lamda3 0.3 \
  --unknown_loss_ramp_epochs 3 --confidence_estimator msp \
  --unknown_loss_weight_mode uniform --threshold 0.85 \
  --seed 0 --set_gpu 0 --exp_code D1_D2_UE8_s0 \
  --result_dir results --export_predictions
```

Use `--no_cuda` for CPU execution. For `--feature stft` or `--feature dscg`,
provide just one list per split. Each run writes its metrics, logs, prediction
files (when requested), and `data_audit.json` under `results/`; these outputs
are excluded from Git. The main reported metrics include overall accuracy,
known and unknown accuracy, HOS, AUROC, and OSCR. `--max_batches` is for smoke
checks only and must not be used for reported experiments.

The public-release protocol specifies the following 12 directed transfers:

```text
D1→D2  D1→D3  D2→D1  D2→D3  D3→D1  D3→D2
P1→P2  P1→P3  P2→P1  P2→P3  P3→P1  P3→P2
```

Generate a new list directory and result code for each transfer and random
seed. Full experimental results are not shipped with this source release.

## Provenance

The project builds on open-set domain adaptation research and retains parts
of an earlier UADAL-based codebase. See [THIRD_PARTY.md](THIRD_PARTY.md) and
the attribution comments inside source files. If you use this project for a
paper, cite the CWODA paper and the relevant upstream methods. Repository
publication should follow a separate review of source-code rights and data
distribution rights.
