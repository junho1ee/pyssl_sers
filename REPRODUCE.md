# Reproducing the reported results

Every number in the manuscript comes from one of the runs below. The batch
scripts that submitted them were written for a specific SLURM site and are not
distributed; the commands they wrapped are given here in full. Each run writes a
`hparams.yaml` next to its outputs that records the complete configuration
actually used, so a stored run can always be checked against this file.

Runs are identified by their `--run_tag`, which is also the directory name under
`results/bacteria-id/finetuning/`.

Sections 1–4 are the runs of the submitted manuscript; section 5 adds the runs
of the first revision (tag `rev1`). Table numbers refer to the revised
manuscript, in which the component ablation is Table 5 and the COVID-19 results
are Table 6 (Table 5 of the submission).

## 0. Environment and data

    python >= 3.10, torch >= 2.0, lightning, omegaconf, scipy, scikit-learn, matplotlib

Download the two datasets as described in `README.md`, then

    python preprocess/data_preprocess_bacteria.py
    python preprocess/data_preprocess_covid.py

`--data_variant full` applies asymmetric least-squares baseline correction and
Savitzky--Golay smoothing before normalization and interpolation;
`--data_variant minimal` omits both.

## 1. Pretraining

Supervised reference pretraining on the labeled Bacteria-ID reference subset.
This produces `version_ho_adam_es10_aug`, the checkpoint every supervised
downstream row starts from:

    python lightning_pretrain_supervised.py --pre supervised --augtype phys \
        --data_variant full --optimizer adam --lr 1e-3 --weight_decay 0 \
        --batch_size 512 --n_epochs 200 --patience 10 --valid_size 0.1 \
        --use_augmentation

Self-supervised pretraining at the shared batch size of 1024, six encoders
(three objectives x two view sets). This produces `version_bs1024`:

    for PRE in simclrv2 mocov3 byol; do
      for AUG in phys crop; do
        python lightning_pretrain_ssl.py --pre "$PRE" --augtype "$AUG" \
            --batch_size 1024 --n_epochs 3000
      done
    done

BYOL additionally pretrained at batch size 2048. This produces `version_1`:

    for AUG in phys crop; do
      python lightning_pretrain_ssl.py --pre byol --augtype "$AUG" \
          --batch_size 2048 --n_epochs 3000
    done

No early stopping is applied to any self-supervised run; the length of 3000
epochs is fixed in advance. Section 3 below measures what that costs.

## 2. Downstream runs

### Table 3 and Figure 4 — five random hold-out splits, downstream augmentation on

    COMMON="--augtype phys --data_variant full --pretrained_ckpt_name last.ckpt \
            --use_pretrained --split_mode random_holdout --n_splits 5 --valid_size 0.1 \
            --optimizer adam --adam_beta1 0.5 --adam_beta2 0.999 --lr 1e-3 \
            --weight_decay 0 --batch_size 64 --n_epochs 200 --patience 50 \
            --use_augmentation"

Table 3, self-supervised rows. SimCLR v2 and MoCo v3 use `version_bs1024`, BYOL
uses `version_1`, because Table 3 reports BYOL at batch size 2048 and the two
contrastive objectives at 1024:

    for TASK in class30 class2; do for FOLD in 0 1 2 3 4; do
      python lightning_finetune_pred.py --pre simclrv2 --task "$TASK" --fold "$FOLD" \
          --pretrained_version version_bs1024 $COMMON --run_tag ho_split_adam_downstream_aug
      python lightning_finetune_pred.py --pre mocov3 --task "$TASK" --fold "$FOLD" \
          --pretrained_version version_bs1024 $COMMON --run_tag ho_split_adam_downstream_aug
      python lightning_finetune_pred.py --pre byol --task "$TASK" --fold "$FOLD" \
          --pretrained_version version_1 $COMMON --run_tag ho_split_adam_downstream_aug
    done; done

Table 3, supervised rows. `--reuse_pretrained_classifier` carries the reference
classifier over; the "w/o aug." row repeats this with the encoder pretrained
without augmentation and `--no-use_augmentation` downstream:

    for TASK in class30 class2; do for FOLD in 0 1 2 3 4; do
      python lightning_finetune_pred.py --pre supervised --task "$TASK" --fold "$FOLD" \
          --pretrained_version version_ho_adam_es10_aug --reuse_pretrained_classifier \
          $COMMON --run_tag ho_pre_adam_es10_last_ft_ho_split_adam
    done; done

Figure 4 adds `--n_labels_per_class` to the same protocol:

    for NLAB in 10 20 50 100; do for FOLD in 0 1 2 3 4; do
      python lightning_finetune_pred.py --pre simclrv2 --task class30 --fold "$FOLD" \
          --pretrained_version version_bs1024 $COMMON \
          --n_labels_per_class "$NLAB" --run_tag "labelsweep_n${NLAB}"
      # ... likewise mocov3 (version_bs1024), byol (version_1),
      #     supervised (version_ho_adam_es10_aug --reuse_pretrained_classifier),
      #     and no_pre (--pre no_pre --no-use_pretrained)
    done; done

### Table 4 — ten stratified folds, downstream augmentation off

The pretraining-augmentation column of Table 4 therefore isolates the views.

    for PRE in simclrv2 mocov3 byol; do for AUG in phys crop; do
      for FOLD in $(seq 0 9); do
        python lightning_finetune_pred.py --pre "$PRE" --task class30 --augtype "$AUG" \
            --fold "$FOLD" --data_variant full --pretrained_version version_bs1024 \
            --pretrained_ckpt_name last.ckpt --use_pretrained --no-use_augmentation \
            --run_tag matched_bs1024
      done
    done; done

The two rows without pretraining, one per preprocessing pipeline:

    for VARIANT in full minimal; do for FOLD in $(seq 0 9); do
      python lightning_finetune_pred.py --pre no_pre --task class30 --augtype phys \
          --fold "$FOLD" --data_variant "$VARIANT" --no-use_pretrained --no-use_augmentation
    done; done

The two supervised rows, differing only in whether the reference classifier is
carried over:

    for FOLD in $(seq 0 9); do
      SUP="--pre supervised --task class30 --augtype phys --fold $FOLD --data_variant full \
           --pretrained_version version_ho_adam_es10_aug --pretrained_ckpt_name last.ckpt \
           --use_pretrained --no-use_augmentation"
      python lightning_finetune_pred.py $SUP --reuse_pretrained_classifier    --run_tag matched_table3
      python lightning_finetune_pred.py $SUP --no-reuse_pretrained_classifier --run_tag matched_table3_freshhead
    done

### Submitted Table 5 — COVID-19 transfer, 50 repeats under both partitioning protocols

`covid_transfer_eval.py` produced Table 5 of the submission and is kept
unchanged. It passes each spectrum to the augmentation as a (696, 1) array,
under which the baseline shift and wander have almost no effect, so its
"w aug." ResNet rows do not use the stated augmentation. Table 6 of the revision
comes from `covid_rev1_eval.py` (section 5.5), which uses the (1, 696) layout of
the Bacteria-ID pipeline and the same partitions.

    for TASK in covid_vs_suspected covid_vs_healthy suspected_vs_healthy; do
      for PRE in svm no_pre_noaug no_pre_aug supervised byol mocov3 simclrv2; do
        python covid_transfer_eval.py --task "$TASK" --pre "$PRE" --repeats 50 \
            --protocols spectrum subject --out results/covid
      done
    done

The two protocols share the same 50 seeds, so each model sees the same 50
partitions.

## 3. Controls behind the BYOL discussion

Neither control changes anything about the downstream protocol.

BYOL pretrained at batch size 2048, fine-tuned on the same ten folds as Table 4:

    for FOLD in $(seq 0 9); do
      python lightning_finetune_pred.py --pre byol --task class30 --augtype phys \
          --fold "$FOLD" --data_variant full --pretrained_version version_1 \
          --pretrained_ckpt_name last.ckpt --use_pretrained --no-use_augmentation \
          --run_tag ctrl_bs2048_table3proto
    done

Downstream accuracy of the batch size 1024 BYOL encoder as a function of
pretraining epoch, five common folds under one fixed fine-tuning protocol. This
probe read the test subset and was used only in the submitted manuscript; the
revision replaces it with the validation-based probe of section 5.2:

    for EP in 499 999 1499 1999 2499 2999; do for FOLD in 0 1 2 3 4; do
      python lightning_finetune_pred.py --pre byol --task class30 --augtype phys \
          --fold "$FOLD" --data_variant full --pretrained_version version_bs1024 \
          --pretrained_ckpt_name "$(basename "$(ls results/bacteria-id/pretraining/phys/byol/lightning_logs/version_bs1024/checkpoints/epoch=${EP}-loss=*.ckpt | head -1)")" \
          --use_pretrained \
          --split_mode random_holdout --n_splits 5 --valid_size 0.1 \
          --optimizer adam --adam_beta1 0.5 --adam_beta2 0.999 --lr 1e-3 --weight_decay 0 \
          --batch_size 64 --n_epochs 200 --patience 50 --use_augmentation \
          --run_tag "epochprobe_e${EP}"
    done; done

## 4. Aggregation and figures

    python scripts/recompute_paper_tables.py     # Tables 3 and 4, and the per-fold values
                                                 # behind the paired differences
    python scripts/generate_bacteria_figures.py  # Figures 2 and 3
    python scripts/recompute_fig3_roc.py         # Figure 3B ROC curve and its area
    python scripts/plot_label_efficiency.py      # Figure 4
    python scripts/plot_objective_schematic.py   # Figure 1

`recompute_paper_tables.py` scores every run at the sample level, pooling all
test spectra per fold. The `test_acc` written into `test_results.json` is a
batch average and differs from the sample-level value by about 0.05 percentage
points whenever the last batch is short; the manuscript reports the sample-level
value.

## 5. Runs added in the first revision

Every revision run writes to a new version directory (pretraining, set through
the `LOGGER_VERSION` environment variable) or under a new run tag (fine-tuning)
starting with `rev1_`, so none of them overwrites a run of sections 1–4. The
original runs are seed 0. `CKPT` below stands for the checkpoint of a given
epoch in the version directory concerned, found as in section 3:

    ckpt() { basename "$(ls results/bacteria-id/pretraining/$1/lightning_logs/$2/checkpoints/epoch=$3-loss=*.ckpt | head -1)"; }

Fine-tuning always uses the Table 4 protocol (ten stratified folds, downstream
augmentation off):

    T4="--task class30 --data_variant full --use_pretrained --no-use_augmentation"

### 5.1 Table 4 — two further pretraining runs of every encoder

    for SEED in 1 2; do
      for CFG in "phys simclrv2 1024" "phys mocov3 1024" "phys byol 1024" "phys byol 2048" \
                 "crop simclrv2 1024" "crop mocov3 1024" "crop byol 1024" "crop byol 2048"; do
        set -- $CFG
        LOGGER_VERSION="rev1_bs$3_seed${SEED}" python lightning_pretrain_ssl.py \
            --pre "$2" --augtype "$1" --batch_size "$3" --n_epochs 3000 --seed "$SEED"
      done
    done

### 5.2 Table 4 — validation-chosen checkpoints

Every encoder is fine-tuned from its checkpoints at 500, ..., 3000 epochs, and
the epoch with the highest mean validation accuracy is chosen per encoder.
Seed 0 (the original runs; `version_1` is BYOL at batch size 2048):

    for AUG in phys crop; do
      for PV in "simclrv2 version_bs1024" "mocov3 version_bs1024" "byol version_bs1024" "byol version_1"; do
        set -- $PV
        for EP in 499 999 1499 1999 2499 2999; do for FOLD in $(seq 0 9); do
          python lightning_finetune_pred.py --pre "$1" --augtype "$AUG" --fold "$FOLD" $T4 \
              --pretrained_version "$2" --pretrained_ckpt_name "$(ckpt $AUG/$1 $2 $EP)" \
              --run_tag "rev1_s0_e${EP}"
        done; done
      done
    done

Seeds 1 and 2: the same loop over the versions `rev1_bs<batch>_seed<seed>` of
section 5.1, with `--run_tag "rev1_s1_e${EP}"`.

### 5.3 Table 5 — components of the physically motivated view

SimCLR v2, batch size 1024, seed 0, 1000 epochs. The augtypes are defined in
`utils.AUGTYPES`; the reference is the epoch-1000 checkpoint of the original
`phys` run (`rev1_s0_e999` in section 5.2).

    for AUG in phys_no_powerline phys_no_emg phys_no_shift phys_no_wander \
               phys_only_powerline phys_only_emg phys_only_shift phys_only_wander \
               phys_shift_x0p1 phys_all_x0p5 phys_noise_x2 phys_slow_x0p1; do
      LOGGER_VERSION=rev1_bs1024_seed0_e1000 python lightning_pretrain_ssl.py \
          --pre simclrv2 --augtype "$AUG" --batch_size 1024 --n_epochs 1000 --seed 0
      for EP in 499 999; do for FOLD in $(seq 0 9); do
        python lightning_finetune_pred.py --pre simclrv2 --augtype "$AUG" --fold "$FOLD" $T4 \
            --pretrained_version rev1_bs1024_seed0_e1000 \
            --pretrained_ckpt_name "$(ckpt $AUG/simclrv2 rev1_bs1024_seed0_e1000 $EP)" \
            --run_tag "rev1_s2_e${EP}"
      done; done
    done

### 5.4 Section 3.2 — the four extrapolated grid points

`scripts/rev1_make_edgehold.py` writes the Bacteria-ID subsets with the four
points above 1792.4 cm-1 held at the last measured value
(`data_rev1/bacteria-id/preprocessed_edgehold`), used by `--data_variant edgehold`.

    python scripts/rev1_make_edgehold.py

Random initialization on original and edge-held inputs, and the epoch-3000
SimCLR v2 encoder on edge-held inputs:

    for FOLD in $(seq 0 9); do
      for V in full edgehold; do
        python lightning_finetune_pred.py --pre no_pre --task class30 --augtype phys \
            --fold "$FOLD" --data_variant "$V" --no-use_augmentation --run_tag rev1_s3
      done
      python lightning_finetune_pred.py --pre simclrv2 --task class30 --augtype phys \
          --fold "$FOLD" --data_variant edgehold --use_pretrained --no-use_augmentation \
          --pretrained_version version_bs1024 \
          --pretrained_ckpt_name "$(ckpt phys/simclrv2 version_bs1024 2999)" --run_tag rev1_s3
    done

SimCLR v2 pretrained on edge-held reference spectra (seed 0, 1000 epochs) for
both view sets, then the 2 x 2 of pretraining inputs and fine-tuning inputs at
epoch 1000 (the original/original cell is `rev1_s0_e999`):

    for AUG in phys crop; do
      LOGGER_VERSION=rev1_bs1024_seed0_e1000_edgehold python lightning_pretrain_ssl.py \
          --pre simclrv2 --augtype "$AUG" --batch_size 1024 --n_epochs 1000 --seed 0 \
          --data_variant edgehold
      for CELL in "rev1_bs1024_seed0_e1000_edgehold edgehold" "rev1_bs1024_seed0_e1000_edgehold full" \
                  "version_bs1024 edgehold"; do
        set -- $CELL
        for FOLD in $(seq 0 9); do
          python lightning_finetune_pred.py --pre simclrv2 --task class30 --augtype "$AUG" \
              --fold "$FOLD" --data_variant "$2" --use_pretrained --no-use_augmentation \
              --pretrained_version "$1" --pretrained_ckpt_name "$(ckpt $AUG/simclrv2 $1 999)" \
              --run_tag rev1_s3_e999
        done
      done
    done

### 5.5 Table 6 — COVID-19 transfer at the subject level

`covid_rev1_eval.py` reuses the partitions of `covid_transfer_eval.py` (same
seeds), seeds every repeat, applies the augmentation in the (1, 696) layout,
and saves the score of every test spectrum with its subject under
`results/covid_rev1/<task>/<tag>/`. Encoders are those of the submitted
Table 5: the last checkpoints of supervised `version_ho_adam_es10_aug`, BYOL
`version_1`, and MoCo v3 and SimCLR v2 `version_bs1024`.

    for TASK in covid_vs_suspected covid_vs_healthy suspected_vs_healthy; do
      R="--task $TASK --repeats 0 50"
      # SVM on ANOVA positions (the baseline) and on the full spectrum
      python covid_rev1_eval.py $R --model svm --features anova
      python covid_rev1_eval.py $R --model svm --features full
      # logistic regression on the raw spectrum and on frozen encoder outputs
      for INIT in raw random supervised byol mocov3 simclrv2; do
        python covid_rev1_eval.py $R --model probe --init "$INIT"
      done
      # ResNets, without and with the augmentation
      for INIT in random supervised byol mocov3 simclrv2; do
        python covid_rev1_eval.py $R --model resnet --init "$INIT" --no-aug --features full
        python covid_rev1_eval.py $R --model resnet --init "$INIT" --aug --features full
      done
      # ResNets on ANOVA positions only
      for INIT in random simclrv2; do
        python covid_rev1_eval.py $R --model resnet --init "$INIT" --no-aug --features anova
      done
      # half and a quarter of the training subjects
      for F in 0.5 0.25; do
        python covid_rev1_eval.py $R --model svm --features anova --train_fraction "$F"
        for INIT in random supervised byol mocov3 simclrv2; do
          python covid_rev1_eval.py $R --model resnet --init "$INIT" --no-aug --features full \
              --train_fraction "$F"
        done
      done
    done

### 5.6 Compute cost (section 2.3.2)

Run on the GPU node; measures memory and GPU time of one pretraining step per
objective and the inference cost of the classifier:

    python scripts/rev1_compute_cost.py

### 5.7 Aggregation and figures

    python scripts/rev1_summary.py         # Tables 4 and 5, section 3.2, paired differences per seed
    python scripts/rev1_s0_summary.py      # accuracy against pretraining epoch, seed 0
    python scripts/rev1_variability.py     # within- and between-isolate variability against the views
    python scripts/rev1_covid_summary.py   # Table 6, subject-level metrics and corrected intervals
    python scripts/rev1_augbug_check.py    # effect of the (696, 1) layout in covid_transfer_eval.py
    python scripts/rev1_fig3.py --class2-root \
        results/bacteria-id/finetuning/class2/phys/supervised/version_ho_adam_es10_aug/ho_pre_adam_es10_last_ft_ho_split_adam_class2
                                           # Figure 3 with bootstrap deviations and per-isolate accuracy
