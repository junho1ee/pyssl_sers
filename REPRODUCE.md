# Reproducing the reported results

Every number in the manuscript comes from one of the runs below. The batch
scripts that submitted them were written for a specific SLURM site and are not
distributed; the commands they wrapped are given here in full. Each run writes a
`hparams.yaml` next to its outputs that records the complete configuration
actually used, so a stored run can always be checked against this file.

Runs are identified by their `--run_tag`, which is also the directory name under
`results/bacteria-id/finetuning/`.

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

### Table 5 — COVID-19 transfer, 50 repeats under both partitioning protocols

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
is the measurement behind the claim that the encoder peaks two thirds of the way
through the fixed number of epochs and then loses accuracy:

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
