#!/usr/bin/env bash
### To reproduce the curves, one has to run all the models for seed in {0, 1, 2, 3, 4}, and then average the results

# FGSM
python train.py --attack=fgsm --eps=1  --attack_init=zero --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=2  --attack_init=zero --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=3  --attack_init=zero --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=4  --attack_init=zero --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=5  --attack_init=zero --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=6  --attack_init=zero --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=7  --attack_init=zero --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=8  --attack_init=zero --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=9  --attack_init=zero --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=10 --attack_init=zero --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=11 --attack_init=zero --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=12 --attack_init=zero --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=13 --attack_init=zero --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=14 --attack_init=zero --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=15 --attack_init=zero --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=16 --attack_init=zero --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0

# FGSM-RS
python train.py --attack=fgsm --eps=1  --attack_init=random --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=2  --attack_init=random --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=3  --attack_init=random --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=4  --attack_init=random --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=5  --attack_init=random --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=6  --attack_init=random --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=7  --attack_init=random --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=8  --attack_init=random --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=9  --attack_init=random --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=10 --attack_init=random --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=11 --attack_init=random --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=12 --attack_init=random --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=13 --attack_init=random --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=14 --attack_init=random --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=15 --attack_init=random --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=16 --attack_init=random --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0

# GradAlign: the log-linear interpolation scheme: exp(eps * 1/8*ln(10) + ln(0.02))
python train.py --attack=fgsm --eps=1  --attack_init=zero --epochs=30 --grad_align_cos_lambda=0.027 --lr_max=0.30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=2  --attack_init=zero --epochs=30 --grad_align_cos_lambda=0.036 --lr_max=0.30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=3  --attack_init=zero --epochs=30 --grad_align_cos_lambda=0.047 --lr_max=0.30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=4  --attack_init=zero --epochs=30 --grad_align_cos_lambda=0.063 --lr_max=0.30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=5  --attack_init=zero --epochs=30 --grad_align_cos_lambda=0.084 --lr_max=0.30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=6  --attack_init=zero --epochs=30 --grad_align_cos_lambda=0.112 --lr_max=0.30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=7  --attack_init=zero --epochs=30 --grad_align_cos_lambda=0.150 --lr_max=0.30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=8  --attack_init=zero --epochs=30 --grad_align_cos_lambda=0.200 --lr_max=0.30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=9  --attack_init=zero --epochs=30 --grad_align_cos_lambda=0.267 --lr_max=0.30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=10 --attack_init=zero --epochs=30 --grad_align_cos_lambda=0.356 --lr_max=0.30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=11 --attack_init=zero --epochs=30 --grad_align_cos_lambda=0.474 --lr_max=0.30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=12 --attack_init=zero --epochs=30 --grad_align_cos_lambda=0.632 --lr_max=0.30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=13 --attack_init=zero --epochs=30 --grad_align_cos_lambda=0.843 --lr_max=0.30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=14 --attack_init=zero --epochs=30 --grad_align_cos_lambda=1.124 --lr_max=0.30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=15 --attack_init=zero --epochs=30 --grad_align_cos_lambda=1.500 --lr_max=0.30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=16 --attack_init=zero --epochs=30 --grad_align_cos_lambda=2.000 --lr_max=0.30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0

# PGD-10
python train.py --attack=pgd --eps=1  --attack_init=zero --pgd_alpha_train=0.2 --pgd_train_n_iters=10 --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=pgd --eps=2  --attack_init=zero --pgd_alpha_train=0.4 --pgd_train_n_iters=10 --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=pgd --eps=3  --attack_init=zero --pgd_alpha_train=0.6 --pgd_train_n_iters=10 --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=pgd --eps=4  --attack_init=zero --pgd_alpha_train=0.8 --pgd_train_n_iters=10 --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=pgd --eps=5  --attack_init=zero --pgd_alpha_train=1.0 --pgd_train_n_iters=10 --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=pgd --eps=6  --attack_init=zero --pgd_alpha_train=1.2 --pgd_train_n_iters=10 --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=pgd --eps=7  --attack_init=zero --pgd_alpha_train=1.4 --pgd_train_n_iters=10 --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=pgd --eps=8  --attack_init=zero --pgd_alpha_train=1.6 --pgd_train_n_iters=10 --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=pgd --eps=9  --attack_init=zero --pgd_alpha_train=1.8 --pgd_train_n_iters=10 --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=pgd --eps=10 --attack_init=zero --pgd_alpha_train=2.0 --pgd_train_n_iters=10 --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=pgd --eps=11 --attack_init=zero --pgd_alpha_train=2.2 --pgd_train_n_iters=10 --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=pgd --eps=12 --attack_init=zero --pgd_alpha_train=2.4 --pgd_train_n_iters=10 --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=pgd --eps=13 --attack_init=zero --pgd_alpha_train=2.6 --pgd_train_n_iters=10 --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=pgd --eps=14 --attack_init=zero --pgd_alpha_train=2.8 --pgd_train_n_iters=10 --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=pgd --eps=15 --attack_init=zero --pgd_alpha_train=3.0 --pgd_train_n_iters=10 --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=pgd --eps=16 --attack_init=zero --pgd_alpha_train=3.2 --pgd_train_n_iters=10 --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0

# PGD-2
python train.py --attack=pgd --eps=1  --attack_init=zero --pgd_alpha_train=0.5 --pgd_train_n_iters=2 --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=pgd --eps=2  --attack_init=zero --pgd_alpha_train=1.0 --pgd_train_n_iters=2 --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=pgd --eps=3  --attack_init=zero --pgd_alpha_train=1.5 --pgd_train_n_iters=2 --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=pgd --eps=4  --attack_init=zero --pgd_alpha_train=2.0 --pgd_train_n_iters=2 --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=pgd --eps=5  --attack_init=zero --pgd_alpha_train=2.5 --pgd_train_n_iters=2 --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=pgd --eps=6  --attack_init=zero --pgd_alpha_train=3.0 --pgd_train_n_iters=2 --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=pgd --eps=7  --attack_init=zero --pgd_alpha_train=3.5 --pgd_train_n_iters=2 --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=pgd --eps=8  --attack_init=zero --pgd_alpha_train=4.0 --pgd_train_n_iters=2 --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=pgd --eps=9  --attack_init=zero --pgd_alpha_train=4.5 --pgd_train_n_iters=2 --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=pgd --eps=10 --attack_init=zero --pgd_alpha_train=5.0 --pgd_train_n_iters=2 --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=pgd --eps=11 --attack_init=zero --pgd_alpha_train=5.5 --pgd_train_n_iters=2 --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=pgd --eps=12 --attack_init=zero --pgd_alpha_train=6.0 --pgd_train_n_iters=2 --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=pgd --eps=13 --attack_init=zero --pgd_alpha_train=6.5 --pgd_train_n_iters=2 --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=pgd --eps=14 --attack_init=zero --pgd_alpha_train=7.0 --pgd_train_n_iters=2 --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=pgd --eps=15 --attack_init=zero --pgd_alpha_train=7.5 --pgd_train_n_iters=2 --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=pgd --eps=16 --attack_init=zero --pgd_alpha_train=8.0 --pgd_train_n_iters=2 --epochs=30 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0

# Free-AT
python train.py --attack=fgsm --eps=1 --attack_init=zero --fgsm_alpha=1.0 --minibatch_replay=8 --epochs=96 --lr_max=0.04 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=2 --attack_init=zero --fgsm_alpha=1.0 --minibatch_replay=8 --epochs=96 --lr_max=0.04 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=3 --attack_init=zero --fgsm_alpha=1.0 --minibatch_replay=8 --epochs=96 --lr_max=0.04 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=4 --attack_init=zero --fgsm_alpha=1.0 --minibatch_replay=8 --epochs=96 --lr_max=0.04 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=5 --attack_init=zero --fgsm_alpha=1.0 --minibatch_replay=8 --epochs=96 --lr_max=0.04 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=6 --attack_init=zero --fgsm_alpha=1.0 --minibatch_replay=8 --epochs=96 --lr_max=0.04 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=7 --attack_init=zero --fgsm_alpha=1.0 --minibatch_replay=8 --epochs=96 --lr_max=0.04 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=8 --attack_init=zero --fgsm_alpha=1.0 --minibatch_replay=8 --epochs=96 --lr_max=0.04 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=9 --attack_init=zero --fgsm_alpha=1.0 --minibatch_replay=8 --epochs=96 --lr_max=0.04 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=10 --attack_init=zero --fgsm_alpha=1.0 --minibatch_replay=8 --epochs=96 --lr_max=0.04 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=11 --attack_init=zero --fgsm_alpha=1.0 --minibatch_replay=8 --epochs=96 --lr_max=0.04 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=12 --attack_init=zero --fgsm_alpha=1.0 --minibatch_replay=8 --epochs=96 --lr_max=0.04 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=13 --attack_init=zero --fgsm_alpha=1.0 --minibatch_replay=8 --epochs=96 --lr_max=0.04 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=14 --attack_init=zero --fgsm_alpha=1.0 --minibatch_replay=8 --epochs=96 --lr_max=0.04 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=15 --attack_init=zero --fgsm_alpha=1.0 --minibatch_replay=8 --epochs=96 --lr_max=0.04 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
python train.py --attack=fgsm --eps=16 --attack_init=zero --fgsm_alpha=1.0 --minibatch_replay=8 --epochs=96 --lr_max=0.04 --eval_iter_freq=200 --batch_size_eval=1024 --half_prec --eval_early_stopped_model --n_final_eval=1000 --seed=0
