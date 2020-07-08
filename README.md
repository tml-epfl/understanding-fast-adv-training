# Understanding and Improving Fast Adversarial Training

**Maksym Andriushchenko (EPFL), Nicolas Flammarion (EPFL)**

**Paper:** [https://arxiv.org/abs/2007.02617](https://arxiv.org/abs/2007.02617)

**A short version is accepted to [ICML'20 UDL workshop](https://sites.google.com/view/udlworkshop2020/)**



## Abstract
A recent line of work focused on making adversarial training computationally efficient for deep learning models. 
In particular, Wong et al. (2020) showed that Linf-adversarial training with fast gradient sign method (FGSM) can fail 
due to a phenomenon called "catastrophic overfitting", when the model quickly loses its robustness over a single epoch 
of training. We show that adding a random step to FGSM, as proposed in Wong et al. (2020), does not prevent catastrophic 
overfitting, and that randomness is not important per se -- its main role being simply to reduce the magnitude of the 
perturbation. Moreover, we show that catastrophic overfitting is not inherent to deep and overparametrized networks, 
but can occur in a single-layer convolutional network with a few filters. In an extreme case, even a single filter can 
make the network highly non-linear locally, which is the main reason why FGSM training fails. Based on this observation, 
we propose a new regularization method, GradAlign, that prevents catastrophic overfitting by explicitly maximizing the 
gradient alignment inside the perturbation set and improves the quality of the FGSM solution. As a result, GradAlign 
allows to successfully apply FGSM training also for larger Linf-perturbations and reduce the gap to multi-step adversarial 
training.



## About the paper
We first show that not only FGSM training is prone to catastrophic overfitting, but the recently proposed fast 
adversarial training methods [34, 46] as well (see Fig. 1). 
<p align="center"><img src="img/fig1_robustnes_diff_eps_cifar10.png" width="700"></p>

We crucially observe that after catastrophic overfitting not just FGSM and PGD directions become misaligned, but even
gradients at two **random** points inside the Linf-ball (see the **right** plot).
<p align="center"><img src="img/resnet18_training_metrics.png" width="700"></p>

Surprisingly, this phenomenon is not inherent to deep and overparametrized networks, but can be 
observed even in a single-layer CNN. We analyze this setting both empirically and theoretically:
<p align="center"><img src="img/cnn4_training_metrics.png" width="700"></p>
<p align="center"><img src="img/cnn4_filters_plots.png" width="700"></p>
<p align="center"><img src="img/cnn4_feature_maps_small.png" width="295"></p>

The important property of FGSM training is that standard weight initialization schemes ensure high 
gradient alignment at the beginning of the training. We observe this empirically both in shallow
and deep networks, and formalize it for a single-layer CNN in the following lemma:
<p align="center"><img src="img/lemma2_grad_alignment_at_init.png" width="700"></p>
The high gradient alignment at initialization implies that at least at the beginning of the training,
FGSM solves the inner maximization problem accurately. However, this may change during training if the 
step size of FGSM is too large.

The importance of gradient alignment motivates our regularizer, **GradAlign**, that aims to increase the gradient alignment. 
<p align="center"><img src="img/grad_align_formula.png" width="500"></p>

**GradAlign** prevents catastrophic overfitting even for large Linf-perturbations and reduces the gap to multi-step adversarial training:
<p align="center"><img src="img/main_exps_curves.png" width="700"></p>



## Code

### Code of GradAlign


### Training code
The codebase is based on the codebase from [Wong et al, ICLR'20](https://arxiv.org/abs/2001.03994).
All the required dependencies are specified in `Dockerfile`.

The results reported in Fig. 1, Fig. 7, Tables 4 and 5 for CIFAR-10 and SVHN can be obtained by running the following scripts:
`sh/exps_diff_eps_cifar10.sh` and `sh/exps_diff_eps_svhn.sh` and varying the random seed from 0 to 4. 

Note that the evaluation is performed automatically at the end of training. 
In order to evaluate some model specifically, one can run `python eval.py` with appropriate parameters.


For ImageNet experiments see the .sh files `fgsm_eps6.sh`, `fgsmga_eps6_lambda0.1.sh`, `fgsmrs_eps6.sh`.
`python train.py`


### Models
Models will be uploaded soon.

The models can be evaluated via `python eval.py --eps=8 --n_eval=1000 --model='2020-03-19 23:51:05 dataset=cifar10 model=resnet18 eps=8.0 attack=pgd attack_init=zero fgsm_alpha=1.25 epochs=30 pgd_train_n_iters=7 grad_align_cos_lambda=0.0 seed=1 epoch=30'`

