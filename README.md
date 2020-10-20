# Understanding and Improving Fast Adversarial Training

**Maksym Andriushchenko (EPFL), Nicolas Flammarion (EPFL)**

**Paper:** [https://arxiv.org/abs/2007.02617](https://arxiv.org/abs/2007.02617)

**NeurIPS 2020**



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
The following code snippet shows a concise implementation of **GradAlign** (see `train.py` for more details):
```python
grad1 = utils.get_input_grad(model, X, y, opt, eps, half_prec, delta_init='none', backprop=True)
grad2 = utils.get_input_grad(model, X, y, opt, eps, half_prec, delta_init='random_uniform', backprop=True)
grad1, grad2 = grad1.reshape(len(grad1), -1), grad2.reshape(len(grad2), -1)
cos = torch.nn.functional.cosine_similarity(grad1, grad2, 1)
reg = grad_align_lambda * (1.0 - cos.mean())
```


### Training code
This code of `train.py` is partially based on the code from [Wong et al, ICLR'20](https://arxiv.org/abs/2001.03994).
All the required dependencies for our code are specified in `Dockerfile`.

Training ResNet-18 using FGSM+GradAlign on CIFAR-10 can be done as follows:
`python train.py --dataset=cifar10 --attack=fgsm --eps=8  --attack_init=zero --epochs=30 --grad_align_cos_lambda=0.2 --lr_max=0.30 --half_prec --n_final_eval=1000`

Training CNN with 4 filters using FGSM (as reported in the paper) can be done via:
`python train.py --model=cnn --attack=fgsm --eps=10 --attack_init=zero --n_layers=1 --n_filters_cnn=4  --epochs=30 --eval_iter_freq=50 --lr_max=0.003 --gpu=0 --n_final_eval=1000`


The results reported in Fig. 1, Fig. 7, Tables 4 and 5 for CIFAR-10 and SVHN can be obtained by running the following scripts:
`sh/exps_diff_eps_cifar10.sh` and `sh/exps_diff_eps_svhn.sh` and varying the random seed from 0 to 4. 


Note that the evaluation is performed automatically at the end of training. 
In order to evaluate some model specifically, one can run the evaluation script
`python eval.py --eps=8 --n_eval=1000 --model='<model name>'`.


### Models
GradAlign models will be uploaded soon.

The models can be evaluated via 
`python eval.py --eps=8 --n_eval=1000 --model='<model name>'`


### Citation
```
@inproceedings{andriushchenko2020understanding,
  title={Understanding and Improving Fast Adversarial Training},
  author={Andriushchenko, Maksym and Flammarion, Nicolas},
  booktitle={NeurIPS},
  year={2020}
}
```
