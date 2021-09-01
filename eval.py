import argparse
import os
import utils
import apex.amp as amp
import numpy as np
import torch
import time
import data
import models
from utils import rob_acc


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--dataset', default='cifar10', choices=['mnist', 'svhn', 'cifar10', 'cifar10_binary', 'uniform_noise'], type=str)
    parser.add_argument('--model', default='resnet18', choices=['resnet18', 'cnn', 'fc', 'linear', 'lenet'], type=str)
    parser.add_argument('--set', default='test', type=str, choices=['train', 'test'])
    parser.add_argument('--model_path', default='2020-03-19 23:51:05 dataset=cifar10 model=resnet18 eps=8.0 attack=pgd attack_init=zero fgsm_alpha=1.25 epochs=30 pgd_train_n_iters=7 grad_align_cos_lambda=0.0 seed=1 epoch=30',
                        type=str, help='model name')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--eps', default=8, type=float)
    parser.add_argument('--n_eval', default=256, type=int, help='#examples to evaluate on')
    parser.add_argument('--n_layers', default=1, type=int, help='#layers on each conv layer (for model in [fc, cnn])')
    parser.add_argument('--n_filters_cnn', default=16, type=int, help='#filters on each conv layer (for model==cnn)')
    parser.add_argument('--n_hidden_fc', default=1024, type=int, help='#filters on each conv layer (for model==fc)')
    parser.add_argument('--batch_size_eval', default=1024, type=int, help='batch size for evaluation')
    parser.add_argument('--half_prec', action='store_true', help='eval in half precision')
    parser.add_argument('--early_stopped_model', action='store_true', help='eval the best model according to pgd_acc evaluated every k iters (typically, k=200)')
    return parser.parse_args()


args = get_args()
eps = args.eps
half_prec = args.half_prec  # for more reliable evaluations: keep in the single precision
print_stats = False
n_eval = args.n_eval
n_cls = 2 if 'binary' in args.dataset else 10
n_sampling_attack = 40
pgd_attack_iters = 50
pgd_alpha, pgd_alpha_rr, alpha_fgm = args.eps/4, args.eps/4, 300.0
pgd_rr_n_iter, pgd_rr_n_restarts = (50, 10)

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

np.set_printoptions(precision=4, suppress=True)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

model = models.get_model(args.model, n_cls, half_prec, data.shapes_dict[args.dataset], args.n_filters_cnn)
model = model.cuda()
model_dict = torch.load('models/{}.pth'.format(args.model_path))
if args.early_stopped_model:
    model.load_state_dict(model_dict['best'])
else:
    model.load_state_dict(model_dict['last'] if 'last' in model_dict else model_dict)

opt = torch.optim.SGD(model.parameters(), lr=0)  # needed for backprop only
if half_prec:
    model, opt = amp.initialize(model, opt, opt_level="O1")
utils.model_eval(model, half_prec)

eps, pgd_alpha, pgd_alpha_rr = eps / 255, pgd_alpha / 255, pgd_alpha_rr / 255

eval_batches_all = data.get_loaders(args.dataset, -1, args.batch_size_eval, train_set=True if args.set == 'train' else False,
                                    shuffle=False, data_augm=False)
eval_batches = data.get_loaders(args.dataset, n_eval, args.batch_size_eval, train_set=True if args.set == 'train' else False,
                                shuffle=False, data_augm=False)

time_start = time.time()
acc_clean, loss_clean, _ = rob_acc(eval_batches, model, 0, 0, opt, half_prec, 0, 1)
print('acc={:.2%}, loss={:.3f}'.format(acc_clean, loss_clean))

acc_pgd_rr, loss_pgd_rr, delta_pgd_rr = rob_acc(eval_batches, model, eps, pgd_alpha_rr, opt, half_prec, pgd_rr_n_iter, pgd_rr_n_restarts, print_fosc=False)
time_elapsed = time.time() - time_start

print('[test on {} points] acc_clean {:.2%}, pgd_rr {:.2%} ({:.2f}m)'.format(n_eval, acc_clean, acc_pgd_rr, time_elapsed/60))
