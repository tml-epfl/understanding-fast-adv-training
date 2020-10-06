"""
This code is partially based on the repository of https://github.com/locuslab/fast_adversarial (Wong et al., ICLR'20)
"""
import argparse
import os
import time
import numpy as np
import apex.amp as amp
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import utils
import data
import models

from datetime import datetime
from utils import rob_acc, l2_norm_batch, get_input_grad, clamp


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--data_dir', default='../cifar-data', type=str)
    parser.add_argument('--dataset', default='cifar10', choices=['mnist', 'svhn', 'cifar10', 'cifar10_binary', 'cifar10_binary_gs',
                                                                 'uniform_noise', 'imagenet'], type=str)
    parser.add_argument('--model', default='resnet18', choices=['resnet18', 'lenet', 'cnn'], type=str)
    parser.add_argument('--epochs', default=30, type=int,
                        help='15 epochs to reach 45% adv acc, 30 epochs to reach the reported clean/adv accs')
    parser.add_argument('--lr_schedule', default='cyclic', choices=['cyclic', 'piecewise'])
    parser.add_argument('--lr_max', default=0.2, type=float, help='0.05 in Table 1, 0.2 in Figure 2')
    parser.add_argument('--attack', default='fgsm', type=str, choices=['pgd', 'pgd_corner', 'fgsm', 'random_corner', 'free', 'none'])
    parser.add_argument('--eps', default=8.0, type=float)
    parser.add_argument('--attack_iters', default=10, type=int, help='n_iter of pgd for evaluation')
    parser.add_argument('--pgd_train_n_iters', default=10, type=int, help='n_iter of pgd for training (if attack=pgd)')
    parser.add_argument('--pgd_alpha_train', default=2.0, type=float)
    parser.add_argument('--fgsm_alpha', default=1.25, type=float)
    parser.add_argument('--minibatch_replay', default=1, type=int, help='minibatch replay as in AT for Free (default=1 is usual training)')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay aka l2 regularization')
    parser.add_argument('--attack_init', default='random', choices=['zero', 'random'])
    parser.add_argument('--fname', default='plain_cifar10', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--half_prec', action='store_true', help='if enabled, runs everything as half precision')
    parser.add_argument('--grad_align_cos_lambda', default=0.0, type=float, help='coefficient of the cosine gradient alignment regularizer')
    parser.add_argument('--eval_early_stopped_model', action='store_true', help='whether to evaluate the model obtained via early stopping')
    parser.add_argument('--eval_iter_freq', default=200, type=int, help='how often to evaluate test stats')
    parser.add_argument('--n_eval_every_k_iter', default=256, type=int, help='on how many examples to eval every k iters')
    parser.add_argument('--n_layers', default=1, type=int, help='#layers on each conv layer (for model == cnn)')
    parser.add_argument('--n_filters_cnn', default=16, type=int, help='#filters on each conv layer (for model==cnn)')
    parser.add_argument('--batch_size_eval', default=256, type=int, help='batch size for the final eval with pgd rr; 6 GB memory is consumed for 1024 examples with fp32 network')
    parser.add_argument('--n_final_eval', default=-1, type=int, help='on how many examples to do the final evaluation; -1 means on all test examples.')
    return parser.parse_args()


def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cur_timestamp = str(datetime.now())[:-3]  # we also include ms to prevent the probability of name collision
    model_width = {'linear': '', 'cnn': args.n_filters_cnn, 'lenet': '', 'resnet18': ''}[args.model]
    model_str = '{}{}'.format(args.model, model_width)
    model_name = '{} dataset={} model={} eps={} attack={} m={} attack_init={} fgsm_alpha={} epochs={} pgd={}-{} grad_align_cos_lambda={} lr_max={} seed={}'.format(
        cur_timestamp, args.dataset, model_str, args.eps, args.attack, args.minibatch_replay, args.attack_init, args.fgsm_alpha, args.epochs,
        args.pgd_alpha_train, args.pgd_train_n_iters, args.grad_align_cos_lambda, args.lr_max, args.seed)
    if not os.path.exists('models'):
        os.makedirs('models')
    logger = utils.configure_logger(model_name, args.debug)
    logger.info(args)
    half_prec = args.half_prec
    n_cls = 2 if 'binary' in args.dataset else 10

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    double_bp = True if args.grad_align_cos_lambda > 0 else False
    n_eval_every_k_iter = args.n_eval_every_k_iter
    args.pgd_alpha = args.eps / 4

    eps, pgd_alpha, pgd_alpha_train = args.eps / 255, args.pgd_alpha / 255, args.pgd_alpha_train / 255
    train_data_augm = False if args.dataset in ['mnist'] else True
    train_batches = data.get_loaders(args.dataset, -1, args.batch_size, train_set=True, shuffle=True, data_augm=train_data_augm)
    train_batches_fast = data.get_loaders(args.dataset, n_eval_every_k_iter, args.batch_size, train_set=True, shuffle=False, data_augm=False)
    test_batches = data.get_loaders(args.dataset, args.n_final_eval, args.batch_size_eval, train_set=False, shuffle=False, data_augm=False)
    test_batches_fast = data.get_loaders(args.dataset, n_eval_every_k_iter, args.batch_size_eval, train_set=False, shuffle=False, data_augm=False)

    model = models.get_model(args.model, n_cls, half_prec, data.shapes_dict[args.dataset], args.n_filters_cnn).cuda()
    model.apply(utils.initialize_weights)
    model.train()

    if args.model == 'resnet18':
        opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=0.9, weight_decay=args.weight_decay)
    elif args.model == 'cnn':
        opt = torch.optim.Adam(model.parameters(), lr=args.lr_max, weight_decay=args.weight_decay)
    elif args.model == 'lenet':
        opt = torch.optim.Adam(model.parameters(), lr=args.lr_max, weight_decay=args.weight_decay)
    else:
        raise ValueError('decide about the right optimizer for the new model')

    if half_prec:
        if double_bp:
            amp.register_float_function(torch, 'batch_norm')
        model, opt = amp.initialize(model, opt, opt_level="O1")

    if args.attack == 'fgsm':  # needed here only for Free-AT
        delta = torch.zeros(args.batch_size, *data.shapes_dict[args.dataset][1:]).cuda()
        delta.requires_grad = True

    lr_schedule = utils.get_lr_schedule(args.lr_schedule, args.epochs, args.lr_max)
    loss_function = nn.CrossEntropyLoss()

    train_acc_pgd_best, best_state_dict = 0.0, copy.deepcopy(model.state_dict())
    start_time = time.time()
    time_train, iteration, best_iteration = 0, 0, 0
    for epoch in range(args.epochs + 1):
        train_loss, train_reg, train_acc, train_n, grad_norm_x, avg_delta_l2 = 0, 0, 0, 0, 0, 0
        for i, (X, y) in enumerate(train_batches):
            if i % args.minibatch_replay != 0 and i > 0:  # take new inputs only each `minibatch_replay` iterations
                X, y = X_prev, y_prev
            time_start_iter = time.time()
            # epoch=0 runs only for one iteration (to check the training stats at init)
            if epoch == 0 and i > 0:
                break
            X, y = X.cuda(), y.cuda()
            lr = lr_schedule(epoch - 1 + (i + 1) / len(train_batches))  # epoch - 1 since the 0th epoch is skipped
            opt.param_groups[0].update(lr=lr)

            if args.attack in ['pgd', 'pgd_corner']:
                pgd_rs = True if args.attack_init == 'random' else False
                n_eps_warmup_epochs = 5
                n_iterations_max_eps = n_eps_warmup_epochs * data.shapes_dict[args.dataset][0] // args.batch_size
                eps_pgd_train = min(iteration / n_iterations_max_eps * eps, eps) if args.dataset == 'svhn' else eps
                delta = utils.attack_pgd_training(
                    model, X, y, eps_pgd_train, pgd_alpha_train, opt, half_prec, args.pgd_train_n_iters, rs=pgd_rs)
                if args.attack == 'pgd_corner':
                    delta = eps * utils.sign(delta)  # project to the corners
                    delta = clamp(X + delta, 0, 1) - X

            elif args.attack == 'fgsm':
                if args.minibatch_replay == 1:
                    if args.attack_init == 'zero':
                        delta = torch.zeros_like(X, requires_grad=True)
                    elif args.attack_init == 'random':
                        delta = utils.get_uniform_delta(X.shape, eps, requires_grad=True)
                    else:
                        raise ValueError('wrong args.attack_init')
                else:  # if Free-AT, we just reuse the existing delta from the previous iteration
                    delta.requires_grad = True

                X_adv = clamp(X + delta, 0, 1)
                output = model(X_adv)
                loss = F.cross_entropy(output, y)
                if half_prec:
                    with amp.scale_loss(loss, opt) as scaled_loss:
                        grad = torch.autograd.grad(scaled_loss, delta, create_graph=True if double_bp else False)[0]
                        grad /= scaled_loss / loss  # reverse back the scaling
                else:
                    grad = torch.autograd.grad(loss, delta, create_graph=True if double_bp else False)[0]

                grad = grad.detach()

                argmax_delta = eps * utils.sign(grad)

                n_alpha_warmup_epochs = 5
                n_iterations_max_alpha = n_alpha_warmup_epochs * data.shapes_dict[args.dataset][0] // args.batch_size
                fgsm_alpha = min(iteration / n_iterations_max_alpha * args.fgsm_alpha, args.fgsm_alpha) if args.dataset == 'svhn' else args.fgsm_alpha
                delta.data = clamp(delta.data + fgsm_alpha * argmax_delta, -eps, eps)
                delta.data = clamp(X + delta.data, 0, 1) - X

            elif args.attack == 'random_corner':
                delta = utils.get_uniform_delta(X.shape, eps, requires_grad=False)
                delta = eps * utils.sign(delta)

            elif args.attack == 'none':
                delta = torch.zeros_like(X, requires_grad=False)
            else:
                raise ValueError('wrong args.attack')

            # extra FP+BP to calculate the gradient to monitor it
            if args.attack in ['none', 'random_corner', 'pgd', 'pgd_corner']:
                grad = get_input_grad(model, X, y, opt, eps, half_prec, delta_init='none',
                                      backprop=args.grad_align_cos_lambda != 0.0)

            delta = delta.detach()

            output = model(X + delta)
            loss = loss_function(output, y)

            reg = torch.zeros(1).cuda()[0]  # for .item() to run correctly
            if args.grad_align_cos_lambda != 0.0:
                grad2 = get_input_grad(model, X, y, opt, eps, half_prec, delta_init='random_uniform', backprop=True)
                grads_nnz_idx = ((grad**2).sum([1, 2, 3])**0.5 != 0) * ((grad2**2).sum([1, 2, 3])**0.5 != 0)
                grad1, grad2 = grad[grads_nnz_idx], grad2[grads_nnz_idx]
                grad1_norms, grad2_norms = l2_norm_batch(grad1), l2_norm_batch(grad2)
                grad1_normalized = grad1 / grad1_norms[:, None, None, None]
                grad2_normalized = grad2 / grad2_norms[:, None, None, None]
                cos = torch.sum(grad1_normalized * grad2_normalized, (1, 2, 3))
                reg += args.grad_align_cos_lambda * (1.0 - cos.mean())

            loss += reg

            if epoch != 0:
                opt.zero_grad()
                utils.backward(loss, opt, half_prec)
                opt.step()

            time_train += time.time() - time_start_iter
            train_loss += loss.item() * y.size(0)
            train_reg += reg.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)

            with torch.no_grad():  # no grad for the stats
                grad_norm_x += l2_norm_batch(grad).sum().item()
                delta_final = clamp(X + delta, 0, 1) - X  # we should measure delta after the projection onto [0, 1]^d
                avg_delta_l2 += ((delta_final ** 2).sum([1, 2, 3]) ** 0.5).sum().item()

            if iteration % args.eval_iter_freq == 0:
                train_loss, train_reg = train_loss / train_n, train_reg / train_n
                train_acc, avg_delta_l2 = train_acc / train_n, avg_delta_l2 / train_n

                # it'd be incorrect to recalculate the BN stats on the test sets and for clean / adversarial points
                utils.model_eval(model, half_prec)

                test_acc_clean, _, _ = rob_acc(test_batches_fast, model, eps, pgd_alpha, opt, half_prec, 0, 1)
                test_acc_fgsm, test_loss_fgsm, fgsm_deltas = rob_acc(test_batches_fast, model, eps, eps, opt, half_prec, 1, 1, rs=False)
                test_acc_pgd, test_loss_pgd, pgd_deltas = rob_acc(test_batches_fast, model, eps, pgd_alpha, opt, half_prec, args.attack_iters, 1)
                cos_fgsm_pgd = utils.avg_cos_np(fgsm_deltas, pgd_deltas)
                train_acc_pgd, _, _ = rob_acc(train_batches_fast, model, eps, pgd_alpha, opt, half_prec, args.attack_iters, 1)  # needed for early stopping

                grad_x = utils.get_grad_np(model, test_batches_fast, eps, opt, half_prec, rs=False)
                grad_eta = utils.get_grad_np(model, test_batches_fast, eps, opt, half_prec, rs=True)
                cos_x_eta = utils.avg_cos_np(grad_x, grad_eta)

                time_elapsed = time.time() - start_time
                train_str = '[train] loss {:.3f}, reg {:.3f}, acc {:.2%} acc_pgd {:.2%}'.format(train_loss, train_reg, train_acc, train_acc_pgd)
                test_str = '[test] acc_clean {:.2%}, acc_fgsm {:.2%}, acc_pgd {:.2%}, cos_x_eta {:.3}, cos_fgsm_pgd {:.3}'.format(
                    test_acc_clean, test_acc_fgsm, test_acc_pgd, cos_x_eta, cos_fgsm_pgd)
                logger.info('{}-{}: {}  {} ({:.2f}m, {:.2f}m)'.format(epoch, iteration, train_str, test_str,
                                                                      time_train/60, time_elapsed/60))

                if train_acc_pgd > train_acc_pgd_best:  # catastrophic overfitting can be detected on the training set
                    best_state_dict = copy.deepcopy(model.state_dict())
                    train_acc_pgd_best, best_iteration = train_acc_pgd, iteration

                utils.model_train(model, half_prec)
                train_loss, train_reg, train_acc, train_n, grad_norm_x, avg_delta_l2 = 0, 0, 0, 0, 0, 0

            iteration += 1
            X_prev, y_prev = X.clone(), y.clone()  # needed for Free-AT

        if epoch == args.epochs:
            torch.save({'last': model.state_dict(), 'best': best_state_dict}, 'models/{} epoch={}.pth'.format(model_name, epoch))
            # disable global conversion to fp16 from amp.initialize() (https://github.com/NVIDIA/apex/issues/567)
            context_manager = amp.disable_casts() if half_prec else utils.nullcontext()
            with context_manager:
                last_state_dict = copy.deepcopy(model.state_dict())
                half_prec = False  # final eval is always in fp32
                model.load_state_dict(last_state_dict)
                utils.model_eval(model, half_prec)
                opt = torch.optim.SGD(model.parameters(), lr=0)

                attack_iters, n_restarts = (50, 10) if not args.debug else (10, 3)
                test_acc_clean, _, _ = rob_acc(test_batches, model, eps, pgd_alpha, opt, half_prec, 0, 1)
                test_acc_pgd_rr, _, deltas_pgd_rr = rob_acc(test_batches, model, eps, pgd_alpha, opt, half_prec, attack_iters, n_restarts)
                logger.info('[last: test on 10k points] acc_clean {:.2%}, pgd_rr {:.2%}'.format(test_acc_clean, test_acc_pgd_rr))

                if args.eval_early_stopped_model:
                    model.load_state_dict(best_state_dict)
                    utils.model_eval(model, half_prec)
                    test_acc_clean, _, _ = rob_acc(test_batches, model, eps, pgd_alpha, opt, half_prec, 0, 1)
                    test_acc_pgd_rr, _, deltas_pgd_rr = rob_acc(test_batches, model, eps, pgd_alpha, opt, half_prec, attack_iters, n_restarts)
                    logger.info('[best: test on 10k points][iter={}] acc_clean {:.2%}, pgd_rr {:.2%}'.format(
                        best_iteration, test_acc_clean, test_acc_pgd_rr))

        utils.model_train(model, half_prec)

    logger.info('Done in {:.2f}m'.format((time.time() - start_time) / 60))


if __name__ == "__main__":
    main()
