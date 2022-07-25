import argparse, logging, collections
import random, time, sys
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from utils import create__dir, count_parameters_in_MB
import utils
from Build_Dataset import build_train_cifar10, build_train_cifar100, build_train_Optimizer_Loss

from Node import NetworkCIFAR
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class individual():
    def __init__(self, dec):
        self.dec = dec
        self.init_channel = dec[0]  # 初始通道数
        self.stages = dec[1:4]  # stage操作列表 [[],[],[]]
        self.pools = dec[4]  # 两个pool层 [,]


def train_cifar10(train_queue, model, train_criterion, optimizer, args, epoch, since_time):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()
    total = len(train_queue)

    for step, (inputs, targets) in enumerate(train_queue):
        print('\r  Epoch{0:>2d}/600,   Training {1:>2d}/{2:>2d}, used_time {3:.2f}min]'.format(epoch, step + 1, total, (
                    time.time() - since_time) / 60), end='')

        inputs, targets = inputs.to(args.device), targets.to(args.device)
        optimizer.zero_grad()

        outputs = model(inputs)

        if args.use_aux_head:
            outputs, outputs_aux = outputs[0], outputs[1]

        loss = train_criterion(outputs, targets)
        if args.use_aux_head:
            loss_aux = train_criterion(outputs_aux, targets)
            loss += args.auxiliary_weight * loss_aux

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_bound)
        optimizer.step()

        prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
        n = inputs.size(0)
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)

        # if (step + 1) % 100 == 0:
        #     print('epoch:{}, step:{}, loss:{}, top1:{}, top5:{}'.format(epoch+1, step+1, objs.avg, top1.avg, top5.avg))

    return top1.avg, top5.avg, objs.avg


def evaluation_cifar10(valid_queue, model, eval_criterion, args):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    with torch.no_grad():
        model.eval()
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda()

            outputs = model(input)

            if args.use_aux_head:
                outputs, outputs_aux = outputs[0], outputs[1]

            loss = eval_criterion(outputs, target)

            prec1, prec5 = utils.accuracy(outputs, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)

    return top1.avg, top5.avg, objs.avg


def run_main(args):
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True

    solution = individual([16, [3, 0], [0, 0, 2, 1, 0, 5, 3], [6, 5, 2, 0, 1], [0, 6]])
    if args.dataset == 'cifar10':
        train_queue, valid_queue = build_train_cifar10(args=args, cutout_size=args.cutout_size,
                                                       autoaugment=args.autoaugment)
        args.classes = 10
    elif args.dataset == 'cifar100':
        args.classes = 100
        train_queue, valid_queue = build_train_cifar100(args=args, cutout_size=args.cutout_size,
                                                        autoaugment=args.autoaugment)


    model = NetworkCIFAR(args, args.classes, solution.init_channel, solution.stages, solution.pools,
                         args.use_aux_head, args.keep_prob)
    # print(model)

    print('Model: {0}, params: {1} M'.format('25-0', count_parameters_in_MB(model)))
    logging.info('Model: {0}, params: {1} M'.format('25-0', count_parameters_in_MB(model)))

    train_criterion, eval_criterion, optimizer, scheduler = build_train_Optimizer_Loss(model, args, epoch=-1)

    epoch = 0
    best_acc_top1 = 0
    since_time = time.time()
    while epoch < args.epochs:

        logging.info('epoch %d lr %e', epoch + 1, scheduler.get_last_lr()[0])
        print('epoch:{}, lr:{}, '.format(epoch + 1, scheduler.get_last_lr()[0]))

        train_acc, top5_avg, train_obj = train_cifar10(train_queue, model, train_criterion, optimizer, args, epoch, since_time)
        scheduler.step()

        logging.info('train_accuracy: %f, top5_avg: %f, loss: %f', train_acc, top5_avg, train_obj)
        print('\n       train_accuracy: {}, top5_avg: {}, loss: {}'.format(train_acc, top5_avg, train_obj))

        valid_acc_top1, valid_acc_top5, valid_obj = evaluation_cifar10(valid_queue, model, eval_criterion, args)

        logging.info('valid_accuracy: %f, valid_top5_accuracy: %f', valid_acc_top1, valid_acc_top5)
        print('         valid_accuracy: {}, valid_top5_accuracy: {}'.format(valid_acc_top1, valid_acc_top5))

        epoch += 1
        is_best = False
        if valid_acc_top1 > best_acc_top1:
            best_acc_top1 = valid_acc_top1
            is_best = True

            utils.save(args.save, args, model, epoch, epoch * (int(np.ceil(50000 / args.train_batch_size))), optimizer,
                       best_acc_top1, is_best)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train on cifar')
    # ***************************  common setting ******************
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('-save', type=str, default='result')
    parser.add_argument('-device', type=str, default='cuda')
    # ***************************  dataset setting ******************
    parser.add_argument('-data', type=str, default="/home/**/projects/data")
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10, cifar100'])
    parser.add_argument('-classes', type=int, default=10)  # 16
    parser.add_argument('-autoaugment', action='store_true', default=True)
    parser.add_argument('-cutout_size', type=int, default=16)  # 16
    # ***************************  optimization setting******************
    parser.add_argument('-epochs', type=int, default=600)
    parser.add_argument('-lr_max', type=float, default=0.025)  # 0.1
    parser.add_argument('-lr_min', type=float, default=0)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-l2_reg', type=float, default=5e-4)
    parser.add_argument('-grad_bound', type=float, default=5.0)
    parser.add_argument('-train_batch_size', type=int, default=80)
    parser.add_argument('-eval_batch_size', type=int, default=500)
    # ***************************  structure setting******************
    parser.add_argument('-search_last_channel', type=int, default=512)
    parser.add_argument('-use_aux_head', action='store_true', default=True)
    parser.add_argument('-auxiliary_weight', type=float, default=0.4)
    parser.add_argument('-keep_prob', type=float, default=0.6)
    args = parser.parse_args()

    # =====================================setting=======================================
    args.save = '{}/train_{}'.format(args.save, time.strftime("%Y-%m-%d-%H-%M-%S"))
    create__dir(args.save)

    # =====================================setting=======================================

    # ===================================  logging  ===================================
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(filename='{}/logs.log'.format(args.save),
                        level=logging.INFO, format=log_format, datefmt='%Y-%m-%d %I:%M:%S %p')

    logging.info("[Experiments Setting]\n" + "".join(
        ["[{0}]: {1}\n".format(name, value) for name, value in args.__dict__.items()]))

    run_main(args)
