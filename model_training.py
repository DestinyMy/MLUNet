import argparse, logging
import os.path
import random, time, sys
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from Build_Dataset import build_search_Optimizer_Loss
from Node import NetworkCIFAR

import utils


# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # 该设置GPU的方式必须在最开始指定，最好在import torch之前，其他地方不管用；优点在于强制程序可见


def Model_train(f, solution_id, train_queue, model, train_criterion, optimizer, scheduler, args, valid_queue,
                eval_criterion, print_=False):
    since_time = time.time()
    total = len(train_queue)

    for epoch in range(args.search_epochs):
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()
        model.train()

        for step, (inputs, targets) in enumerate(train_queue):
            # print('\r[Epoch:{0:>2d}/{1:>2d}, Training {2:>2d}/{3:>2d}, used_time {4:.2f}min]'.format(epoch + 1,
            #                                                                                          args.search_epochs,
            #                                                                                          step + 1, total,
            #                                                                                          (time.time() - since_time) / 60),
            #                                                                                          end='')

            inputs, targets = inputs.to(args.device), targets.to(args.device)

            optimizer.zero_grad()
            outputs = model(inputs)

            if args.search_use_aux_head:
                outputs, outputs_aux = outputs[0], outputs[1]

            loss = train_criterion(outputs, targets)
            if args.search_use_aux_head:
                loss_aux = train_criterion(outputs_aux, targets)
                loss += args.search_auxiliary_weight * loss_aux

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.search_grad_bound)
            optimizer.step()

            # outputs ,targets = outputs.to('cpu'), targets.to('cpu')
            prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
            n = inputs.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)

        scheduler.step()

        if print_ or (epoch + 1) == args.search_epochs:
            # logging.info('epoch %d lr %e', epoch + 1, scheduler.get_last_lr()[0])
            print('{0}-th solution train accuracy top1:{1:.3f}, train accuracy top5:{2:.3f}, train loss:{3:.5f}'.format(
                solution_id, top1.avg,
                top5.avg,
                objs.avg),
                file=sys.stdout)
            print('{0}-th solution train accuracy top1:{1:.3f}, train accuracy top5:{2:.3f}, train loss:{3:.5f}'.format(
                solution_id, top1.avg,
                top5.avg,
                objs.avg),
                file=f, flush=True)

            logging.info(
                '{0}-th solution train accuracy top1:{1:.3f}, train accuracy top5:{2:.3f}, train loss:{3:.5f}'.format(
                    solution_id, top1.avg,
                    top5.avg,
                    objs.avg))

            valid_top1_acc, valid_top5_acc, loss = Model_valid(f, solution_id, valid_queue, model, eval_criterion, args)

    used_time = (time.time() - since_time) / 60
    print('{0}-th solution\'s used_time {1:.2f}min'.format(solution_id, used_time))
    return top1.avg, top5.avg, objs.avg, valid_top1_acc, valid_top5_acc, loss, used_time


def Model_valid(f, solution_id, valid_queue, model, eval_criterion, args):
    total = len(valid_queue)

    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    with torch.no_grad():
        model.eval()
        for step, (inputs, targets) in enumerate(valid_queue):
            # print('\r[-------------Validating {0:>2d}/{1:>2d}]'.format(step + 1, total), end='')

            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = model(inputs)
            if args.search_use_aux_head:
                outputs, outputs_aux = outputs[0], outputs[1]

            loss = eval_criterion(outputs, targets)

            prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
            n = inputs.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)

    print('{0}-th solution valid accuracy top1:{1:.3f}, valid accuracy top5:{2:.3f}, valid loss:{3:.5f}'.format(
        solution_id, top1.avg, top5.avg,
        objs.avg),
          file=sys.stdout)
    print('{0}-th solution valid accuracy top1:{1:.3f}, valid accuracy top5:{2:.3f}, valid loss:{3:.5f}'.format(
        solution_id, top1.avg, top5.avg,
        objs.avg), file=f,
          flush=True)
    logging.info(
        '{0}-th solution valid accuracy top1:{1:.3f}, valid accuracy top5:{2:.3f}, valid loss:{3:.5f}'.format(
            solution_id, top1.avg, top5.avg,
            objs.avg))

    return top1.avg, top5.avg, objs.avg


def solution_evaluation(device_id, f, solution_id, init_channel, stages, pools, train_queue, valid_queue, args):
    torch.cuda.set_device(device_id)  # 构建网络之前，可在程序不同地方指定，但是只能指定一块GPU

    model = NetworkCIFAR(args, 10, init_channel, stages, pools, args.search_use_aux_head,
                         args.search_keep_prob)

    #  ============================================ build optimizer, loss and scheduler ============================================
    train_criterion, eval_criterion, optimizer, scheduler = build_search_Optimizer_Loss(model, args, last_epoch=-1)
    #  ============================================ training the individual model and get valid accuracy ============================================
    result = Model_train(f, solution_id, train_queue, model, train_criterion, optimizer, scheduler, args, valid_queue,
                         eval_criterion, print_=False)  # True


    # return (1 - result[3] / 100).cpu(), num_parameters, Flops, result[6], 0  # validate error, params, Flops, used_time, padding
    return np.array([(1 - result[3] / 100).cpu(), 0])  # validate error, padding
