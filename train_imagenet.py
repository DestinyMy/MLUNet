import argparse, logging, collections
import random, time, sys, os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as dset

from utils import create__dir, count_parameters_in_MB
import utils
from Node import NetworkImageNet
from autoaugment import ImageNetPolicy

# from folder2lmdb import ImageFolderLMDB
from prefetch_generator import BackgroundGenerator

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'


class DataLoaderX(torch.utils.data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class individual():
    def __init__(self, dec):
        self.dec = dec
        self.init_channel = dec[0]  # 初始通道数
        self.stages = dec[1:4]  # stage操作列表 [[],[],[]]
        self.pools = dec[4]  # 两个pool层 [,]


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def train(train_queue, model, train_criterion, optimizer, args, epoch, since_time):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()
    total = len(train_queue)
    data_time = utils.AvgrageMeter()
    batch_time = utils.AvgrageMeter()
    end = time.time()

    for step, (inputs, targets) in enumerate(train_queue):
        data_time.update(time.time() - end)
        # inputs, targets = inputs.to(args.device,non_blocking=True), targets.to(args.device,non_blocking=True)
        inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
        optimizer.zero_grad()

        outputs = model(inputs)

        if args.use_aux_head:
            outputs, outputs_aux = outputs[0], outputs[1]

        loss = train_criterion(outputs, targets)
        if args.use_aux_head:
            loss_aux = train_criterion(outputs_aux, targets)
            loss += args.auxiliary_weight * loss_aux

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
        n = inputs.size(0)
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)

        batch_time.update(time.time() - end)
        end = time.time()

        # print('\r  Epoch{0:>2d}/250, Training {1:>2d}/{2:>2d}, data time:{3:.4f}s, batch time:{4:.4f}s, total_used_time {5:.3f}min]'.
        #       format(epoch,step + 1, total, data_time.avg,batch_time.avg,(time.time() - since_time)/60 ),end='')
        print(
            '\r  Epoch{0:>2d}/{1:>2d}, Training {2:>2d}/{3:>2d}, data time:{4}s, batch time:{5}s, total_used_time {6:.3f}min]'.
                format(epoch, args.epochs, step + 1, total, data_time._print, batch_time._print, (time.time() - since_time) / 60),
            end='')

    return top1.avg, top5.avg, objs.avg


def valid(valid_queue, model, eval_criterion, args):
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


def build_imagenet(model_state_dict, optimizer_state_dict, **kwargs):
    solution = kwargs.pop('solution')
    epoch = kwargs.pop('epoch')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if args.autoaugment:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            ImageNetPolicy(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2),
            transforms.ToTensor(),
            normalize,
        ])

    valid_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    if args.load == 'lmdb':
        logging.info('Loading data from lmdb file')
        traindir = os.path.join(args.data, 'train.lmdb')
        validdir = os.path.join(args.data, 'val.lmdb')
        print('https://github.com/xunge/pytorch_lmdb_imagenet')

        train_data = ImageFolderLMDB(traindir, train_transform)
        valid_data = ImageFolderLMDB(validdir, valid_transform)
    elif args.load == 'original':
        logging.info('Loading data from directory')
        traindir = os.path.join(args.data, 'train')
        validdir = os.path.join(args.data, 'val')

        train_data = dset.ImageFolder(traindir, train_transform)
        valid_data = dset.ImageFolder(validdir, valid_transform)
    elif args.load == 'memory':
        logging.info('Loading data into memory')
        traindir = os.path.join(args.data, 'train')
        validdir = os.path.join(args.data, 'val')
        train_data = utils.InMemoryDataset(traindir, train_transform, num_workers=args.num_workers)
        valid_data = utils.InMemoryDataset(validdir, valid_transform, num_workers=args.num_workers)

    logging.info('Found %d in training data', len(train_data))
    logging.info('Found %d in validation data', len(valid_data))

    # ------------------------------------------------ steps -------------------------------------------------
    args.steps = int(np.ceil(len(train_data) / (args.batch_size))) * torch.cuda.device_count() * args.epochs
    # ---------------------------------------------------------------------------------------------------------

    train_queue = DataLoaderX(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
    valid_queue = DataLoaderX(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    # 0.00005s for DataLoaderX each batch=256
    # 0.0003s  for DataLoader  each batch=256

    # train_queue = torch.utils.data.DataLoader(
    #     train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
    # valid_queue = torch.utils.data.DataLoader(
    #     valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    model = NetworkImageNet(args, args.classes, solution.init_channel, solution.stages, solution.pools,
                            args.use_aux_head, args.keep_prob)

    print('Model Parameters: {} MB'.format(count_parameters_in_MB(model)))
    logging.info('Model Parameters: %f MB', count_parameters_in_MB(model))

    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)

    if torch.cuda.device_count() > 1:
        logging.info("Use %d %s", torch.cuda.device_count(), "GPUs !")
        model = nn.DataParallel(model)
        # model = torch.nn.parallel.DistributedDataParallel(model)

    model = model.cuda()

    train_criterion = CrossEntropyLabelSmooth(args.classes, args.label_smooth).cuda()
    eval_criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.l2_reg,
    )
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_period, args.gamma, epoch)  # cosine, warm-up
    return train_queue, valid_queue, model, train_criterion, eval_criterion, optimizer, scheduler


def main(args):
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

    solution = individual([64, [1, 0, 6, 4, 6], [2, 1, 2, 2, 4, 1], [6, 1, 6, 1, 7, 2], [2, 6]])

    _, model_state_dict, epoch, step, optimizer_state_dict, best_acc_top1 = utils.load(args.resume)

    train_queue, valid_queue, model, train_criterion, eval_criterion, optimizer, scheduler = \
        build_imagenet(model_state_dict, optimizer_state_dict, epoch=epoch - 1, solution=solution)

    print('Model: {0}, params: {1} M'.format('25-4', count_parameters_in_MB(model)))
    logging.info('Model: {0}, params: {1} M'.format('25-4', count_parameters_in_MB(model)))

    # epoch = 0
    # best_acc_top1= 0
    since_time = time.time()

    while epoch < args.epochs:
        logging.info('epoch %d lr %e', epoch + 1, scheduler.get_last_lr()[0])
        print('epoch:{}, lr:{}, '.format(epoch + 1, scheduler.get_last_lr()[0]))

        train_acc, top5_avg, train_obj = train(train_queue, model, train_criterion, optimizer, args, epoch, since_time)
        logging.info('train_accuracy: %f, top5_avg: %f, loss: %f', train_acc, top5_avg, train_obj)
        print('\n       train_accuracy: {}, top5_avg: {}, loss: {}'.format(train_acc, top5_avg, train_obj))

        valid_acc_top1, valid_acc_top5, valid_obj = valid(valid_queue, model, eval_criterion, args)
        logging.info('valid_accuracy: %f, valid_top5_accuracy: %f', valid_acc_top1, valid_acc_top5)
        print('         valid_accuracy: {}, valid_top5_accuracy: {}'.format(valid_acc_top1, valid_acc_top5))

        epoch += 1
        is_best = False
        if valid_acc_top1 > best_acc_top1:
            best_acc_top1 = valid_acc_top1
            is_best = True

            utils.save(args.save, args, model, epoch, 0, optimizer, best_acc_top1, is_best)

        scheduler.step()  # 只是用来更新学习率的


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training on imagenet')
    # ***************************  common setting******************
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('-save', type=str, default='result_imagenet')
    parser.add_argument('-resume', type=str, default=None, help='The path to the dir you want resume')
    parser.add_argument('-device', type=str, default='cuda')
    # ***************************  DDP setting **********************

    # ***************************  dataset setting******************
    parser.add_argument('-data', type=str, default="/home/ImageNet2012")
    parser.add_argument('-classes', type=int, default=1000)
    parser.add_argument('-autoaugment', action='store_true', default=False)
    parser.add_argument('-load', type=str, default='original', choices=['original', 'lmdb', 'memory'])  # True
    parser.add_argument('-num_workers', type=int, default=16)  # 16
    parser.add_argument('-data_prefetch', action='store_true', default=True,
                        help='Accelerate DataLoader by prefetch_generator')

    # ***************************  optimization setting******************
    parser.add_argument('-batch_size', type=int, default=512)
    parser.add_argument('-eval_batch_size', type=int, default=500)

    parser.add_argument('-epochs', type=int, default=250)
    parser.add_argument('--lr', type=float, default=0.3, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--grad_clip', type=float, default=5.0)
    parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
    parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
    parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
    parser.add_argument('--l2_reg', type=float, default=3e-5)

    # ***************************  structure setting******************
    parser.add_argument('-search_last_channel', type=int, default=1280)
    parser.add_argument('-use_aux_head', action='store_true', default=True)
    parser.add_argument('-auxiliary_weight', type=float, default=0.4)
    parser.add_argument('-keep_prob', type=float, default=1.0)
    args = parser.parse_args()

    # =====================================setting=======================================
    # args.resume = 'result_imagenet/train_2021-09-22-21-42-28/'

    # args.load = 'lmdb'

    # args.lr = 0.4 # for 4 card, batch size = 512
    # args.lr = 0.1 # for 1 card, batch size = 128

    # ============================================================================

    if args.resume is None:
        args.save = '{}/train_{}'.format(args.save, time.strftime("%Y-%m-%d-%H-%M-%S"))
        create__dir(args.save)
    else:
        args.save = args.resume
        print('resume from the dir: {file}'.format(file=args.resume))

    # ===================================  logging  ===================================
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(filename='{}/logs.log'.format(args.save),
                        level=logging.INFO, format=log_format, datefmt='%Y-%m-%d %I:%M:%S %p')

    if args.resume is None:
        logging.info("[Experiments Setting]\n" + "".join(
            ["[{0}]: {1}\n".format(name, value) for name, value in args.__dict__.items()]))
    else:
        logging.info('resume from the dir: {file}'.format(file=args.resume))

    main(args)
