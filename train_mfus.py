import argparse
import random, time, sys
import torch, os
import torchvision.datasets as dataset
import torch_mlu
import torch_mlu.core.mlu_model as ct
import torch_mlu.core.mlu_quantize as mlu_quantize

from utils import count_parameters_in_MB
from Node import NetworkCIFAR
from Build_Dataset import _data_transforms_cifar10


class individual():
    def __init__(self, dec):
        self.dec = dec
        self.init_channel = dec[0]  # 初始通道数
        self.stages = dec[1:4]  # stage操作列表 [[],[],[]]
        self.pools = dec[4]  # 两个pool层 [,]


def build_train_cifar10(args, cutout_size=None, autoaugment=False):
    # used for training process, so valid_data "train=False"

    train_transform, valid_transform = _data_transforms_cifar10(cutout_size, autoaugment)

    # train_data = dataset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    valid_data = dataset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

    # train_queue = torch.utils.data.DataLoader(
    #     train_data, batch_size=args.train_batch_size, shuffle=True, pin_memory=True, num_workers=16)
    # valid_queue = torch.utils.data.DataLoader(
    #     valid_data, batch_size=args.eval_batch_size, shuffle=False, pin_memory=True, num_workers=16)
    infer_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=1, shuffle=False, pin_memory=True, num_workers=16)

    return infer_queue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train test')
    # ***************************  common setting******************
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--core', type=int, default=1)
    # ***************************  dataset setting******************
    # parser.add_argument('-data', type=str, default="/home/**/projects/data")
    parser.add_argument('-data', type=str, default="/opt/cambricon/data")
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10, cifar100'])
    parser.add_argument('-classes', type=int, default=10)
    parser.add_argument('-autoaugment', default=False)  # True
    parser.add_argument('-cutout_size', type=int, default=None)  # 16
    # ***************************  optimization setting******************
    # ***************************  structure setting******************
    parser.add_argument('-search_last_channel', type=int, default=512)
    parser.add_argument('-use_aux_head', action='store_true', default=False)
    parser.add_argument('-auxiliary_weight', type=float, default=0.4)
    parser.add_argument('-keep_prob', type=float, default=0.6)
    args = parser.parse_args()

    infer_queue = build_train_cifar10(args=args, cutout_size=args.cutout_size,
                                      autoaugment=args.autoaugment)

    solutions = [[64, [1, 5], [0, 7, 7, 5, 5, 6, 3], [0, 4, 5, 6, 5, 6], [0, 0]],
                 [64, [1, 0, 6, 4, 6], [2, 1, 2, 2, 4, 1], [6, 1, 6, 1, 7, 2], [2, 6]],
                 [40, [5, 0], [5, 5, 3, 7, 2, 2], [4, 4, 4, 5, 6], [5, 4]],
                 [32, [0, 1, 2], [7, 7, 4, 4, 5, 5], [7, 2, 4, 1, 4, 2], [7, 7]],
                 [32, [4, 0, 7], [3, 1, 7, 4, 7], [1, 3, 2, 4], [3, 1]],
                 [32, [4, 5], [1, 1, 1, 4, 1, 7, 2], [1, 1, 2, 1, 1, 3], [1, 1]],
                 [32, [7, 0], [0, 4, 2, 0, 6, 5], [7, 2, 0, 0, 0, 1], [0, 7]],
                 [32, [3, 4], [1, 2, 2, 6, 1], [1, 1, 1, 0], [1, 1]],
                 [16, [4, 5], [0, 1, 1, 1, 4, 7], [4, 5, 0, 2, 1], [0, 4]],
                 [16, [3, 0], [0, 0, 2, 1, 0, 5, 3], [6, 5, 2, 0, 1], [0, 6]]]

    model_name = {0:'25-1', 1:'25-4', 2:'25-7', 3:'25-14', 4:'25-18', 5:'25-10', 6:'25-12', 7:'25-6', 8:'25-2', 9:'25-0'}

    data = torch.randn(1, 3, 32, 32)
    ct.set_core_number(args.core)
    ct.set_core_version('MLU270')
    with open(os.path.join(os.getcwd(), 'time.txt'), 'a+') as f:
        # f.write('2080ti\n')
        f.write('mfus\n')
        f.write('core {}\n'.format(args.core))
        for i, s in enumerate(solutions):
            solution = individual(s)
            model = NetworkCIFAR(args, args.classes, solution.init_channel, solution.stages, solution.pools,
                                 args.use_aux_head, args.keep_prob)

            check = model.state_dict()
            torch.save(check, os.path.join(os.getcwd(), 'test.pth'))
            print('Model: {0}, params: {1} M'.format(model_name[i], count_parameters_in_MB(model)))
            f.write('Model: {0}, params: {1} M\n'.format(model_name[i], count_parameters_in_MB(model)))

            # 2080ti
            # model = model.cuda()
            # model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'test.pth')))
            # model.eval().float()
            # total_infer = len(infer_queue)
            # sum = 0
            # for j in range(3):
            #     avg_time = 0
            #     with torch.no_grad():
            #         for step, (input, _) in enumerate(infer_queue):
            #             input = input.cuda()
            #             since = time.time()
            #             outputs = model(input)
            #             end = time.time()
            #             avg_time += (end - since) / total_infer
            #         print('time{0}: {1} ms'.format(j + 1, avg_time * 1000))
            #         f.write('time{0}: {1} ms\n'.format(j + 1, avg_time * 1000))
            #     sum += avg_time * 1000
            # print('avg_time: {0} ms'.format(sum / 3))
            # f.write('avg_time: {0} ms\n'.format(sum / 3))

            # mfus
            print('quantize ...')
            model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'test.pth'), map_location='cpu'))
            model.eval().float()
            mean = [0.49139968, 0.48215827, 0.44653124]
            std = [0.24703233, 0.24348505, 0.26158768]
            qconfig = {'iteration': 1, 'use_avg': False, 'data_scale': 1.0, 'firstconv': False, 'mean': mean,
                       'std': std,
                       'per_channel': False}
            model_quantized = torch_mlu.core.mlu_quantize.quantize_dynamic_mlu(model,
                                                                               qconfig_spec=qconfig,
                                                                               dtype='int16',
                                                                               gen_quant=True)
            model_quantized(data)
            torch.save(model_quantized.state_dict(), os.path.join(os.getcwd(), 'test-quantize.pth'))

            print('loading parameters ...')
            quantized_model = torch_mlu.core.mlu_quantize.quantize_dynamic_mlu(model)
            quantized_model.load_state_dict(
                torch.load(os.path.join(os.getcwd(), 'test-quantize.pth'), map_location='cpu'), False)
            quantized_model.eval().float()
            quantized_model = torch.jit.trace(quantized_model.to(ct.mlu_device()),
                                              data.to(ct.mlu_device()),
                                              check_trace=False)
            print('ok')

            with torch.no_grad():
                for step, (input, _) in enumerate(infer_queue):
                    input = input.to(ct.mlu_device())
                    outputs = quantized_model(input)

            total_infer = len(infer_queue)
            sum = 0
            for j in range(3):
                avg_time = 0
                with torch.no_grad():
                    for step, (input, _) in enumerate(infer_queue):
                        input = input.to(ct.mlu_device())
                        since = time.time()
                        outputs = quantized_model(input)
                        end = time.time()
                        avg_time += (end - since) / total_infer
                    print('time{0}: {1} ms'.format(j + 1, avg_time * 1000))
                    f.write('time{0}: {1} ms\n'.format(j + 1, avg_time * 1000))
                sum += avg_time * 1000
            print('avg_time: {0} ms'.format(sum / 3))
            f.write('avg_time: {0} ms\n'.format(sum / 3))
