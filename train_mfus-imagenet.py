import argparse, logging, collections
import codecs
import random, time, sys
import numpy as np
import torch, os
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch_mlu
import torch_mlu.core.mlu_model as ct
import torch_mlu.core.mlu_quantize as mlu_quantize

from utils import create__dir, count_parameters_in_MB, Calculate_flops
from Node import NetworkImageNet


class individual():
    def __init__(self, dec):
        self.dec = dec
        self.init_channel = dec[0]  # 初始通道数
        self.stages = dec[1:4]  # stage操作列表 [[],[],[]]
        self.pools = dec[4]  # 两个pool层 [,]

def build_imagenet(**kwargs):
    solution = kwargs.pop('solution')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    valid_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    validdir = os.path.join(args.data, 'val')
    valid_data = dset.ImageFolder(validdir, valid_transform)

    infer_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=1, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    model = NetworkImageNet(args, args.classes, solution.init_channel, solution.stages, solution.pools,
                            args.use_aux_head, args.keep_prob)
    return infer_queue, model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train test')
    # ***************************  common setting******************
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--core', type=int, default=1)
    # ***************************  dataset setting******************
    parser.add_argument('-data', type=str, default="/home/Cambricon-Test/ImageNet")
    # parser.add_argument('-data', type=str, default="/home/data/ImageNet")
    parser.add_argument('-classes', type=int, default=1000)
    parser.add_argument('-autoaugment', action='store_true', default=False)  # True
    parser.add_argument('-num_workers', type=int, default=16)  # 16
    parser.add_argument('-normal', type=bool, default=True, help='indicate the down-sample type')

    # ***************************  structure setting******************
    parser.add_argument('-search_last_channel', type=int, default=1280)
    parser.add_argument('-use_aux_head', action='store_true', default=False)
    parser.add_argument('-auxiliary_weight', type=float, default=0.4)
    parser.add_argument('-keep_prob', type=float, default=0.6)
    args = parser.parse_args()

    solutions = [[64, [1, 5], [0, 7, 7, 5, 5, 6, 3], [0, 4, 5, 6, 5, 6], [0, 0]],
                 [64, [1, 0, 6, 4, 6], [2, 1, 2, 2, 4, 1], [6, 1, 6, 1, 7, 2], [2, 6]],
                 [40, [5, 0], [5, 5, 3, 7, 2, 2], [4, 4, 4, 5, 6], [5, 4]],
                 [32, [0, 1, 2], [7, 7, 4, 4, 5, 5], [7, 2, 4, 1, 4, 2], [7, 7]],
                 [32, [4, 0, 7], [3, 1, 7, 4, 7], [1, 3, 2, 4], [3, 1]],
                 [32, [4, 5], [1, 1, 1, 4, 1, 7, 2], [1, 1, 2, 1, 1, 3], [1, 1]],
                 [32, [7, 0], [0, 4, 2, 0, 6, 5], [7, 2, 0, 0, 0, 1], [0, 7]],
                 [32, [3, 4], [1, 2, 2, 6, 1], [1, 1, 1, 0], [1, 1]],
                 [16, [4, 5], [0, 1, 1, 1, 4, 7], [4, 5, 0, 2, 1], [0, 4]],
                 [16, [3, 0], [0, 0, 2, 1, 0, 5, 3], [6, 5, 2, 0, 1], [0, 6]],
                 [64, [5, 4, 6, 0, 2], [2, 7, 1, 1, 5, 7, 2], [1, 7, 3, 4, 7, 2, 4], [2, 1]],
                 [64, [5, 4, 6, 1, 6], [2, 1, 1, 4, 7, 2], [1, 2, 7, 0, 3, 7, 2, 6], [2, 1]],
                 [64, [2, 4, 3, 7, 1], [2, 1, 2, 5, 4, 7, 7], [3, 7, 6, 6, 0, 6, 6, 2, 6], [2, 3]],
                 [64, [5, 0], [6, 4, 5, 6, 0, 5], [2, 2, 2, 0, 1, 6, 5], [6, 2]],
                 [40, [1, 1, 2, 6, 5], [1, 1, 5, 5, 6, 0], [1, 1, 1, 5, 0, 3], [1, 1]],
                 [32, [3, 1], [6, 1, 7, 7, 5, 3], [1, 3, 1, 3, 5, 4], [6, 1]],
                 [32, [2, 4], [6, 6, 4, 4, 6], [1, 3, 2, 6], [6, 1]],
                 [32, [1, 1], [4, 6, 1, 4, 1, 5], [1, 2, 4], [4, 1]],
                 [32, [3, 3], [6, 5, 0, 0, 4, 2], [1, 1, 1, 0, 3], [6, 1]],
                 [16, [3, 0], [0, 4, 0, 4, 0, 1, 1], [1, 1, 5, 2, 5], [0, 1]]]

    model_name = {0: '25-1', 1: '25-4', 2: '25-7', 3: '25-14', 4: '25-18', 5: '25-10', 6: '25-12', 7: '25-6', 8: '25-2',
                  9: '25-0', 10:'25-14', 11:'25-0', 12:'25-15', 13:'25-8', 14:'25-6', 15:'25-3', 16:'25-11', 17:'25-7', 18:'25-1', 19:'25-9'}


    data = torch.randn(1, 3, 224, 224)
    ct.set_core_number(args.core)
    ct.set_core_version('MLU270')
    with open(os.path.join(os.getcwd(), 'time_imagenet.txt'), 'a+') as f:
        # f.write('2080ti\n')
        f.write('mfus\n')
        f.write('core {}\n'.format(args.core))
        for i, s in enumerate(solutions):
            solution = individual(s)
            infer_queue, model = build_imagenet(epoch=-1, solution=solution)
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
            # for j in range(1):
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
            # print('avg_time: {0} ms'.format(sum / 1))
            # f.write('avg_time: {0} ms\n'.format(sum / 1))

            # mfus
            print('quantize ...')
            model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'test.pth'),  map_location='cpu'))
            model.eval().float()
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            qconfig = {'iteration': 1, 'use_avg': False, 'data_scale': 1.0, 'firstconv': False, 'mean': mean, 'std': std,
                       'per_channel': False}
            model_quantized = torch_mlu.core.mlu_quantize.quantize_dynamic_mlu(model,
                                                                               qconfig_spec=qconfig,
                                                                               dtype='int16',
                                                                               gen_quant=True)
            model_quantized(data)
            torch.save(model_quantized.state_dict(), os.path.join(os.getcwd(), 'test-quantize.pth'))

            print('loading parameters ...')
            quantized_model = torch_mlu.core.mlu_quantize.quantize_dynamic_mlu(model)
            quantized_model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'test-quantize.pth'), map_location='cpu'), False)
            quantized_model.eval().float()
            quantized_model = torch.jit.trace(quantized_model.to(ct.mlu_device()),
                                            data.to(ct.mlu_device()),
                                            check_trace=False)

            print('inference on mfus ...')
            with torch.no_grad():
                for step, (input, _) in enumerate(infer_queue):
                    input = input.to(ct.mlu_device())
                    outputs = quantized_model(input)

            total_infer = len(infer_queue)
            avg_time = 0
            with torch.no_grad():
                for step, (input, _) in enumerate(infer_queue):
                    input = input.to(ct.mlu_device())
                    since = time.time()
                    outputs = quantized_model(input)
                    end = time.time()
                    avg_time += (end - since) / total_infer
            print('avg_time: {0} ms'.format(avg_time * 1000))
            f.write('avg_time: {0} ms\n'.format(avg_time * 1000))
