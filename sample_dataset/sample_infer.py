import argparse
import random, time, sys
import numpy as np
import torch, os
import torchvision.datasets as dataset
import torch_mlu
import torch_mlu.core.mlu_model as ct
import torch_mlu.core.mlu_quantize as mlu_quantize

from Node import NetworkCIFAR
from Build_Dataset import _data_transforms_cifar10


def build_train_cifar10(args, cutout_size=None):
    _, valid_transform = _data_transforms_cifar10(cutout_size)
    valid_data = dataset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

    num_ = len(valid_data)  # CIFAR10: 10000
    indices = list(range(num_))
    np.random.shuffle(indices)

    warm_queue = torch.utils.data.DataLoader(
        valid_data,
        batch_size=1,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:1000]),
        pin_memory=True,
        num_workers=0)

    inference_queue = torch.utils.data.DataLoader(
        valid_data,
        batch_size=1,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[1000:2000]),
        pin_memory=True,
        num_workers=0)

    return warm_queue, inference_queue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train test')
    # ***************************  common setting******************
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--start', type=int)
    parser.add_argument('--end', type=int)

    # ***************************  dataset setting******************
    parser.add_argument('-data', type=str, default="/opt/cambricon/data")
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10, cifar100'])
    parser.add_argument('-classes', type=int, default=10)

    # ***************************  structure setting******************
    parser.add_argument('-use_aux_head', action='store_true', default=False)
    parser.add_argument('-search_last_channel', type=int, default=512)
    parser.add_argument('-auxiliary_weight', type=float, default=0.4)
    parser.add_argument('-keep_prob', type=float, default=0.6)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    warm_queue, infer_queue = build_train_cifar10(args=args, cutout_size=None)

    solutions = []
    with open(os.path.join(os.getcwd(), 'sample_arch.txt'), 'r') as f:
        while True:
            line = f.readline()
            if line:
                solutions.append(eval(line))
            else:
                break

    f1 = open(os.path.join(os.getcwd(), 'sample_info-8.txt'), "a+")  # record infer info

    f2 = open(os.path.join(os.getcwd(), 'sample_time-8.txt'), "a+")  # record infer time

    data = torch.randn(1, 3, 32, 32)
    ct.set_core_number(8)
    ct.set_core_version('MLU270')

    for i in range(args.start, args.end):
        s = solutions[i]

        model = NetworkCIFAR(args, args.classes, s[0], s[1:4], s[-1], args.use_aux_head, args.keep_prob)

        check = model.state_dict()
        torch.save(check, os.path.join(os.getcwd(), 'test.pth'))

        # 2080ti
        # model = model.cuda()
        # model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'test.pth')))
        # model.eval().float()
        # # warm
        # with torch.no_grad():
        #     for step, (input, _) in enumerate(warm_queue):
        #         input = input.cuda()
        #         outputs = model(input)
        # # infer
        # total_infer = len(infer_queue)
        # sum = 0
        # with torch.no_grad():
        #     for step, (input, _) in enumerate(infer_queue):
        #         input = input.cuda()
        #         since = time.time()
        #         outputs = model(input)
        #         end = time.time()
        #         sum += (end - since) / total_infer
        # times.append(sum)
    # save(times)

        # mfus
        print('quantize ...')
        f1.write('quantize ...\n')
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
        f1.write('loading parameters ...\n')
        quantized_model = torch_mlu.core.mlu_quantize.quantize_dynamic_mlu(model)
        quantized_model.load_state_dict(
            torch.load(os.path.join(os.getcwd(), 'test-quantize.pth'), map_location='cpu'), False)
        quantized_model.eval().float()
        quantized_model = torch.jit.trace(quantized_model.to(ct.mlu_device()),
                                          data.to(ct.mlu_device()),
                                          check_trace=False)

        print('inference ...')
        f1.write('inference ...\n')

        # warm
        with torch.no_grad():
            for step, (input, _) in enumerate(warm_queue):
                input = input.to(ct.mlu_device())
                outputs = quantized_model(input)

        # infer
        total_infer = len(infer_queue)
        sum = 0
        with torch.no_grad():
            for step, (input, _) in enumerate(infer_queue):
                input = input.to(ct.mlu_device())
                since = time.time()
                _ = quantized_model(input)
                end = time.time()
                sum += (end - since) / total_infer

        print('core_{0}-solution_{1}: {2} ms ({3})\n'.format(8, i + 1, sum * 1000, s))
        f1.write('core_{0}-solution_{1}: {2} ms ({3})\n'.format(8, i + 1, sum * 1000, s))
        f2.write('core_{0}-solution_{1}: {2} ms ({3})\n'.format(8, i + 1, sum * 1000, s))
    f1.close()
    f2.close()
