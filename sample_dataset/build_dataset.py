import argparse
import torch
import os, random
import numpy as np
from utils import count_parameters_in_MB, Calculate_flops
from Node import NetworkCIFAR
from Search_space import Operations_len


parser = argparse.ArgumentParser(description='predictor configurations')
parser.add_argument('-seed', type=int, default=1000)

parser.add_argument('-search_last_channel', type=int, default=512)

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


def build_dataset(args):
    def get_architecture_info_case(arch):
        init_channel = arch[0]
        stages = arch[1:4]
        pools = arch[4]
        stages[1].insert(0, pools[0])
        stages[2].insert(0, pools[1])

        stages_tmp = stages[0].copy()
        stages_tmp.extend(stages[1].copy())
        stages_tmp.extend(stages[2].copy())

        d0 = stages_tmp.count(0)
        d1 = stages_tmp.count(1)
        d2 = stages_tmp.count(2)
        d3 = stages_tmp.count(3)
        d4 = stages_tmp.count(4)
        d5 = stages_tmp.count(5)
        d6 = stages_tmp.count(6)
        d7 = stages_tmp.count(7)

        model = NetworkCIFAR(args, 10, init_channel, stages, pools, False, 0.6).cuda()
        Flops = Calculate_flops(model)
        num_parameters = count_parameters_in_MB(model)

        return [arch[0],
                len(arch[1])+len(arch[2])+len(arch[3]),
                d0 ,d1, d2, d3, d4, d5 ,d6, d7,
                Flops,
                num_parameters]


    # sampled architectures
    solutions = []
    with open(os.path.join(os.getcwd(), 'sample_arch.txt'), 'r') as f:
        while True:
            line = f.readline()
            if line:
                solutions.append(eval(line))
            else:
                break

    # time
    times = []
    f1 = open(os.path.join(os.getcwd(), 'sample_time-1.txt'), 'r')
    f4 = open(os.path.join(os.getcwd(), 'sample_time-4.txt'), 'r')
    f8 = open(os.path.join(os.getcwd(), 'sample_time-8.txt'), 'r')
    f16 = open(os.path.join(os.getcwd(), 'sample_time-16.txt'), 'r')
    for _ in range(len(solutions)):
        times.append([float(f.readline().split(' ')[1]) for f in [f1, f4, f8, f16]])
    f1.close()
    f4.close()
    f8.close()
    f16.close()

    dataset = []
    for i in range(len(solutions)):
        arch_info = get_architecture_info_case(solutions[i])
        arch_info1 = arch_info.copy()
        arch_info1.append(1)
        arch_info4 = arch_info.copy()
        arch_info4.append(4)
        arch_info8 = arch_info.copy()
        arch_info8.append(8)
        arch_info16 = arch_info.copy()
        arch_info16.append(16)

        dataset.append([arch_info1, times[i][0]])
        dataset.append([arch_info4, times[i][1]])
        dataset.append([arch_info8, times[i][2]])
        dataset.append([arch_info16, times[i][3]])

    with open(os.path.join(os.getcwd(), 'dataset-no_shuffle.txt'), 'w') as f:
        for item in dataset:
            f.write(str(item)+'\n')

    random.shuffle(dataset)

    with open(os.path.join(os.getcwd(), 'dataset.txt'), 'w') as f:
        for item in dataset:
            f.write(str(item)+'\n')


build_dataset(args)
