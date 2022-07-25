import gc

import torch
import random
import torch.backends.cudnn as cudnn
import numpy as np
import argparse, time, logging, sys, os
from threading import Thread


from EMO_public import P_generator, NDsort, F_distance, F_mating, F_EnvironmentSelect

from model_training import solution_evaluation

from utils import create__dir, Calculate_flops, count_parameters_in_MB
from Node import NetworkCIFAR
from Search_space import Operations_name, Init_Channel

from Build_Dataset import build_search_cifar10

from predictor import build_dataset, MLP


class individual():
    def __init__(self, dec, solution_id):
        self.dec = dec
        self.id = solution_id
        self.init_channel = dec[0]  # 初始通道数
        self.stages = dec[1:4]  # stage操作列表 [[],[],[]]
        self.pools = dec[4]  # 两个pool层 [,]

        self.fitness = np.random.rand(2, )

    def evaluate(self, device_id, train_queue, valid_queue, args):
        if device_id is None:
            device_id = 0

        f = open(os.path.join(args.save, 'train_gpu_' + str(device_id)), 'a+')

        print('Evaluating {}-th solution'.format(self.id), file=sys.stdout)
        print('Evaluating {}-th solution'.format(self.id), file=f,
              flush=True)  # False时，print到f中会先存到内存中，close之后一并放入f中，True则是直接存入f中

        self.fitness = solution_evaluation(device_id, f, self.id, self.init_channel, self.stages, self.pools, train_queue,
                                           valid_queue, args)  # error, 0
        # self.fitness = np.array([0,0])
        # fitness: err, 0

        gc.collect()
        f.close()

    def get_architecture_info(self, arch):
        # [16, [1, 6, 0, 7, 0], [4, 6, 2, 6, 4, 1], [7], [7, 0]]
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

    def infer_time(self, latency_predictor):
        print('infer {}-th solution'.format(self.id))
        logging.info('infer {}-th solution'.format(self.id))
        arch_info = self.get_architecture_info(self.dec)

        arch_info1 = arch_info.copy()
        arch_info1.append(1)
        arch_info4 = arch_info.copy()
        arch_info4.append(4)
        arch_info8 = arch_info.copy()
        arch_info8.append(8)
        arch_info16 = arch_info.copy()
        arch_info16.append(16)

        latency1 = latency_predictor.predict(np.array(arch_info1))[0]
        latency4 = latency_predictor.predict(np.array(arch_info4))[0]
        latency8 = latency_predictor.predict(np.array(arch_info8))[0]
        latency16 = latency_predictor.predict(np.array(arch_info16))[0]

        self.fitness[1] = min(latency1, latency4, latency8, latency16)

        print('latency: {:.6f} ms  (ps. latency1:{:.6f} ms, latency4:{:.6f} ms, latency8:{:.6f} ms, latency16:{:.6f} ms)'.format(
            self.fitness[1], latency1, latency4, latency8, latency16))
        logging.info('latency: {:.6f} ms  (ps. latency1:{:.6f} ms, latency4:{:.6f} ms, latency8:{:.6f} ms, latency16:{:.6f} ms)'.format(
            self.fitness[1], latency1, latency4, latency8, latency16))


class MyThread(Thread):
    def __init__(self, func, args):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except:
            return None


class EMO():
    def __init__(self, args, visualization=False):
        self.args = args
        self.popsize = args.popsize
        self.Max_Gen = args.Max_Gen
        self.Gen = 0
        self.save_dir = args.save

        self.init_channel = Init_Channel
        self.node_num = args.node_num_range
        self.op_nums = len(Operations_name)

        self.visualization = visualization

        self.Population = []
        self.Pop_fitness = []
        self.fitness_best = 0

        self.offspring = []
        self.off_fitness = []

        self.tour_index = []
        self.FrontValue = []
        self.CrowdDistance = []
        self.select_index = []

        self.build_dataset()

        self.threshold = 0.06 # 0.08

        # predictor
        inputs, targets = build_dataset()
        self.lat_predictor = MLP()
        self.lat_predictor.fit(inputs, targets)

    def build_dataset(self):
        train_queue, valid_queue = build_search_cifar10(args=self.args, ratio=0.9,
                                                                         num_workers=self.args.search_num_work)
        self.train_queue = train_queue
        self.valid_queue = valid_queue

    def initialization(self):
        for i in range(self.popsize):
            init_c = np.random.choice(self.init_channel)  # 初始通道数
            node_list = np.random.randint(self.node_num[0], self.node_num[1] + 1, 3)  # 三阶段，每阶段操作数量
            list_individual = [init_c]  # init_channel, [stage1], [stage2], [stage3], [pool]

            rate = 0.3
            for j, node in enumerate(node_list):  # 三阶段每阶段的操作
                ops = np.random.randint(0, self.op_nums, node)
                if j < 1:
                    op_c = np.random.randint(0, 4, node)
                    indicator = np.random.rand(node, ) < 1 - rate
                elif j > 1:
                    op_c = np.random.randint(4, 8, node)
                    indicator = np.random.rand(node, ) < rate
                else:
                    op_c = np.random.randint(4, 8, node)
                    indicator = np.random.rand(node, ) < rate
                    ops[indicator] = op_c[indicator]
                    op_c = np.random.randint(0, 4, node)
                    indicator = np.random.rand(node, ) < 1 - rate
                ops[indicator] = op_c[indicator]

                rate += 0.2
                list_individual.append(ops.tolist())

            list_individual.append(np.random.randint(0, self.op_nums, 2).tolist())  # 两个pool选取的操作，stride=2
            self.Population.append(individual(list_individual, i))

        self.Pop_fitness = self.Evaluation(self.Population)

        self.fitness_best = np.min(self.Pop_fitness[:, 0])
        self.save('initial')

    def save(self, path=None):
        if path is None:
            path = 'Gene_{}'.format(self.Gen + 1)
        whole_path = '{}/{}/'.format(self.save_dir, path)
        create__dir(whole_path)

        fitness_file = whole_path + 'fitness.txt'
        np.savetxt(fitness_file, self.Pop_fitness, delimiter=' ')

        Pop_file = whole_path + 'Population.txt'
        with open(Pop_file, "w") as file:
            for j, solution in enumerate(self.Population):
                file.write('solution {}: {} \n'.format(j, solution.dec))

        # best_index = np.argmin(self.Pop_fitness[:, 0])
        # solution = self.Population[best_index]
        # Plot_network(solution.dag[0], '{}/{}_conv_dag.png'.format(whole_path, best_index))
        # Plot_network(solution.dag[1], '{}/{}_reduc_dag.png'.format(whole_path, best_index))

    def Evaluation(self, Pop):
        if self.args.para_evaluation and self.args.num_gpus > 1:
            fitness = []

            for i in range(0, len(Pop), self.args.num_gpus):
                logging.info(
                    'solution {0} --- solution {1} (Parallel Evaluation)'.format(i, (i + self.args.num_gpus - 1) if (i + self.args.num_gpus - 1) < len(Pop) else len(Pop)-1))

                solution_set = Pop[i:(i + self.args.num_gpus) if (i + self.args.num_gpus)<len(Pop) else len(Pop)]
                self.Para_Evaluation(solution_set)

                fitness = [x.fitness for x in Pop]
            fitness = np.array(fitness)
            for i, solution in enumerate(Pop):
                solution.infer_time(self.lat_predictor)
                fitness[i] = solution.fitness
        else:
            fitness = np.zeros((len(Pop), 2))
            for i, solution in enumerate(Pop):
                logging.info('solution: {0:>2d}'.format(i + 1))
                print('solution: {0:>2d}'.format(i + 1))
                solution.evaluate(None, self.train_queue, self.valid_queue, self.args)
                solution.infer_time(self.lat_predictor)
                fitness[i] = solution.fitness
        return fitness  # validate error, inference time

    def Para_Evaluation(self, solution_set):
        thread = [MyThread(solution.evaluate, (id, self.train_queue, self.valid_queue, self.args,)) for id, solution in
                  enumerate(solution_set)]

        for t in thread:
            t.start()
            time.sleep(3)

        _ = [t.join() for t in thread]
        del thread
        gc.collect()

    # def evaluation(self, Pop):
    #     fitness = np.zeros((len(Pop), 5))
    #     for i, solution in enumerate(Pop):
    #         logging.info('solution: {0:>2d}'.format(i + 1))
    #         print('solution: {0:>2d}'.format(i + 1))
    #         solution.evaluate(self.train_queue, self.valid_queue, self.inference_queue, self.args)
    #         fitness[i] = solution.fitness
    #     return fitness[:, (0, 4)]  # validate error, num params, Flops, inference time, 0 1 2 4

    def Binary_Envirmental_tour_selection(self):
        self.MatingPool, self.tour_index = F_mating.F_mating(self.Population.copy(), self.FrontValue,
                                                             self.CrowdDistance)

    def genetic_operation(self):
        offspring_dec = P_generator.P_generator(self.MatingPool, self.popsize, self.op_nums)
        offspring_dec = self.deduplication(offspring_dec)  # 检查重复个体
        self.offspring = [individual(o, i) for i, o in enumerate(offspring_dec)]
        self.off_fitness = self.Evaluation(self.offspring)

    def first_selection(self):
        Population = []
        Population.extend(self.Population)
        Population.extend(self.offspring)

        Population_temp = []
        threshold = self.threshold
        while True:
            for i, solution in enumerate(Population):
                if (solution.fitness[0] < self.fitness_best + threshold) and \
                        solution not in Population_temp:  # 错误率阈值预选择
                    Population_temp.append(solution)
            if len(Population_temp) < self.popsize:
                threshold += 0.01
            else:
                break
        FunctionValue = np.zeros((len(Population_temp), 2))
        for i, solution in enumerate(Population_temp):
            FunctionValue[i] = [solution.fitness[0], solution.fitness[1]]

        return Population_temp, FunctionValue

    def Envirment_Selection(self):
        Population, FunctionValue = self.first_selection()

        Population, FunctionValue, FrontValue, CrowdDistance, select_index = F_EnvironmentSelect. \
            F_EnvironmentSelect(Population, FunctionValue, self.popsize)

        self.Population = Population
        self.Pop_fitness = FunctionValue
        self.FrontValue = FrontValue
        self.CrowdDistance = CrowdDistance
        self.select_index = select_index

        self.fitness_best = np.min(self.Pop_fitness[:, 0])

    #  检查重复个体
    def deduplication(self, offspring_dec):
        pop_dec = [i.dec for i in self.Population]
        dedup_offspring_dec = []
        for i in offspring_dec:
            if i not in dedup_offspring_dec and i not in pop_dec:
                dedup_offspring_dec.append(i)
        return dedup_offspring_dec

    def print_logs(self, since_time=None, initial=False):
        if initial:

            logging.info(
                '********************************************************************Initializing**********************************************')
            print(
                '********************************************************************Initializing**********************************************')
        else:
            used_time = (time.time() - since_time) / 60

            logging.info(
                '*******************************************************{0:>2d}/{1:>2d} processing, time spent so far:{2:.2f} min******'
                '*****************************************'.format(self.Gen + 1, self.Max_Gen, used_time))

            print(
                '*******************************************************{0:>2d}/{1:>2d} processing, time spent so far:{2:.2f} min******'
                '*****************************************'.format(self.Gen + 1, self.Max_Gen, used_time))

    # def plot_fitness(self):
    #     if self.visualization:
    #         plt.clf()
    #         plt.scatter(self.Pop_fitness[:, 0], self.Pop_fitness[:, 1])
    #         plt.xlabel('Error')
    #         plt.ylabel('parameters: MB')
    #         plt.pause(0.001)

    def Main_loop(self):
        since_time = time.time()
        # plt.ion()

        self.print_logs(initial=True)
        self.initialization()
        # self.plot_fitness()

        # self.Pop_fitness维度是种群大小*2，这个2表示错误率、推理延迟
        self.FrontValue = NDsort.NDSort(self.Pop_fitness, self.popsize)[0]
        self.CrowdDistance = F_distance.F_distance(self.Pop_fitness, self.FrontValue)

        while self.Gen < self.Max_Gen:
            self.print_logs(since_time=since_time)

            self.Binary_Envirmental_tour_selection()
            self.genetic_operation()
            self.Envirment_Selection()

            # self.plot_fitness()
            self.save()
            self.Gen += 1
        # plt.ioff()
        # plt.savefig("{}/final.png".format(self.save_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test argument')

    # ***************************  common setting ***************
    parser.add_argument('-seed', type=int, default=1000)
    parser.add_argument('-device', type=str, default='cuda')
    parser.add_argument('-para_evaluation', type=bool, default=True,
                        help='use more gpus to parallel evaluate solutions')
    parser.add_argument('-num_gpus', type=int, default=3, help='the number of gpus for parallel evaluation')
    parser.add_argument('-save', type=str, default='result')

    # ***************************  EMO setting ******************
    parser.add_argument('-popsize', type=int, default=20)  # 20
    parser.add_argument('-Max_Gen', type=int, default=25)  # 25
    parser.add_argument('-node_num_range', type=list, default=[1, 10])  # 3, 6

    # ***************************  dataset setting ******************
    parser.add_argument('-data', type=str, default="/home/**/projects/data")
    parser.add_argument('-search_cutout_size', type=int, default=16)
    parser.add_argument('-search_autoaugment', default=False)
    parser.add_argument('-search_num_work', type=int, default=0,
                        help='the number of the data worker. 0 for parallel search training')

    # ***************************  optimization setting ******************
    parser.add_argument('-search_epochs', type=int, default=25)  # 25
    parser.add_argument('-search_lr_max', type=float, default=0.1)  # 0.025 NAO
    parser.add_argument('-search_lr_min', type=float, default=0.001)  # 0 for final training
    parser.add_argument('-search_momentum', type=float, default=0.9)
    parser.add_argument('-search_l2_reg', type=float, default=3e-4)  # 5e-4 for final training
    parser.add_argument('-search_grad_bound', type=float, default=5.0)
    parser.add_argument('-search_train_batch_size', type=int, default=128)
    parser.add_argument('-search_eval_batch_size', type=int, default=500)

    # ***************************  structure setting ******************
    parser.add_argument('-search_last_channel', type=int, default=512)
    parser.add_argument('-search_use_aux_head', default=True)
    parser.add_argument('-search_auxiliary_weight', type=float, default=0.4)
    parser.add_argument('-search_keep_prob', type=float, default=0.6)  # 0.6 also for final training

    args = parser.parse_args()
    args.save = '{}/EMO_search_{}'.format(args.save, time.strftime("%Y-%m-%d-%H-%M-%S"))
    args.num_gpus = torch.cuda.device_count()

    create__dir(args.save)

    # ===================================  logging  ===================================
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(filename='{}/logs.log'.format(args.save),
                        level=logging.INFO, format=log_format, datefmt='%Y-%m-%d %I:%M:%S %p')

    logging.info("[Experiments Setting]\n" + "".join(
        ["[{0}]: {1}\n".format(name, value) for name, value in args.__dict__.items()]))

    # ===================================  random seed setting  ===================================
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

    EMO_NAS = EMO(args, visualization=True)
    EMO_NAS.Main_loop()
