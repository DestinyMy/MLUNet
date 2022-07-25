import numpy as np
from Search_space import Init_Channel, Operations_len


def P_generator(MatingPool, MaxOffspring, op_nums):
    N = len(MatingPool)
    if MaxOffspring < 1 or MaxOffspring > N:
        MaxOffspring = N

    Offspring = []
    cross_ratio = 0.2  # 0.2

    for i in range(0, N, 2):  # 每两个个体作为父代
        P1 = MatingPool[i].dec.copy()
        P2 = MatingPool[i + 1].dec.copy()

        cross_flag = np.random.rand(1) < cross_ratio

        for j in range(1, 4):  # 循环三个stage
            p1 = np.array(P1[j]).copy()
            p2 = np.array(P2[j]).copy()

            # ----------------------------crossover-------------------------------
            L1, L2 = len(p1), len(p2)
            L_flag = L1 > L2
            common_L = L2 if L_flag else L1
            cross_L = np.random.choice(common_L)

            if cross_flag:
                if not L_flag:
                    p1 = np.append(p1, p2[common_L:], axis=0)
                    p2[:cross_L] = p1[:cross_L]
                else:
                    p2 = np.append(p2, p1[common_L:], axis=0)
                    p1[:cross_L] = p2[:cross_L]

            muta_indicator_1, muta_indicator_2 = mutation_indicator(p1.copy(), p2.copy())

            muta_p1 = mutation(p1.copy(), op_nums)
            muta_p2 = mutation(p2.copy(), op_nums)

            p1[muta_indicator_1] = muta_p1[muta_indicator_1]
            p2[muta_indicator_2] = muta_p2[muta_indicator_2]

            P1[j] = list(p1.copy())
            P2[j] = list(p2.copy())

        # mutate channel
        temp = P1[0].copy()
        if np.random.rand() < cross_ratio:
            P1[0] = mutation([temp], Init_Channel)[0]
        temp = P2[0].copy()
        if np.random.rand() < cross_ratio:
            P2[0] = mutation([temp], Init_Channel)[0]

        # mutate pool
        tmp = np.array(P1.copy()[4])
        candidate = mutation(tmp, op_nums)
        indicator = np.random.rand(2, ) < cross_ratio
        tmp[indicator] = candidate[indicator]
        P1[4] = list(tmp.copy())

        tmp = np.array(P2.copy()[4])
        candidate = mutation(tmp, op_nums)
        indicator = np.random.rand(2, ) < cross_ratio
        tmp[indicator] = candidate[indicator]
        P2[4] = list(tmp.copy())

        # ----------------------------crossover between cell-------------------------------
        if not cross_flag:
            index = np.random.randint(1, 4)
            temp_p1 = P1[index].copy()
            P1[index] = P2[index]
            P2[index] = temp_p1

        Offspring.append(P1)
        Offspring.append(P2)
    return Offspring[:MaxOffspring]


def mutation_indicator(solution_1_stage, solution_2_stage):
    s1_len = len(solution_1_stage)
    s2_len = len(solution_2_stage)

    s1_depth = np.sum(Operations_len[np.array(solution_1_stage)])
    s2_depth = np.sum(Operations_len[np.array(solution_2_stage)])

    mutate_indicator_1 = np.random.rand(s1_len, ) < (1.0 - s1_len * 1.0 / (s1_depth + 1))
    mutate_indicator_2 = np.random.rand(s2_len, ) < (1.0 - s2_len * 1.0 / (s2_depth + 1))

    return mutate_indicator_1, mutate_indicator_2


def mutation(solution, op_nums):
    op_candidate = []
    for i in range(len(solution)):
        a = np.random.choice(op_nums)
        while a == solution[i]:
            a = np.random.choice(op_nums)
        op_candidate.append(a)
    return np.array(op_candidate)
