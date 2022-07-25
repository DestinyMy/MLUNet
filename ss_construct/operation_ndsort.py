import os
import numpy as np
import matplotlib.pyplot as plt


def dominate(solution1, solution2):
    """
    define the dominate relationship
    :param solution1:
    :param solution2:
    :return: solution1 dominate solution2, return True, else False
    """
    value1 = solution1[1:3]
    value2 = solution2[1:3]
    flag1 = True
    flag2 = False  # if there is one objective that solution1's value bigger than solution2's
    for i in range(len(value1)):
        if value1[i] <= value2[i]:
            if value1[i] < value2[i]:
                flag2 = True
        else:
            flag1 = False
    if flag1 and flag2:
        return True
    return False


def fast_nd_sort(polulation):
    """
    NSGA-II: fast nd
    :param data
    :return: the whole fronts
    """
    fronts = []  # to store all fronts
    front_tmp = []  # store the members of next front
    Sp = []  # store all sp
    for p in polulation:
        sp = []  # sp save individuals dominated by p
        n = 0  # n count the number of individuals dominating p
        for q in polulation:
            if dominate(p, q):  # if p dominate q, add q to the set of solutions dominated by p
                sp.append(q.tolist())
            elif dominate(q, p):  # if q dominate p, np+1
                n += 1
        Sp.append(sp)
        p[3] = n  # np
        if n == 0:
            p[4] = 1  # rank
            front_tmp.append(p.tolist())
    fronts.append(front_tmp)
    for i, f in enumerate(fronts):
        front_tmp = []
        for p in f:
            for q in Sp[int(p[0])]:
                polulation[int(q[0])][3] -= 1  # update population np
                if polulation[int(q[0])][3] == 0:
                    polulation[int(q[0])][4] = i + 2
                    q[4] = i+2
                    front_tmp.append(q)
        if len(front_tmp) == 0:
            break
        fronts.append(front_tmp)
    return fronts


def crowding_distance_assignment(population, Set_I):
    l = len(Set_I)  # length of this front
    Set_I = np.array(Set_I)
    for i in range(2):
        Set_I = Set_I[Set_I[:, i+1].argsort()]  # sort using objective value
        fmax = Set_I[-1][i+1]
        fmin = Set_I[0][i+1]
        population[int(Set_I[0, 0])][-1] = 4444444444444444
        population[int(Set_I[l - 1, 0])][-1] = 4444444444444444
        Set_I[0][-1] = 4444444444444444
        Set_I[0][-1] = 4444444444444444
        for j in range(1, l - 1):
            population[int(Set_I[j][0])][-1] = \
            population[int(Set_I[j][0])][-1] + (
                        Set_I[j + 1][i+1] - Set_I[j - 1][i+1]) / (fmax - fmin)
            Set_I[j][-1] = Set_I[j][-1] + (Set_I[j + 1][i+1] - Set_I[j - 1][i+1]) / (fmax - fmin)


def get_data(root, device):
    """

    Returns:
        file data, np.array
    """
    data = []
    with open(os.path.join(root, 'operation_test-'+device+'.log')) as f:
        i = 0
        while True:
            line = f.readline()
            if line:
                l = line.split('\t')
                # id|value1|value2|np|rank|crowd distance
                data.append([i, 100-float(l[1].split(' ')[-1]), float(l[2].split(' ')[-2]), 0, 0, 0])
                i += 1
            else:
                break
    f.close()
    return np.array(data)


if __name__ == '__main__':
    root = os.getcwd()
    device = 'mfus_c1'
    data = get_data(root, device)
    # for da in data:
    #     print(da[0],'\t',da[1],'\t',da[2],'\t',da[3],'\t',da[4],'\t',da[5])

    fronts = fast_nd_sort(data)
    # for i, front in enumerate(fronts):
    #     print('front', i+1)
    #     for indiv in front:
    #         print(indiv[0],'\t',indiv[1],'\t',indiv[2],'\t',indiv[3],'\t',indiv[4],'\t',indiv[5])
    #     print('')

    for front in fronts:
        crowding_distance_assignment(data, front)
    with open(os.path.join(root, 'operation_test-'+device+'-ndsort.log'), 'w+') as f:
        for i, front in enumerate(fronts):
            f.write('front'+ str(i+1) + '\n')
            for indiv in front:
                line = str(indiv[0])+'\t'+str(indiv[1])+'\t'+str(indiv[2])+'\t'+\
                       str(indiv[3])+'\t'+str(indiv[4])+'\t'+str(indiv[5])
                f.write(line)
                f.write('\n')
            f.write('\n\n')
    f.close()
