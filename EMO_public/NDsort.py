import numpy as np
from EMO_public import sortrows


def NDSort(PopObj, Remain_Num):
    # PopObj:
    #           error_rate  num_parameters
    # individual1   xx          xx
    # individual2   xx          xx
    # ...
    # ...
    # size: pop_size * 2
    # Remain_Num: population size
    N, M = PopObj.shape
    FrontNO = np.inf * np.ones((1, N))
    MaxFNO = 0
    PopObj, rank = sortrows.sortrows(PopObj)

    while (np.sum(FrontNO < np.inf) < Remain_Num):
        MaxFNO += 1
        for i in range(N):
            if FrontNO[0, i] == np.inf:
                Dominated = False
                for j in range(i - 1, -1, -1):
                    if FrontNO[0, j] == MaxFNO:
                        m = 2
                        while (m <= M) and (PopObj[i, m - 1] >= PopObj[j, m - 1]):
                            m += 1
                        Dominated = m > M
                        if Dominated or (M == 2):
                            break
                if not Dominated:
                    FrontNO[0, i] = MaxFNO
    # temp=np.loadtxt("solution.txt")
    # print((FrontNO==temp).all())
    front_temp = np.zeros((1, N))
    front_temp[0, rank] = FrontNO  # 相当于直接赋值
    # FrontNO[0, rank] = FrontNO 不能这么操作，因为 FrontNO值 在 发生改变 ，会影响后续的结果

    return front_temp, MaxFNO