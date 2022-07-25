import numpy as np

#Programmed By Yang Shang shang  Email：yangshang0308@gmail.com  GitHub: https://github.com/DevilYangS/codes
def sortrows(Matrix, order = "ascend"):
    # 默认先以第一列值从小到大进行排序，若第一列值相同，则按照第二列，以此类推,返回排序结果及对应索引
    # （Reason: list.sort() 仅仅返回排序后的结果，np.argsort()需要多次排序，
    # 其中np.lexsort()的操作对象等同于sortcols ，先排以最后一行对列进行排序，然后以倒数第二列，以此类推. np.lexsort((d,c,b,a)来对[a,b,c,d]进行排序、其中a为一列向量）
    Matrix_temp = Matrix[:, ::-1]  # 因为np.lexsort()默认从最后一行对列开始排序，需要将matrix反向并转置
    Matrix_row = Matrix_temp.T
    if order == "ascend":
        rank = np.lexsort(Matrix_row)
    elif order == "descend":
        rank = np.lexsort(-Matrix_row)
    Sorted_Matrix = Matrix[rank,:]  # Matrix[rank] 也可以
    return Sorted_Matrix, rank
