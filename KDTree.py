#coding:utf-8
import numpy as np
from time import *

class KD_node(object):
    #定义的kd树节点
    def __init__(self, point = None, split = None,index = None, LL = None, RR = None):
        #节点值
        self.point = point
        #节点分割维度
        self.split = split
        # 该坐标点在原数据集的下标号
        self.index = index
        #节点左孩子
        self.left = LL
        #节点右孩子
        self.right = RR



'''
建立KD树
'''
def createKDTree(root, data_list):
    #print(type(data_list))
    #start是传入的数据集的第一个元素在原数据集中的下标
    length = len(data_list)#第一次传入的的确是ndarray，但之后递归的时候传入的就是列表了
    if length == 0:
        return
    dimension = len(data_list[0])-1#去掉标号所在的维度
    max_var = 0

    split = 0
    for i in range(dimension):
        ll = []
        for t in data_list:
            ll.append(t[i+1])#取数据的时候也要略过标号那一维
        var = computerVariance(ll)
        if var > max_var:
            max_var = var
            split = i
    #以最大方差的点为维度，进行划分
    data_list = sorted(data_list, key = lambda x : x[split+1])

    #找中位下标
    point = data_list[int(length / 2)][1:]#取后两维的坐标
    index =int(data_list[int(length / 2)][0])#取第一维的id号
    #print(index)
    root = KD_node(point, split, index)
    #递归建立左子树
    root.left = createKDTree(root.left, data_list[0:int(length / 2)])
    #递归建立右子树
    root.right = createKDTree(root.right, data_list[int(length / 2) + 1 : length])
    return root

#计算方差,对方差公式进行转化了，每个数据点的平方均值减去数据均值的平方
def computerVariance(arraylist):
    #arraylist = array(arraylist)
    for i in range(len(arraylist)):
        arraylist[i] = float(arraylist[i])
    length = len(arraylist)
    sum1 = sum(arraylist)
    array2 = [arraylist[i]*arraylist[i] for i in range(length)]
    sum2 = sum(array2)
    mean = sum1 / length
    variance = sum2 / length - mean ** 2
    return variance

'''
基于KD树进行索引
'''
#用于计算欧式距离
def computerDistance(pt1, pt2):
    sum = 0.0
    for i in range(len(pt1)):
        sum = sum + (pt1[i] - pt2[i]) ** 2#注意第一维被标号占用，传入的只有两维，不过循环遍历的控制次数要修改
    return sum ** 0.5
#query中保存着最近k节点，先进行K近邻查询，再从中挑出最优解

def findNN(root, query,k):
    min_dist = computerDistance(query,root.point)
    node_K = []#node_K中和nodeList保持同步，存入计算的距离的结果
    nodeList = []#存放自上而下进行搜索的过程中途径的父节点
    idList = []#存放最优解在原始数据集中的下标
    temp_root = root
    #为了方便，在找到叶子节点同时，把所走过的父节点的距离都保存下来，下一次回溯访问就只需要访问子节点，不需要再访问一遍父节点。
    while temp_root:
        nodeList.append(temp_root)#将当前父节点存入，只要还有子节点，那么在判断的位置上存入的都是父节点
        dd = computerDistance(query,temp_root.point)
        if len(node_K) < k:
            node_K.append(dd)
            idList.append(temp_root.index)
        else :
            max_dist = max(node_K)
            if dd < max_dist:#当在进行KNN查询时，列表中已经存入当前K个距离最小的解，下一个解进入的条件就是它是否小于最大距离
                index = node_K.index(max_dist)#求表中最大值对应的索引
                del(node_K[index])
                del(idList[index])
                node_K.append(dd)
                idList.append(temp_root.index)
        ss = temp_root.split
        #找到最靠近的叶子节点
        #在当前划分轴的那个维度上进行比较，找到下一个进行比较的节点
        if query[ss] <= temp_root.point[ss]:
            temp_root = temp_root.left
        else:
            temp_root = temp_root.right
    print('node_k :',node_K)
    print('idList :',idList)

    #回溯访问父节点
    while nodeList:
        back_point = nodeList.pop()
        ss = back_point.split
        print('父亲节点 : ',back_point.point,'维度 ：',back_point.split,'节点标号 ：',back_point.index)
        max_dist = max(node_K)
        print('该节点到查询节点的距离 ：',computerDistance(back_point.point,query))
        #若满足进入该父节点的另外一个子节点的条件
        #算法的描述是以查询点为中心，以中心点和当前最近点的距离为半径做一个圆，与父节点相交的话，那么该父节点的另一个子节点有可能是更近点
        #代码实现起来就是，以查询点到当前父节点的划分轴的距离是否小于KNN列表中的最大值，最大值是因为K有多个的话，只要能进到表中即可
        if  len(node_K) < k or abs(query[ss] - back_point.point[ss]) < max_dist :
            #进入另外一个子节点
            if query[ss] <= back_point.point[ss]:
                temp_root = back_point.right
            else:
                temp_root = back_point.left
            if temp_root:
                nodeList.append(temp_root)
                curDist = computerDistance(temp_root.point,query)
                print('curDist :',curDist)
                #如果当前点满足入表的条件，但是表已经满了的时候
                if max_dist > curDist and len(node_K) == k:
                    index = node_K.index(max_dist)
                    #把当前表内最大的数据点给移出，再把新的点放入表中
                    del(node_K[index])
                    del(idList[index])
                    node_K.append(curDist)
                    idList.append(temp_root.index)
                elif len(node_K) < k:
                    #如果表没有满，那么这个点无理由进入，因为一定是当前 前K个最近的点
                    node_K.append(curDist)
                    idList.append(temp_root.index)
    return node_K,nodeList,idList

if __name__ == "__main__":
    index = 0
    data = np.loadtxt("BJ/real.txt")[:, -2:]#导入数据
    id_sum = data.shape[0]
    data_id = np.loadtxt("BJ/real.txt")[:, 0].reshape((id_sum, 1))#把数据的节点也导入
    data2 = np.concatenate((data_id, data), axis=1)
    begin_time = time()
    root = createKDTree(None, data2)
    res_list,_,res_id = findNN(root, data2[50][1:], 2)
    end_time = time()
    run_time = end_time - begin_time
    print("最优距离值为： ",res_list)
    print("最优点为： ",res_id)
    print("程序运行时间： ",run_time)

