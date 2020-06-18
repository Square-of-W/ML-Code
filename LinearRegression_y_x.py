#coding:gbk
import numpy as np
import matplotlib.pyplot as plt
'''
题目描述： 生成500个数据（x，y），其中x=y+n，n为高斯分布，平均值为0，标准差为delt。请使用线性回归学习算法从输入x估计y
'''
def generate_data(delta,number):
    mu = 0
    sigma = delta
    num = number
    n = np.random.normal(mu,sigma,num)
    #print(n.shape)
    x = np.arange(-25,25,0.1)
    y = x - n
    plt.title("Data set")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(x,y,"ob")
    plt.show()
    return x,y

def learning_x_y(number,learning_rate,x,y):
    #初始化参数
    theta = 2
    cost = 0
    # 计算总误差值
    for i in range(number):
        cost += (x[i] - theta - y[i]) * (x[i] - theta - y[i])
    cost /= 2 * number
    last_cost = 100
    i = 0#迭代次数
    while abs(last_cost-cost)>1e-4:
        last_cost = cost
        cost = 0
        #进行梯度下降
        d_cost_theta = np.sum(x-y-theta)*(-1)
        d_cost_theta /= number
        theta -= learning_rate*d_cost_theta
        # 计算总误差值
        n = 0
        for n in range(number):
            cost += (x[n] - theta - y[n]) * (x[n] - theta - y[n])
        cost /= 2 * number
        i += 1
        print("第",i,"次迭代计算,","theta=",theta,"cost=",cost)
    print("学习结束，一共学习了,",i,"次，最终误差为cost=",cost,"最终直线表达式为x=y+",round(theta,3))
    return round(theta,3)



x,y = generate_data(1,500)
n = learning_x_y(500,0.03,x,y)
plt.title("Regression Result")
plt.xlabel("x")
plt.ylabel("y")
x = np.arange(-25,25,0.1)
plt.plot(x,x-n)
plt.show()