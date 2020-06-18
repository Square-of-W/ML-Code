#coding:gbk
import numpy as np
import matplotlib.pyplot as plt
'''
��Ŀ������ ����500�����ݣ�x��y��������x=y+n��nΪ��˹�ֲ���ƽ��ֵΪ0����׼��Ϊdelt����ʹ�����Իع�ѧϰ�㷨������x����y
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
    #��ʼ������
    theta = 2
    cost = 0
    # ���������ֵ
    for i in range(number):
        cost += (x[i] - theta - y[i]) * (x[i] - theta - y[i])
    cost /= 2 * number
    last_cost = 100
    i = 0#��������
    while abs(last_cost-cost)>1e-4:
        last_cost = cost
        cost = 0
        #�����ݶ��½�
        d_cost_theta = np.sum(x-y-theta)*(-1)
        d_cost_theta /= number
        theta -= learning_rate*d_cost_theta
        # ���������ֵ
        n = 0
        for n in range(number):
            cost += (x[n] - theta - y[n]) * (x[n] - theta - y[n])
        cost /= 2 * number
        i += 1
        print("��",i,"�ε�������,","theta=",theta,"cost=",cost)
    print("ѧϰ������һ��ѧϰ��,",i,"�Σ��������Ϊcost=",cost,"����ֱ�߱��ʽΪx=y+",round(theta,3))
    return round(theta,3)



x,y = generate_data(1,500)
n = learning_x_y(500,0.03,x,y)
plt.title("Regression Result")
plt.xlabel("x")
plt.ylabel("y")
x = np.arange(-25,25,0.1)
plt.plot(x,x-n)
plt.show()