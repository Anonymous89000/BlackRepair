import os
import time

import numpy as np
from zoopt import Dimension, ValueType, Dimension2, Objective, Parameter, Opt, ExpOpt
import util.get_fit_data
import  util.fit_optimize
import  util.data_process
import util.quadratic_fitting
import torch


def ackley(solution):
    x = solution.get_x()
    bias = 0.2
    value = -20 * np.exp(-0.2 * np.sqrt(sum([(i - bias) * (i - bias) for i in x]) / len(x))) - \
            np.exp(sum([np.cos(2.0*np.pi*(i-bias)) for i in x]) / len(x)) + 20.0 + np.e
    return value


def FairnessObj(solution):
    x = solution.get_x()
    #根据我们验证性质的计算方法 将其提供给Racos
    #也就是在这里用到了vgg的模型以及参数 其他地方用不到 不会耦合
    #遇到不得不修改的情况 再去修改Racos的源代码

    params_index=util.get_fit_data.params_index_c
    model = torch.load("data/census.pt")

    count=0
    for i in params_index:
        layer_index, neuron_index = i
        neuron_index -= 1
        key = f'fc{layer_index + 1}.weight'
        tmp_matrix = model[key].T
        for j in range(0, len(tmp_matrix[neuron_index])):
            tmp_matrix[neuron_index][j] = x[count]
            count += 1
        model[key] = tmp_matrix.T

    fairness=util.get_fit_data.cal_fairness1(model.copy())
    return fairness

def ComputeAcc(solution):
    netname = 'racos'
    optimized_net = 'result/' + netname + '_opt.pt'

    model = torch.load("data/census.pt")

    x=solution.get_x()
    count = 0
    params_index = util.get_fit_data.params_index_c
    for i in params_index:
        layer_index, neuron_index = i
        neuron_index -= 1
        key = f'fc{layer_index + 1}.weight'
        tmp_matrix = model[key].T
        for j in range(0, len(tmp_matrix[neuron_index])):
            tmp_matrix[neuron_index][j] += x[count]
            #tmp_matrix[neuron_index][j] = x[count]
            count += 1
        model[key] = tmp_matrix.T

    optimized_acc = util.data_process.recal_acc1(model)
    return optimized_acc

def CombineObj(solution):
    alpha=0.5
    beta=1-alpha
    Fairness_coe=0.076
    Acc_coe=0.84
    value=alpha*(FairnessObj(solution)-Fairness_coe)+beta*(Acc_coe-ComputeAcc(solution))
    return value


def repairFairness(arg):
    #为racos需要准备的东西 目标函数以及可行域
    start=time.time()
    dim_size = 24  # dimension size
    dim = Dimension(dim_size, [[-1, 1]]*dim_size, [True]*dim_size)
    # dim = Dimension2([(ValueType.CONTINUOUS, [-1, 1], 1e-6)]*dim_size)
    obj = Objective(CombineObj, dim)
    # perform optimization
    #solution = Opt.min(obj, Parameter(algorithm='racos',budget=10*dim_size,parallel=True,server_num=6))
    solution = Opt.min(obj, Parameter(algorithm='racos', budget=10 * dim_size))
    # print the solution
    print(solution.get_x(), solution.get_value())
    print(f"optimized net - fairness:{FairnessObj(solution):>8f}  acc:{ComputeAcc(solution):>8f}")
    solution.set_x([0]*24)
    print(solution.get_x())
    print(f"raw net - fairness:{FairnessObj(solution):>8f}  acc:{ComputeAcc(solution):>8f}")
    end=time.time()
    print(end-start)
    return

if __name__=='__main__':
    repairFairness(None)
    pass