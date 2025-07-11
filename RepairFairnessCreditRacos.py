import os
import time

import numpy as np
from zoopt import Dimension, ValueType, Dimension2, Objective, Parameter, Opt, ExpOpt

import census_age.cal_fairness_age
import util.get_fit_data
import  util.fit_optimize
import  util.data_process
import util.quadratic_fitting
import credit_age.cal_fairness_age
import credit_gender.cal_fairness_sex
import torch
import cma

# ---------- 移到外部的类和函数定义 ----------
class FakeSolution:
    """用于包装CMA-ES输出的numpy数组，使其兼容原有接口"""
    def __init__(self, x):
        self.x = x
    def get_x(self):
        return self.x

class BestSolution:
    """保存优化结果的包装类"""
    def __init__(self, x):
        self.x = x
    def get_x(self):
        return self.x

def cma_objective(x):
    """适配函数：将numpy数组转换为FakeSolution对象，再计算目标值"""
    return CombineObj(FakeSolution(x))

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
    #model = torch.load("data/census.pt")
    model = torch.load("credit_best_model.pt")

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

    #fairness = credit_gender.cal_fairness_sex.cal_fairness1_gender(model.copy())
    fairness=credit_age.cal_fairness_age.cal_fairness1_age(model.copy())
    #fairness = 1-util.get_fit_data.cal_IDNNfairness(model.copy())
    return fairness

def ComputeAcc(solution):
    netname = 'racos'
    optimized_net = 'result/' + netname + '_opt.pt'

    #model = torch.load("data/census.pt")
    model=torch.load("credit_best_model.pt")

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

    optimized_acc = credit_age.cal_fairness_age.recal_acc1(model)
    #optimized_acc = util.data_process.recal_acc1(model)
    return optimized_acc

def SaveOptedModel(solution,targetfile):
    netname = 'racos'
    optimized_net = 'result/' + netname + '_opt.pt'

    #model = torch.load("data/census.pt")
    model=torch.load("credit_best_model.pt")

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

    optimized_acc = util.data_process.savecreditmodel(model,targetfile)
    return optimized_acc

def CombineObj(solution):
    alpha=0.5
    beta=1-alpha
    Fairness_coe=0.076
    Acc_coe=0.98
    value=alpha*(FairnessObj(solution)-Fairness_coe)+beta*(Acc_coe-ComputeAcc(solution))
    return value


def repairFairness(arg):
    #为racos需要准备的东西 目标函数以及可行域
    start=time.time()
    util.get_fit_data.init_fairness_cache("data/credit/testx.txt", sensitive_index=7, device="cuda")

    if 0:
        dim_size = 24  # dimension size
        dim = Dimension(dim_size, [[-1, 1]]*dim_size, [True]*dim_size)
        # dim = Dimension2([(ValueType.CONTINUOUS, [-1, 1], 1e-6)]*dim_size)
        obj = Objective(CombineObj, dim)
        # perform optimization
        #solution = Opt.min(obj, Parameter(algorithm='racos',budget=10*dim_size,parallel=True,server_num=6))
        solution = Opt.min(obj, Parameter(algorithm='racos', budget=10 * dim_size))
        # print the solution
        print(solution.get_x(), solution.get_value())
        SaveOptedModel(solution,'credit_gender_racos_opt.pt')
        print(f"optimized net - fairness:{FairnessObj(solution):>8f}  acc:{ComputeAcc(solution):>8f}")
        solution.set_x([0]*24)
        print(solution.get_x())
        print(f"raw net - fairness:{FairnessObj(solution):>8f}  acc:{ComputeAcc(solution):>8f}")
    else:
        dim_size = 24
        x0 = np.zeros(dim_size)  # 初始点
        sigma0 = 0.5  # 初始标准差
        opts = cma.CMAOptions()
        opts.set("bounds", [-1, 1])  # 变量边界
        opts.set("maxfevals", 10 * dim_size)  # 最大评估次数

        # 运行CMA-ES优化
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        es.optimize(cma_objective)

        # 处理结果
        best_x = es.result[0]  # 最优解
        best_solution = BestSolution(best_x)
        SaveOptedModel(best_solution, 'credit_age_cmaes_opt.pt')

        # 打印结果
        print(f"CMA-ES最优值: {es.result[1]:.6f}")
        print(f"优化后模型 - 公平性: {FairnessObj(best_solution):.6f} 准确率: {ComputeAcc(best_solution):.6f}")

        # 对比原始模型（全零解）
        null_solution = BestSolution(np.zeros(dim_size))
        print(f"原始模型 - 公平性: {FairnessObj(null_solution):.6f} 准确率: {ComputeAcc(null_solution):.6f}")

        pass


    end=time.time()

    print(end-start)
    return

if __name__=='__main__':
    repairFairness(None)
    pass