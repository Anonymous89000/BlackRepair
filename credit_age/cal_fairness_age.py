#读取data_on_age的数据，然后放到训练好的网络得到输出，计算公平性
import matplotlib.pyplot as plt  # 绘图
import matplotlib
import time
import datetime
import pandas as pd
import numpy as np
# from lib_models import *
# from utils import *

from pandas.plotting import scatter_matrix #绘制散布矩阵图
# from sklearn.model_selection import StratifiedShuffleSplit #分层交叉验证
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.preprocessing import MinMaxScaler
import torch
import random
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import math
import ast

# 定义数据表头即参数名
headers = ['status', 'duration', 'credit_history',
           'purpose', 'amount',
           'savings', 'employment_duration',
           'installment_rate', 'personal_status_sex', 'other_debtors',
           'present_residence', 'property',
           'age', 'other_installment_plans',
           'housing', 'number_credits',
           'job', 'people_liable',
           'telephone', 'foreign_worker',
           'credit_risk'] 

#[(0,3),(1,80),(0,4),(0,10),(1,200),(0,4),(0,4),(1,4),(0,1),(0,2),
# (1,4),(0,3),(1,8),(0,2),(0,2),(1,4),(0,3),(1,2),(0,1),(0,1)]
# 列举变量取值
st = ['A11', 'A12', 'A13', 'A14']#(0,3)
#dura[4,72]->(0,80)
cre_his = ['A30', 'A31', 'A32', 'A33', 'A34']#(0,4) 
pur = ['A40', 'A41', 'A42', 'A43', 'A44','A45', 'A46', 'A47', 'A48',
      'A49', 'A410'] #(0,10)                   
#amo[250,18424]-（1,200）
sav = [ 'A61', 'A62', 'A63', 'A64', 'A65']#(0,4)
employ_dura = [ 'A71', 'A72', 'A73', 'A74', 'A75']#(0,4)
#ins_rate = ['1', '2', '3', '4'](1,4)
per_sta_sex = ['A91', 'A92', 'A93', 'A94']#(0,1)？文件中有A95
other_deb = ['A101', 'A102', 'A103']#(0,2)
#pre_res = ['1', '2', '3', '4'](1,4)
pro = ['A121', 'A122', 'A123', 'A124']#(0,3)
#age [19,75](1,8)
other_ins_pls = ['A141', 'A142', 'A143']#(0,2)
hou = ['A151', 'A152', 'A153']#(0,2)
#num_cre = ['1', '2', '3', '4'](1,4)SS
job = ['A171', 'A172', 'A173', 'A174']#(0,3)
#peo_lia = ['1', '2'] (1,2)
tele = ['A191', 'A192']#(0,1)
for_work = ['A201', 'A202']#(0,1)

cre_risk = ['1', '2']

feature = []
out = []


class CreditNet(nn.Module):
    def __init__(self, num_of_features):
        super().__init__()
        self.fc1 = nn.Linear(num_of_features, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, 2)

    def forward(self, x):
        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        output = x # cross entropy in pytorch already includes softmax
        return output

# class CreditNet(nn.Module):
#     def __init__(self, num_of_features):
#         super().__init__()
#         self.fc1 = nn.Linear(num_of_features, 64)
#         self.fc2 = nn.Linear(64, 32)
#         self.fc3 = nn.Linear(32, 16)
#         self.fc4 = nn.Linear(16, 8)
#         self.fc5 = nn.Linear(8, 4)
#         self.fc6 = nn.Linear(4, 2)
#
#     def forward(self, x):
#         x = torch.flatten(x,1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x)
#         x = F.relu(x)
#         x = self.fc3(x)
#         x = F.relu(x)
#         x = self.fc4(x)
#         x = F.relu(x)
#         x = self.fc5(x)
#         x = F.relu(x)
#         x = self.fc6(x)
#         output = x # cross entropy in pytorch already includes softmax
#         return output

# 加载已有网络并计算二分类数量
def test(model,test_x,device):
    tensor_test_x = torch.FloatTensor(test_x.copy()) # transform to torch tensor 
    test_dataset = TensorDataset(tensor_test_x) # create dataset                                                     
    test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle = False) # create dataloader
    # print(type(test_dataloader))
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    # print(size)
    # print(num_batches)
    test_loss, correct = 0, 0
    pos = 0
    neg = 0
    gt50 = torch.tensor([[1,0]])
    model.eval()
    with torch.no_grad():
        for x in test_dataloader:
            # x = x.to(device)
            # print(x)
            # print(type(x[0]))
            pred = model(x[0])
            pred = pred.type(torch.FloatTensor)
            # print(pred.shape)
            dim0,dim1 = pred.shape
            for i in range(dim0):
                element = pred[0,:]
                element = element.unsqueeze(0)
                # print(element.shape)
                # print(gt50.shape)
                if(element.argmax(1)==gt50.argmax(1)):
                    pos = pos + 1
                else:
                    neg = neg + 1
            # pred_1 = (pred.argmax(1)).type(torch.float).sum().item()
            # pred_0 = (pred.argmax(0)).type(torch.float).sum().item()
    postive = pos / size
    negtive = neg /size
    #print(postive)
    #print(negtive)
    return postive, negtive

# 数据处理函数，生成可以用于输入网络的数据文件，首次运行后不用再运行
def split_a1():#换数据集即可处理其他的文件
    feature = []
    with open('credit_fairness/sample_age/data_on_age/a4.csv', 'r', encoding='utf-8') as file:
        header = next(file)
    # 逐行读取文件内容
        for line in file:
            # 对每一行进行处理
            # 例如，将每一行拆分成单词并打印出来
            features = line.strip().split(',')
            features[0] = st.index(features[0])
            features[1] = np.clip(int(features[1]), 1, 80)
            features[2] = cre_his.index(features[2])
            features[3] = pur.index(features[3])
            features[4] = np.clip(int(features[4]) /  100, 1, 200)
            features[5] = sav.index(features[5])
            features[6] = employ_dura.index(features[6])
            features[7] = np.clip(int(features[7]), 1, 4)
            features[8] = per_sta_sex.index(features[8])
            features[9] = other_deb.index(features[9])
            features[10] = np.clip(int(features[10]), 1, 4)
            features[11] = pro.index(features[11])
            features[12] = np.clip(int(features[12]) / 10, 1, 8)#age
            features[13] = other_ins_pls.index(features[13])
            features[14] = hou.index(features[14])
            features[15] = np.clip(int(features[15]), 1, 4)
            features[16] = job.index(features[16])
            features[17] = np.clip(int(features[17]), 1, 2)
            features[18] = tele.index(features[18])
            features[19] = for_work.index(features[19])
            feature.append(features[:20])
            
        feature = np.asarray(feature)
        np.savetxt("credit_fairness/sample_age/a4_feature.txt", feature)#换数据集同时换输出




def cal_fairness(net):
    device = 'cpu'
    time_start = time.time()
     # 修改网络
    PATH = "data/darta_numeric/credit.pt"
    # 修改输入文件
    test_x_a1 = np.loadtxt('credit_fairness/sample_age/a1_feature.txt')
    test_x_a2 = np.loadtxt('credit_fairness/sample_age/a2_feature.txt') 
    test_x_a3 = np.loadtxt('credit_fairness/sample_age/a3_feature.txt')
    test_x_a4 = np.loadtxt('credit_fairness/sample_age/a4_feature.txt') 
                   
    model.load_state_dict(torch.load(PATH))
 
    positive_a1 = 0
    positive_a2 = 0
    positive_a3 = 0
    positive_a4 = 0
    
    negitive_a1 = 0
    negitive_a2 = 0
    negitive_a3 = 0
    negitive_a4 = 0

    
    #返回二分类的比例，pos表示>=50K$分类的比例，neg表示<50K$分类的比例，pos+neg=1
    pos_a1, neg_a1 = test(model, test_x_a1, device)
    positive_a1 = positive_a1 + pos_a1
    negitive_a1 = negitive_a1 + neg_a1

    pos_a2, neg_a2 = test(model, test_x_a2, device)
    positive_a2 = positive_a2 + pos_a2
    negitive_a2 = negitive_a2 + neg_a2

    pos_a3, neg_a3 = test(model, test_x_a3, device)
    positive_a3 = positive_a3 + pos_a3
    negitive_a3 = negitive_a3 + neg_a3

    pos_a4, neg_a4 = test(model, test_x_a4, device)
    positive_a4 = positive_a4 + pos_a4
    negitive_a4 = negitive_a4 + neg_a4

 
    f1 = abs(positive_a1 - positive_a2)
    f2 = abs(positive_a1 - positive_a3)
    f3 = abs(positive_a1 - positive_a4)
    f4 = abs(positive_a2 - positive_a3)
    f5 = abs(positive_a2 - positive_a4)
    f6 = abs(positive_a3 - positive_a4)
   
    fair = [f1,f2,f3,f4,f5,f6]
    fairness = sum(fair) / len(fair)

    time_end = time.time()
    time_sum = time_end - time_start
    
    print("网络名称为：%s，特征%s的公平性取值为：%f, 所用时间为：%fs\n" % (net,"age", fairness, time_sum))


def cal_fairness1_age(model):
    device = 'cpu'
    device='cuda:0'
    time_start = time.time()
    # 修改网络

    model1=CreditNet(20)
    model1.load_state_dict(model)
    # 修改输入文件
    test_x_a1 = np.loadtxt('credit_age/a1_feature.txt')
    test_x_a2 = np.loadtxt('credit_age/a2_feature.txt')
    test_x_a3 = np.loadtxt('credit_age/a3_feature.txt')
    test_x_a4 = np.loadtxt('credit_age/a4_feature.txt')

    model=model1

    positive_a1 = 0
    positive_a2 = 0
    positive_a3 = 0
    positive_a4 = 0

    negitive_a1 = 0
    negitive_a2 = 0
    negitive_a3 = 0
    negitive_a4 = 0

    # 返回二分类的比例，pos表示>=50K$分类的比例，neg表示<50K$分类的比例，pos+neg=1
    pos_a1, neg_a1 = test(model, test_x_a1, device)
    positive_a1 = positive_a1 + pos_a1
    negitive_a1 = negitive_a1 + neg_a1

    pos_a2, neg_a2 = test(model, test_x_a2, device)
    positive_a2 = positive_a2 + pos_a2
    negitive_a2 = negitive_a2 + neg_a2

    pos_a3, neg_a3 = test(model, test_x_a3, device)
    positive_a3 = positive_a3 + pos_a3
    negitive_a3 = negitive_a3 + neg_a3

    pos_a4, neg_a4 = test(model, test_x_a4, device)
    positive_a4 = positive_a4 + pos_a4
    negitive_a4 = negitive_a4 + neg_a4

    f1 = abs(positive_a1 - positive_a2)
    f2 = abs(positive_a1 - positive_a3)
    f3 = abs(positive_a1 - positive_a4)
    f4 = abs(positive_a2 - positive_a3)
    f5 = abs(positive_a2 - positive_a4)
    f6 = abs(positive_a3 - positive_a4)

    fair = [f1, f2, f3, f4, f5, f6]
    fairness = sum(fair) / len(fair)

    time_end = time.time()
    time_sum = time_end - time_start

    #print("网络名称为：%s，特征%s的公平性取值为：%f, 所用时间为：%fs\n" % ("modelage", "age", fairness, time_sum))
    return  fairness

def recal_acc1(model):
    time_start = time.time()
    device = 'cpu'
    device = 'cuda:0'
    # 修改网络
    #PATH = "gender_fairness/optimize/" + str(net) + ".pt"
    # 修改输入文件

    train_x = np.loadtxt('data/credit/trainx.txt')
    train_y = np.loadtxt('data/credit/trainy.txt')
    test_x = np.loadtxt('data/credit/testx.txt')
    test_y = np.loadtxt('data/credit/testy.txt')

    tensor_test_x = torch.FloatTensor(test_x.copy())
    tensor_test_y = torch.FloatTensor(test_y.copy())

    test_dataset = TensorDataset(tensor_test_x,
                                 tensor_test_y)  # create dataset
    test_dataloader = DataLoader(test_dataset, batch_size=100)  # create dataloader

    model1 = CreditNet(20)
    model1.load_state_dict(model)

    model=model1.to(device)


    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)

    model.eval()
    test_loss, correct = 0, 0
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in test_dataloader:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            pred = pred.type(torch.FloatTensor)
            y = y.type(torch.FloatTensor)
            # print(pred)
            # print(y)
            test_loss += loss_fn(pred, y).item()
            # print(pred.argmax(1))
            # print(y.argmax(1))
            correct += (pred.argmax(1) == y.argmax(1)).sum().item()

    test_loss /= num_batches
    error = size - correct
    correct /= size

    #print(f"Test: \n Test size: {(size):>0.1f}, Error size: {(error):>0.1f}, Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct

def recal_acc(model,net):
    time_start = time.time()
    device = 'cpu'
     # 修改网络
    PATH = "gender_fairness/optimize/"+ str(net) +".pt"
    # 修改输入文件
    test_x = np.loadtxt('data/data_numeric/testx.txt')
    test_y = np.loadtxt('data/data_numeric/testy.txt') 

    tensor_test_x = torch.FloatTensor(test_x.copy())                                                            
    tensor_test_y = torch.FloatTensor(test_y.copy()) 

    test_dataset = TensorDataset(tensor_test_x, tensor_test_y) # create dataset                                                     
    test_dataloader = DataLoader(test_dataset, batch_size=100) # create dataloader

    model.load_state_dict(torch.load(PATH))
 
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    
    model.eval()
    test_loss, correct = 0, 0
    loss_fn = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for x, y in test_dataloader:
            x, y = x.to(device), y.to(device)
            
            pred = model(x)
            pred = pred.type(torch.FloatTensor)
            y = y.type(torch.FloatTensor)
            # print(pred)
            # print(y)
            test_loss += loss_fn(pred, y).item()
            # print(pred.argmax(1))
            # print(y.argmax(1))
            correct += (pred.argmax(1) == y.argmax(1)).sum().item()
    
    test_loss /= num_batches
    error = size - correct
    correct /= size

    print(f"Test: \n Test size: {(size):>0.1f}, Error size: {(error):>0.1f}, Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    # return correct

if __name__ == "__main__":
    model = CreditNet(14)
    # split_a1()
    # recal_acc(model,'optimizednet1')# 当修复后重新调用
    cal_fairness('credit')
    



