# -*- coding: utf-8 -*-
"""
Created on Tuesday, January 9 10:18:04 2024

@author: maosheng
"""
import numpy as np
from pypower.ext2int import ext2int
from pypower.idx_gen import GEN_BUS, PG, QG, VG, QMAX, QMIN, GEN_BUS, GEN_STATUS, PMAX, PMIN
from pypower.idx_brch import PF, QF, BR_STATUS, RATE_A, RATE_B, RATE_C, BR_R, BR_B, BR_X, T_BUS, F_BUS, MU_SF, MU_ST, MU_ANGMIN, MU_ANGMAX, PT
import os
import torch
import torch.nn as nn
import torch.optim as optim

import csv, copy
import numpy as np


def save_csv(data, file_path):
    filename = file_path
    data = np.array(data)
    if len(data.shape) < 2:
        data = data[:, np.newaxis]
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)


def read_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    # 将数字转化为 float 形式
    for row in data:
        for i in range(len(row)):
            try:
                row[i] = float(row[i])
            except ValueError:
                pass  # 如果无法转换为 float，保留原始字符串

    return np.array(data)


class SDAE(nn.Module):
    def __init__(self, ):
        super(SDAE, self).__init__()

        self.predictor = nn.Sequential(
            nn.Linear(127, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            # nn.Sigmoid()
        ).to(device)

    def forward(self, ef):
        cut = self.predictor(ef)
        return cut


def load_data(data_dir):
    # 读取原始数据
    data_size = 50000
    data = np.load(data_dir, allow_pickle=True)
    input_G = data['input_G'][:data_size, :, :]
    input_B = data['input_B'][:data_size, :, :]
    input_Pd = data['input_Pd'][:data_size, :]
    input_Qd = data['input_Qd'][:data_size, :]
    output_cut = data['output_cut'][:data_size, :]

    input_Gen = data['input_Gen'][:data_size, :]

    node_num = input_Pd.shape[1]
    sample_num = input_Pd.shape[0]

    # 处理没有不能直接读取的数据
    input_e = np.ones((sample_num, node_num, 1))
    input_f = np.zeros((sample_num, node_num, 1))
    input_G_diag = np.array([np.diag(input_G[i, :, :])[:, np.newaxis] for i in range(sample_num)])
    input_B_diag = np.array([np.diag(input_B[i, :, :])[:, np.newaxis] for i in range(sample_num)])
    input_G_ndiag = np.array([input_G[i, :, :] - np.diag(np.diag(input_G[i, :, :])) for i in range(sample_num)])
    input_B_ndiag = np.array([input_B[i, :, :] - np.diag(np.diag(input_B[i, :, :])) for i in range(sample_num)])

    return (input_e, input_f, input_G, input_B, input_G_diag, input_B_diag, input_G_ndiag, input_B_ndiag,
            input_Pd, input_Qd, input_Gen, output_cut)


def preprocess(mpc, data_dir, all_data=False):
    (input_e, input_f, input_G, input_B, input_G_diag, input_B_diag, input_G_ndiag, input_B_ndiag, input_Pd, input_Qd,
     input_Gen, output_cut) = load_data(data_dir)

    # 对数据进行预处理
    output_cut[output_cut < 0.000001] = 0
    output_cut = output_cut[:, :, np.newaxis]
    input_Gen = input_Gen[:, :, np.newaxis]
    input_Pd = input_Pd[:, :, np.newaxis] / 100
    input_Qd = input_Qd[:, :, np.newaxis] / 100

    # # 剔除拓扑改变时，系统解裂，只有一个节点独立的情况
    # temp = np.sum(np.abs(input_G), -1) + np.sum(np.abs(input_B), -1)
    # temp = np.min(temp, -1)
    # idx = np.where(temp > 0)[0]
    #
    # (input_e, input_f, input_G, input_B, input_G_diag, input_B_diag, input_G_ndiag, input_B_ndiag, input_Pd, input_Qd,
    #  input_Gen, output_cut) = (input_e[idx, :, :], input_f[idx, :, :], input_G[idx, :, :], input_B[idx, :, :],
    #                            input_G_diag[idx, :, :], input_B_diag[idx, :, :], input_G_ndiag[idx, :, :],
    #                            input_B_ndiag[idx, :, :],
    #                            input_Pd[idx, :, :], input_Qd[idx, :, :], input_Gen[idx, :, :], output_cut[idx, :, :])

    # 对切负荷量进行归一化处理
    output_cut = np.sum(output_cut[:, :, 0], axis=-1)
    output_cut[output_cut > 0.01] = 1
    output_cut[output_cut <= 0.01] = 0
    output_cut = output_cut[:, np.newaxis]
    input_Pd_orginal = copy.deepcopy(input_Pd)

    # 获取部分参数
    sample_num = input_Pd.shape[0]
    bus_num = input_Pd.shape[1]
    In = np.eye(bus_num)

    # 构建输入特征向量
    input_pq_Bdiag = np.concatenate([input_Pd, input_Qd, input_B_diag, input_Gen], axis=-2)[:, :, 0]

    # # 对数据进行归一化处理
    # input_pq_Bdiag, _, _ = Z_Score(input_pq_Bdiag)
    #
    # input_pq_Bdiag = torch.tensor(input_pq_Bdiag, dtype=torch.float32).to(device)
    # output_cut = torch.tensor(output_cut, dtype=torch.float32).to(device)
    if all_data:
        return [input_pq_Bdiag, output_cut[:, :, 0]], input_Pd_orginal[:, :, 0]
    else:
        return [input_pq_Bdiag[:10000, :], output_cut[:10000, :]], [
        input_pq_Bdiag[10000:, :], output_cut[10000:, :]], input_Pd_orginal[:10000, :,
                                                                    0], input_Pd_orginal[10000:, :, 0]
    # return (input_pq_Bdiag[:20000, :], output_cut[:20000, :]), (
    # input_pq_Bdiag[25000:29000, :], output_cut[25000:29000, :]), input_Pd_orginal[:20000, :,
    #                                                              0], input_Pd_orginal[25000:29000, :, 0]


def data_generator(train_data, batch_size=15):
    (input_pq_Bdiag, output_cut) = train_data
    idx = np.random.permutation(input_pq_Bdiag.shape[0])
    for k in range(int(np.ceil(input_pq_Bdiag.shape[0] / batch_size))):
        from_idx = k * batch_size
        to_idx = (k + 1) * batch_size
        index = idx[from_idx:to_idx]
        # input_e, input_f, input_G, input_B, input_G_diag, input_B_diag, input_Pd, input_Qd
        inputs_ = input_pq_Bdiag[index, :]
        outputs_ = output_cut[index, :]

        yield inputs_, outputs_


def Z_Score(trainData):
    trainData = np.array(trainData)
    mean_train = np.mean(trainData, axis=0)
    std_train = np.std(trainData, axis=0)
    std_train = std_train + 0.0000001
    trainData = (trainData - mean_train)
    trainData = trainData / std_train
    trainData[np.isnan(trainData)] = 0
    trainData[np.isinf(trainData)] = 0
    # trainData = 1/(1+np.exp(-1*trainData))
    return trainData, mean_train, std_train


if __name__ == '__main__':
    # 配置环境变量是否使用GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    ################################## set device ##################################
    print("============================================================================================")
    # set device to cpu or cuda
    device = torch.device('cpu')
    if (torch.cuda.is_available()):
        device = torch.device('cuda:0')
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        print("Device set to : cpu")
    print("============================================================================================")

    # 加载IEEE 39节点系统数据
    from pypower.case39 import case39
    import matplotlib.pyplot as plt

    # from utils import *

    mpc = case39()

    # 导入训练数据
    data_dir = "./dataset/case_39_evaluation_testing_samples[30000].npz"
    train_data, test_data, input_Pd_original_training, input_Pd_original_testing = preprocess(mpc, data_dir)

    input_pq_Bdiag, mean_, std_ = Z_Score(train_data[0])
    temp = (test_data[0] - mean_) / std_
    temp[np.isnan(temp)] = 0
    temp[np.isinf(temp)] = 0

    train_data[0] = torch.tensor(input_pq_Bdiag, dtype=torch.float32).to(device)
    train_data[1] = torch.tensor(train_data[1], dtype=torch.float32).to(device)
    test_data[0] = torch.tensor(temp, dtype=torch.float32).to(device)
    test_data[1] = torch.tensor(test_data[1], dtype=torch.float32).to(device)

    # 构建神经网络，并配置训练参数
    model = SDAE()
    lossfun = nn.MSELoss()
    # lossfun = nn.BCELoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    saved_folder = "./models/M8/"
    if (not os.path.exists(saved_folder)):
        os.mkdir(saved_folder)

    # # 进行训练
    # epoch_num = 1000
    # losses_train = []
    # losses_test = []
    # for epoch in range(epoch_num):  # loop over the dataset multiple times
    #     epoch_loss = []
    #     for fs, labels in data_generator(train_data, batch_size=100):
    #         optimizer.zero_grad()
    #         outputs = model(fs)
    #         loss = lossfun(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #
    #         epoch_loss.append(loss.item())
    #
    #     # 训练完成之后使用测试集进行测试
    #     res = model(test_data[0])
    #     error = lossfun(res, test_data[1]).item()
    #
    #     # 保存loss、accuracy值，可用于可视化
    #     losses_train.append(np.mean(epoch_loss))
    #     losses_test.append(error)
    #     # plt.plot(losses_train)
    #     # plt.plot(losses_test)
    #     # plt.legend(['Train', 'Test'])
    #     # plt.savefig(saved_folder + '/lossFig.png')
    #     # plt.close()
    #
    #     # 每一个epoch后打印Loss、Accuracy以及花费的时间
    #     print(
    #         "Epoch {:03d}: Train Loss: {:.6f}, Test Loss: {:.6f}".format(epoch, np.mean(epoch_loss), error))
    #
    #     if epoch % 10 == 0:
    #         # 训练完成后更新保存模型
    #         model_path = saved_folder + "/Torch_Model.pth"
    #         torch.save(model.state_dict(), model_path)
    #
    #         # 保存训练过程的损失值
    #         temp = np.concatenate([np.array(losses_train)[:, np.newaxis], np.array(losses_test)[:, np.newaxis]],
    #                               axis=-1)
    #         save_csv(temp, saved_folder + "/train_and_test_losses.csv")
    #
    # print('finish!')

    model.load_state_dict(torch.load("./models/M8/Torch_Model.pth"))

    data_size = 10000
    data = np.load("./dataset/case_39_evaluation_testing_samples[30000].npz", allow_pickle=True)
    input_G = data['input_G'][data_size:, :, :]
    input_B = data['input_B'][data_size:, :, :]
    input_Pd = data['input_Pd'][data_size:, :] / 100
    input_Qd = data['input_Qd'][data_size:, :] / 100
    output_cut = data['output_cut'][data_size:, :]

    input_Gen = data['input_Gen'][data_size:, :]
    input_Br = data['input_Br'][data_size:, :]

    input_B_diag = np.array([np.diag(input_B[i, :, :]) for i in range(input_Pd.shape[0])])
    # 构建输入特征向量
    input_pq_Bdiag = np.concatenate([input_Pd, input_Qd, input_B_diag, input_Gen], axis=-1)

    temp = (input_pq_Bdiag - mean_) / std_
    temp[np.isnan(temp)] = 0
    temp[np.isinf(temp)] = 0

    import time


    input_pq_Bdiag = torch.tensor(temp, dtype=torch.float32).to(device)

    # res = model(test_data[0]).detach().cpu().numpy()
    res = model(input_pq_Bdiag).detach().cpu().numpy()

    from Loadshedding import loadshedding_model
    from pyomo.environ import value
    import time
    from tqdm import tqdm

    t1 = time.time()
    load_shedding = []
    res[res > 0.5] = 1
    res[res <= 0.5] = 0
    LOLP = np.sum(res) / res.shape[0]
    print(' LOLP 所提方法：{}'.format(LOLP))
    for i in tqdm(range(res.shape[0])):
        if res[i] > 0.5:
            mpc['bus'][:, 2] = input_Pd[i] * 100
            mpc['bus'][:, 3] = input_Qd[i] * 100
            mpc['gen'][:, GEN_STATUS] = input_Gen[i]
            mpc['branch'][:, BR_STATUS] = input_Br[i]
            res_loadshedding, status = loadshedding_model(mpc)
            temp = np.array([value(res_loadshedding.C[j]) for j in range(39)])
            load_shedding.append(temp)

    LOLP = len(load_shedding) / res.shape[0]
    print(' LOLP 所提方法：{}'.format(LOLP))

    EENS = np.sum(load_shedding) * 100 / output_cut.shape[0] * 8760

    t2 = time.time()

    print('EDNS 所提方法：{}'.format(EENS))
    print('计算时间(s)：', t2 - t1)
    print('finish!')
    print(123)

