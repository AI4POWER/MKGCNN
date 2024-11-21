# -*- coding: utf-8 -*-
"""
Created on Tuesday, January 9 10:18:04 2024

@author: maosheng
"""
import numpy as np
from pypower.ext2int import ext2int
from pypower.idx_gen import GEN_BUS, PG, QG, VG, QMAX, QMIN, GEN_BUS, GEN_STATUS, PMAX, PMIN
import os
import torch
import torch.nn as nn
import torch.optim as optim
import csv, copy
import matplotlib.pyplot as plt


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


class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCNLayer, self).__init__()

        # self.fc_v1 = nn.Linear(input_dim * 3 * 2, input_dim).to(device)
        # output_dim = int(output_dim/4)
        self.fc_v1 = nn.Linear(input_dim * 5, output_dim).to(device)
        self.fc_v2 = nn.Linear(input_dim * 5, output_dim).to(device)
        self.trans = torch.ones((57, 1)).to(device)

        self.atten_e = nn.Linear(input_dim, 1).to(device)
        self.atten_f = nn.Linear(input_dim, 1).to(device)

        self.pool = PoolLayer2()

    def forward(self, e, f, k1, k2, G_ndiag, B_ndiag, G_diag, B_diag, Pd, Qd):
        base = e * e + f * f + 0.1
        alpha = Pd * e / base + Qd * f / base - torch.bmm(G_ndiag, e) - torch.bmm(B_ndiag, f)
        beta = Qd * e / base - Pd * f / base + torch.bmm(G_ndiag, f) + torch.bmm(B_ndiag, f)

        base_gb = G_diag * G_diag + B_diag * B_diag
        e3 = alpha * G_diag / base_gb + beta * B_diag / base_gb
        f3 = beta * G_diag / base_gb - alpha * B_diag / base_gb

        base1 = torch.bmm(G_ndiag, e) - torch.bmm(B_ndiag, f)
        base2 = torch.bmm(G_ndiag, f) + torch.bmm(B_ndiag, e)
        # base4 = base1 * base1 + base2 * base2 + 0.1
        base4 = G_diag * G_diag + B_diag * B_diag

        new_e = ((Pd - (e * e + f * f) * G_diag) * base1 + (Qd + (e * e + f * f) * B_diag) * base2) / base4
        new_f = ((Pd - (e * e + f * f) * G_diag) * base2 - (Qd + (e * e + f * f) * B_diag) * base1) / base4

        e1 = torch.bmm(k1, e)
        f1 = torch.bmm(k1, f)

        e2 = torch.bmm(k2, e)
        f2 = torch.bmm(k2, f)

        a1 = torch.sigmoid(self.atten_e(self.pool(e3)))
        a2 = torch.sigmoid(self.atten_e(self.pool(new_e)))
        a3 = torch.sigmoid(self.atten_e(self.pool(e1)))
        a4 = torch.sigmoid(self.atten_e(self.pool(e2)))
        a_base = a1 + a2 + a3 + a4 + 0.0001
        a1 = a1 / a_base
        a2 = a2 / a_base
        a3 = a3 / a_base
        a4 = a4 / a_base

        e_new = torch.cat((a1 * e3, a2 * new_e, a3 * e1, a4 * e2, e), 2)
        # e_new_max = torch.max(e_new, dim=-2, keepdim=True)
        # e_new_max = torch.mm(self.trans, e_new_max)

        b1 = torch.sigmoid(self.atten_f(self.pool(f3)))
        b2 = torch.sigmoid(self.atten_f(self.pool(new_f)))
        b3 = torch.sigmoid(self.atten_f(self.pool(f1)))
        b4 = torch.sigmoid(self.atten_f(self.pool(f2)))
        b_base = b1 + b2 + b3 + b4 + 0.0001
        b1 = b1 / b_base
        b2 = b2 / b_base
        b3 = b3 / b_base
        b4 = b4 / b_base

        f_new = torch.cat((b1 * f3, b2 * new_f, b3 * f1, b4 * f2, f), 2)
        e_new = torch.relu(self.fc_v1(e_new))
        f_new = torch.relu(self.fc_v2(f_new))

        return e_new, f_new


class PoolLayer2(nn.Module):
    def __init__(self, ):
        super(PoolLayer2, self).__init__()

        self.trans = torch.ones((118, 1)).to(device)

    def forward(self, ef):
        ef_pool = torch.mean(ef, dim=-2, keepdim=True)
        # ef_pool = torch.matmul(self.trans, ef_pool)

        return ef_pool


class PoolLayer(nn.Module):
    def __init__(self, ):
        super(PoolLayer, self).__init__()

        self.trans = torch.ones((118, 1)).to(device)

    def forward(self, ef):
        ef_pool = torch.mean(ef, dim=-2, keepdim=True)
        ef_pool = torch.matmul(self.trans, ef_pool)
        ef_new = torch.cat((ef, ef_pool), -1)

        return ef_new


class PIGCNN(nn.Module):
    def __init__(self, ):
        super(PIGCNN, self).__init__()

        self.conv1 = GCNLayer(input_dim=1, output_dim=16)
        self.conv2 = GCNLayer(input_dim=16, output_dim=128)
        self.conv3 = GCNLayer(input_dim=128, output_dim=128)
        self.conv4 = GCNLayer(input_dim=128, output_dim=128)
        self.conv5 = GCNLayer(input_dim=128, output_dim=128)
        # self.flatten = nn.Flatten()
        self.pool = PoolLayer()

        self.predictor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            # nn.Sigmoid()
        ).to(device)

    def forward(self, e, f, inputs, k1, k2, G_ndiag, B_ndiag, G_diag, B_diag, Pd, Qd):
        e, f = self.conv1(e, f, k1, k2, G_ndiag, B_ndiag, G_diag, B_diag, Pd, Qd)
        e, f = self.conv2(e, f, k1, k2, G_ndiag, B_ndiag, G_diag, B_diag, Pd, Qd)
        e, f = self.conv3(e, f, k1, k2, G_ndiag, B_ndiag, G_diag, B_diag, Pd, Qd)
        e, f = self.conv4(e, f, k1, k2, G_ndiag, B_ndiag, G_diag, B_diag, Pd, Qd)
        e, f = self.conv5(e, f, k1, k2, G_ndiag, B_ndiag, G_diag, B_diag, Pd, Qd)

        ef = torch.cat((e, f), 2)
        # fs = self.flatten(ef)
        fs = self.pool(ef)
        cut = self.predictor(fs)
        return cut


def load_data(data_dir):
    # 读取原始数据
    data_size = 60000
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


def preprocess(mpc, data_dir):
    (input_e, input_f, input_G, input_B, input_G_diag, input_B_diag, input_G_ndiag, input_B_ndiag, input_Pd, input_Qd,
     input_Gen, output_cut) = load_data(data_dir)

    # 对数据进行预处理
    output_cut[output_cut < 0.000001] = 0
    output_cut = output_cut[:, :, np.newaxis]
    input_Gen = input_Gen[:, :, np.newaxis]
    input_Pd = input_Pd[:, :, np.newaxis] / 100
    input_Qd = input_Qd[:, :, np.newaxis] / 100

    # 剔除拓扑改变时，系统解裂，只有一个节点独立的情况
    temp = np.sum(np.abs(input_G), -1) + np.sum(np.abs(input_B), -1)
    temp = np.min(temp, -1)
    idx = np.where(temp > 0)[0]

    (input_e, input_f, input_G, input_B, input_G_diag, input_B_diag, input_G_ndiag, input_B_ndiag, input_Pd, input_Qd,
     input_Gen, output_cut) = (input_e[idx, :, :], input_f[idx, :, :], input_G[idx, :, :], input_B[idx, :, :],
                               input_G_diag[idx, :, :], input_B_diag[idx, :, :], input_G_ndiag[idx, :, :],
                               input_B_ndiag[idx, :, :],
                               input_Pd[idx, :, :], input_Qd[idx, :, :], input_Gen[idx, :, :], output_cut[idx, :, :])

    # 对切负荷量进行归一化处理
    output_cut = output_cut / input_Pd
    output_cut[np.isnan(output_cut)] = 0
    output_cut[np.isinf(output_cut)] = 0
    input_Pd_orginal = copy.deepcopy(input_Pd)

    # 获取部分参数
    sample_num = input_Pd.shape[0]
    bus_num = input_Pd.shape[1]
    In = np.eye(bus_num)

    # # 释放部分内存
    # del input_G, input_B

    # 对输入Pd和Qd进行处理，把机组出力设置为最大出力的0.5倍，如果机组出现故障，直接设置为0
    # 构建机组与节点的关联矩阵
    A = np.zeros((mpc['bus'].shape[0], mpc['gen'].shape[0]))
    for i in range(mpc['gen'].shape[0]):
        for j in range(mpc['bus'].shape[0]):
            if mpc['gen'][i, GEN_BUS] == mpc['bus'][j, 0]:
                A[j, i] = 1

    for i in range(input_Pd.shape[0]):
        Pg = mpc['gen'][:, PMAX][:, np.newaxis] * input_Gen[i, :, :] / 100
        input_Pd[i, :, :] = 0.5 * np.dot(A, Pg) - input_Pd[i, :, :]
        Qg = mpc['gen'][:, QMAX][:, np.newaxis] * input_Gen[i, :, :] / 100
        input_Qd[i, :, :] = 0.5 * np.dot(A, Qg) - input_Qd[i, :, :]

    ######################################################
    # 开始构建三种卷积核
    ######################################################
    # 第一种基于G矩阵的卷积核
    k1 = np.zeros((sample_num, bus_num, bus_num))
    for i in range(sample_num):
        k = 0.01
        A = np.sqrt(np.square(input_G[i, :, :]) + np.square(input_B[i, :, :]))
        A = np.exp(-k * A)
        A[A == 1.0] = 0
        A = np.array(A)
        k1[i, :, :] = A

    # 第2种基于B矩阵的卷积核
    k2 = np.zeros((sample_num, bus_num, bus_num))
    for i in range(sample_num):
        D = np.diag((-input_B_diag[i, :, 0]+1) ** (-0.5))
        kernel = np.dot(np.dot(D, (input_B_ndiag[i, :, :] + In)), D)
        k2[i, :, :] = kernel

    inputs = np.concatenate([input_Pd, input_Qd], axis=-1)
    # 对数据进行归一化处理
    inputs_, _, _ = Z_Score(inputs)

    # 转化成tensor
    # inputs_ = torch.tensor(inputs_, dtype=torch.float32).to(device)
    # k1 = torch.tensor(k1, dtype=torch.float32).to(device)
    # k2 = torch.tensor(k2, dtype=torch.float32).to(device)
    # input_Pd = torch.tensor(input_Pd, dtype=torch.float32).to(device)
    # input_Qd = torch.tensor(input_Qd, dtype=torch.float32).to(device)
    # input_G_diag = torch.tensor(input_G_diag, dtype=torch.float32).to(device)
    # input_B_diag = torch.tensor(input_B_diag, dtype=torch.float32).to(device)
    # input_G_ndiag = torch.tensor(input_G_ndiag, dtype=torch.float32).to(device)
    # input_B_ndiag = torch.tensor(input_B_ndiag, dtype=torch.float32).to(device)
    # input_e = torch.tensor(input_e, dtype=torch.float32).to(device)
    # input_f = torch.tensor(input_f, dtype=torch.float32).to(device)

    # output_cut = torch.tensor(output_cut[:, :, 0], dtype=torch.float32).to(device)
    output_cut = output_cut[:, :, 0]

    # e, f, k1, k2, G_ndiag, B_ndiag, G_diag, B_diag, Pd, Qd
    s0 = 20000
    s1 = 30000
    s2 = 60000
    return (input_e[:s0, :, :], input_f[:s0, :, :], inputs_[:s0, :, :], k1[:s0, :, :], k2[:s0, :, :],
            input_G_ndiag[:s0, :, :], input_B_ndiag[:s0, :, :],
            input_G_diag[:s0, :, :], input_B_diag[:s0, :, :],
            input_Pd[:s0, :, :], input_Qd[:s0, :, :], output_cut[:s0, :]), \
        [input_e[s1:s2, :, :], input_f[s1:s2, :, :], inputs_[s1:s2, :, :], k1[s1:s2, :, :], k2[s1:s2, :, :],
         input_G_ndiag[s1:s2, :, :], input_B_ndiag[s1:s2, :, :],
         input_G_diag[s1:s2, :, :], input_B_diag[s1:s2, :, :],
         input_Pd[s1:s2, :, :], input_Qd[s1:s2, :, :], output_cut[s1:s2, :]], input_Pd_orginal[:s1, :,
                                                                              0], input_Pd_orginal[s1:s2, :, 0]


def data_generator(train_data, input_Pd_original_training, batch_size=15):
    (e, f, inputs, k1, k2, G_ndiag, B_ndiag, G_diag, B_diag, Pd, Qd, output_cut) = train_data

    idx = np.random.permutation(e.shape[0])
    for k in range(int(np.ceil(e.shape[0] / batch_size))):
        from_idx = k * batch_size
        to_idx = (k + 1) * batch_size
        index = idx[from_idx:to_idx]
        # input_e, input_f, input_G, input_B, input_G_diag, input_B_diag, input_Pd, input_Qd
        inputs_ = [e[index, :, :],
                   f[index, :, :],
                   inputs[index, :, :],
                   k1[index, :, :],
                   k2[index, :, :],
                   G_ndiag[index, :, :],
                   B_ndiag[index, :, :],
                   G_diag[index, :, :],
                   B_diag[index, :, :],
                   Pd[index, :, :],
                   Qd[index, :, :]]
        outputs_ = output_cut[index, :]
        pd_ = input_Pd_original_training[index, :]

        yield inputs_, outputs_, pd_


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
    return trainData, mean_train[np.newaxis, :, np.newaxis], std_train[np.newaxis, :, np.newaxis]


def add_samples(train_data, input_Pd_original_training):
    # 扩充切负荷样本
    training_data_size = train_data[0].shape[0]
    (e, f, inputs, k1, k2, G_ndiag, B_ndiag, G_diag, B_diag, Pd, Qd, output_cut) = train_data
    temp = np.sum(output_cut, axis=-1)
    idx = np.where(temp != 0)[0]
    labels = output_cut[idx, :] * input_Pd_original_training[idx, :]
    disturbance = (np.random.random(labels.shape) - 0.5) / 0.5 * labels
    labels = labels + disturbance
    pd_new = input_Pd_original_training[idx, :] + disturbance
    input_Pd_original_training = np.concatenate([input_Pd_original_training, pd_new], axis=0)

    labels = labels / pd_new
    labels[np.isnan(labels)] = 0
    labels[np.isinf(labels)] = 0
    PD_new = Pd[idx, :, 0] + input_Pd_original_training[idx, :] - pd_new

    train_data = (np.concatenate([e, e[idx, :, :]], axis=0),
                  np.concatenate([f, f[idx, :, :]], axis=0),
                  np.concatenate([inputs, inputs[idx, :, :]], axis=0),
                  np.concatenate([k1, k1[idx, :, :]], axis=0),
                  np.concatenate([k2, k2[idx, :, :]], axis=0),
                  np.concatenate([G_ndiag, G_ndiag[idx, :, :]], axis=0),
                  np.concatenate([B_ndiag, B_ndiag[idx, :, :]], axis=0),
                  np.concatenate([G_diag, G_diag[idx, :, :]], axis=0),
                  np.concatenate([B_diag, B_diag[idx, :, :]], axis=0),
                  np.concatenate([Pd, PD_new[:, :, np.newaxis]], axis=0),
                  np.concatenate([Qd, Qd[idx, :, :]], axis=0),
                  np.concatenate([output_cut, labels], axis=0)
                  )
    return train_data, input_Pd_original_training


def separate_samples(train_data, input_Pd_original_training, scale):
    train_data_out = [train_data[0][:scale, :, :],
                      train_data[1][:scale, :, :],
                      train_data[2][:scale, :, :],
                      train_data[3][:scale, :, :],
                      train_data[4][:scale, :, :],
                      train_data[5][:scale, :, :],
                      train_data[6][:scale, :, :],
                      train_data[7][:scale, :, :],
                      train_data[8][:scale, :, :],
                      train_data[9][:scale, :, :],
                      train_data[10][:scale, :, :],
                      train_data[11][:scale, :]]
    input_Pd_original_training_out = input_Pd_original_training[:scale, :]
    return train_data_out, input_Pd_original_training_out


if __name__ == '__main__':
    # 配置环境变量是否使用GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('本程序测试不同数量的训练数据下，M5的测试误差')
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
    from pypower.case118 import case118

    mpc = case118()
    # mpc['branch'][:, RATE_A] = 160

    # 导入训练数据
    try:
        data_dir = "/home/cqu/GMS/big_space/maoshenggao_space/xinyuPaper/dataset/case_118_training_dist[1_0.3]_sample[60000].npz"
        train_data0, test_data, input_Pd_original_training0, input_Pd_original_testing = preprocess(mpc, data_dir)
    except:
        data_dir = "./dataset/case_118_training_dist[1_0.3]_sample[60000].npz"
        train_data0, test_data, input_Pd_original_training0, input_Pd_original_testing = preprocess(mpc, data_dir)

    test_data[0] = torch.tensor(test_data[0], dtype=torch.float32).to(device)
    test_data[1] = torch.tensor(test_data[1], dtype=torch.float32).to(device)
    test_data[2] = torch.tensor(test_data[2], dtype=torch.float32).to(device)
    test_data[3] = torch.tensor(test_data[3], dtype=torch.float32).to(device)
    test_data[4] = torch.tensor(test_data[4], dtype=torch.float32).to(device)
    test_data[5] = torch.tensor(test_data[5], dtype=torch.float32).to(device)
    test_data[6] = torch.tensor(test_data[6], dtype=torch.float32).to(device)
    test_data[7] = torch.tensor(test_data[7], dtype=torch.float32).to(device)
    test_data[8] = torch.tensor(test_data[8], dtype=torch.float32).to(device)
    test_data[9] = torch.tensor(test_data[9], dtype=torch.float32).to(device)
    test_data[10] = torch.tensor(test_data[10], dtype=torch.float32).to(device)
    test_data[11] = torch.tensor(test_data[11], dtype=torch.float32).to(device)

    output_cut_eval = test_data[11].detach().cpu().numpy() * input_Pd_original_testing
    p = np.mean(output_cut_eval, axis=1)
    idx = np.where(p > 0.0001)[0]
    lolp0 = idx.shape[0] / p.shape[0]
    EENS0 = np.sum(output_cut_eval) * 100 / input_Pd_original_testing.shape[0] * 8760
    print(lolp0, EENS0)

    # 构建神经网络，并配置训练参数
    model = PIGCNN()
    lossfun = nn.L1Loss()
    lossfun2 = nn.MSELoss()
    # torch.save(model.state_dict(), './models/original_model.pth')

    # testing_scale = [20000]
    #
    # testing_errors = []
    # for scale in testing_scale:
    #     saved_folder = "./models/M5_{}".format(scale)
    #     if (not os.path.exists(saved_folder)):
    #         os.mkdir(saved_folder)
    #
    #     # 初始化定义一个优化器
    #     optimizer = optim.Adam(model.parameters(), lr=0.001)
    #     # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)
    #     # model.load_state_dict(torch.load("./models/Model.pth"))
    #     # model.load_state_dict(torch.load("./models/M5_20000/Model.pth"))
    #     # 切分训练样本
    #     train_data, input_Pd_original_training = separate_samples(train_data0, input_Pd_original_training0, scale)
    #
    #     # 扩充样本集
    #     train_data, input_Pd_original_training = add_samples(train_data, input_Pd_original_training)
    #     train_data, input_Pd_original_training = add_samples(train_data, input_Pd_original_training)

        # # 进行训练
        # epoch_num = 1000
        # losses_train = []
        # losses_test = []
        # best_model = 10000000
        # for epoch in range(epoch_num):  # loop over the dataset multiple times
        #     epoch_loss = []
        #     if epoch == 800:
        #         optimizer = optim.Adam(model.parameters(), lr=0.0001)
        #     # 扰动学习
        #     for fs, labels, pd_ in data_generator(train_data, input_Pd_original_training, batch_size=100):
        #         optimizer.zero_grad()
        #         fs0 = copy.deepcopy(fs)
        #         labels0 = copy.deepcopy(labels)
        #
        #         labels = labels * pd_
        #         disturbance = (np.random.random(labels.shape) - 0.5) / 0.5 * labels
        #         labels = labels + disturbance
        #         pd_new = pd_ + disturbance
        #         fs[9][:, :, 0] = fs[9][:, :, 0] + pd_ - pd_new
        #         labels = labels / pd_new
        #         labels[np.isnan(labels)] = 0
        #         labels[np.isinf(labels)] = 0
        #
        #         fs[0] = torch.tensor(np.concatenate([fs[0], fs0[0]], axis=0), dtype=torch.float32).to(device)
        #         fs[1] = torch.tensor(np.concatenate([fs[1], fs0[1]], axis=0), dtype=torch.float32).to(device)
        #         fs[2] = torch.tensor(np.concatenate([fs[2], fs0[2]], axis=0), dtype=torch.float32).to(device)
        #         fs[3] = torch.tensor(np.concatenate([fs[3], fs0[3]], axis=0), dtype=torch.float32).to(device)
        #         fs[4] = torch.tensor(np.concatenate([fs[4], fs0[4]], axis=0), dtype=torch.float32).to(device)
        #         fs[5] = torch.tensor(np.concatenate([fs[5], fs0[5]], axis=0), dtype=torch.float32).to(device)
        #         fs[6] = torch.tensor(np.concatenate([fs[6], fs0[6]], axis=0), dtype=torch.float32).to(device)
        #         fs[7] = torch.tensor(np.concatenate([fs[7], fs0[7]], axis=0), dtype=torch.float32).to(device)
        #         fs[8] = torch.tensor(np.concatenate([fs[8], fs0[8]], axis=0), dtype=torch.float32).to(device)
        #         fs[9] = torch.tensor(np.concatenate([fs[9], fs0[9]], axis=0), dtype=torch.float32).to(device)
        #         fs[10] = torch.tensor(np.concatenate([fs[10], fs0[10]], axis=0), dtype=torch.float32).to(device)
        #         labels = torch.tensor(np.concatenate([labels[:, :, np.newaxis], labels0[:, :, np.newaxis]], axis=0),
        #                               dtype=torch.float32).to(device)
        #
        #         outputs = model(fs[0], fs[1], fs[2], fs[3], fs[4], fs[5], fs[6], fs[7], fs[8], fs[9], fs[10])
        #         # loss = lossfun(outputs, labels)*0.5 + lossfun2(outputs, labels)*0.5
        #         loss = torch.mean(torch.multiply(torch.abs(outputs - labels), labels + 1))
        #
        #         loss.backward()
        #         optimizer.step()
        #
        #         epoch_loss.append(loss.item())
        #
        #     # 训练完成之后使用测试集进行测试
        #     reses = []
        #     batch_size = 1000
        #     pds = []
        #     for i in range(int(np.ceil(test_data[0].shape[0] / batch_size))):
        #         from_idx = i * batch_size
        #         to_idx = (i + 1) * batch_size
        #         res = model(test_data[0][from_idx:to_idx, :, :],
        #                     test_data[1][from_idx:to_idx, :, :],
        #                     test_data[2][from_idx:to_idx, :, :],
        #                     test_data[3][from_idx:to_idx, :, :],
        #                     test_data[4][from_idx:to_idx, :, :],
        #                     test_data[5][from_idx:to_idx, :, :],
        #                     test_data[6][from_idx:to_idx, :, :],
        #                     test_data[7][from_idx:to_idx, :, :],
        #                     test_data[8][from_idx:to_idx, :, :],
        #                     test_data[9][from_idx:to_idx, :, :],
        #                     test_data[10][from_idx:to_idx, :, :]
        #                     )
        #         res = res.detach().cpu().numpy()
        #         res[res > 1] = 1
        #         res[res < 0] = 0
        #         res[res < 0.0001] = 0
        #         reses.append(res)
        #         pds.append(input_Pd_original_testing[from_idx:to_idx, :])
        #     res = np.concatenate(reses, axis=0)[:, :, 0]
        #     pds = np.concatenate(pds, axis=0)
        #     res = res * pds
        #     res[res<0.0001] = 0
        #
        #     label = test_data[11].detach().cpu().numpy() * pds
        #     label[label<0.0001] = 0
        #
        #     error = np.abs(np.sum(res) - np.sum(label))
        #     error2 = np.sum(np.abs(res - label))
        #
        #     p = np.sum(res, axis=1)
        #     idx = np.where(p > 0)[0]
        #     lolp1 = idx.shape[0] / p.shape[0]
        #     EENS1 = np.sum(res) * 100 / pds.shape[0] * 8760
        #
        #     e_lolp = abs(lolp1 - lolp0) / lolp0
        #     e_eens = abs(EENS1 - EENS0) / EENS0
        #
        #     # 保存loss、accuracy值，可用于可视化
        #     losses_train.append(np.mean(epoch_loss))
        #     losses_test.append(np.mean(error))
        #
        #     if epoch % 10 == 0:
        #         temp = np.concatenate([np.array(losses_train)[:, np.newaxis], np.array(losses_test)[:, np.newaxis]],
        #                               axis=-1)
        #         save_csv(temp, saved_folder + "/train_and_test_losses.csv")
        #
        #     # 每一个epoch后打印Loss、Accuracy以及花费的时间
        #     print(
        #         "Scale:{}, Epoch {:03d}: Train Loss: {:.6f}, Test Loss: {:.6f}, {:.6f}, lolp:{:.6f}, e_lolp:{:.6f}, eens:{:.6f}, e_eens:{:.6f}:".format(scale, epoch,
        #                                                                                np.mean(epoch_loss),
        #                                                                                         np.mean(error),error2,
        #                                                                                        lolp1, e_lolp, EENS1, e_eens))
        #
        #     if np.mean(error) < best_model:
        #         best_model = np.mean(error)
        #         # 训练完成后更新保存模型
        #         model_path = saved_folder + "/Model.pth"
        #         torch.save(model.state_dict(), model_path)
        #
        #         save_csv(np.array([[best_model]]), "./M5_test_mae.csv")
        #
        # print('finish!')

    model.load_state_dict(torch.load("./models/M5_20000/Model.pth"))
    reses = []
    batch_size = 1000
    pds = []
    for i in range(int(np.ceil(test_data[0].shape[0] / batch_size))):
        from_idx = i * batch_size
        to_idx = (i + 1) * batch_size
        res = model(test_data[0][from_idx:to_idx, :, :],
                    test_data[1][from_idx:to_idx, :, :],
                    test_data[2][from_idx:to_idx, :, :],
                    test_data[3][from_idx:to_idx, :, :],
                    test_data[4][from_idx:to_idx, :, :],
                    test_data[5][from_idx:to_idx, :, :],
                    test_data[6][from_idx:to_idx, :, :],
                    test_data[7][from_idx:to_idx, :, :],
                    test_data[8][from_idx:to_idx, :, :],
                    test_data[9][from_idx:to_idx, :, :],
                    test_data[10][from_idx:to_idx, :, :]
                    )
        res = res.detach().cpu().numpy()
        res[res > 1] = 1
        res[res < 0] = 0
        res[res < 0.0001] = 0
        reses.append(res)
        pds.append(input_Pd_original_testing[from_idx:to_idx, :])
    res = np.concatenate(reses, axis=0)[:, :, 0]
    pds = np.concatenate(pds, axis=0)
    res = res * pds
    res[res < 0.0001] = 0

    label = test_data[11].detach().cpu().numpy() * pds
    label[label < 0.0001] = 0

    error = np.abs(np.sum(res) - np.sum(label))
    error2 = np.sum(np.abs(res - label))

    p = np.sum(res, axis=1)
    idx = np.where(p > 0)[0]
    lolp1 = idx.shape[0] / p.shape[0]
    EENS1 = np.sum(res) * 100 / pds.shape[0] * 8760

    e_lolp = abs(lolp1 - lolp0) / lolp0
    e_eens = abs(EENS1 - EENS0) / EENS0


    # 每一个epoch后打印Loss、Accuracy以及花费的时间
    print("Test Loss: {:.6f}, {:.6f}, lolp:{:.6f}, e_lolp:{:.6f}, eens:{:.6f}, e_eens:{:.6f}:".format(
            np.mean(error), error2,
            lolp1, e_lolp, EENS1, e_eens))

