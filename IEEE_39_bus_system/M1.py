# -*- coding: utf-8 -*-
"""
Created on Tuesday, January 9 10:18:04 2024

@author: maosheng
"""
import copy

import numpy as np
from pypower.ext2int import ext2int
from pypower.idx_gen import GEN_BUS, PG, QG, VG, QMAX, QMIN, GEN_BUS, GEN_STATUS, PMAX, PMIN
import os
import torch
import torch.nn as nn
import torch.optim as optim
import csv
import matplotlib.pyplot as plt


def save_csv(data, file_path):
    filename = file_path
    data = np.array(data)
    if len(data.shape)<2:
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
    def __init__(self, input_dim, output_dim, iteration=1):
        super(GCNLayer, self).__init__()
        self.iter = iteration
        self.fc_v1 = nn.Linear(input_dim, output_dim).to(device)
        # self.fc_v2 = nn.Linear(input_dim, output_dim).to(device)

    def forward(self, ef, input_k1):
        ef1 = torch.bmm(input_k1, ef)
        ef_new = torch.relu(self.fc_v1(ef1))

        return ef_new


class PoolLayer(nn.Module):
    def __init__(self,):
        super(PoolLayer, self).__init__()

        self.trans = torch.ones((39, 1)).to(device)

    def forward(self, ef):
        ef_pool = torch.mean(ef, dim=-2, keepdim=True)
        ef_pool = torch.matmul(self.trans, ef_pool)
        ef_new = torch.cat((ef, ef_pool), -1)

        return ef_new


class PIGCNN(nn.Module):
    def __init__(self, ):
        super(PIGCNN, self).__init__()

        self.conv1 = GCNLayer(input_dim=2, output_dim=16)
        self.conv2 = GCNLayer(input_dim=16, output_dim=128)
        self.conv3 = GCNLayer(input_dim=128, output_dim=128)
        self.conv4 = GCNLayer(input_dim=128, output_dim=128)
        self.conv5 = GCNLayer(input_dim=128, output_dim=128)
        # self.flatten = nn.Flatten()
        self.pool = PoolLayer()

        self.predictor = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            # nn.Sigmoid()
            ).to(device)

    def forward(self, ef, input_k1):
        ef = self.conv1(ef, input_k1)
        ef = self.conv2(ef, input_k1)
        ef = self.conv3(ef, input_k1)
        ef = self.conv4(ef, input_k1)
        ef = self.conv5(ef, input_k1)

        # ef = torch.cat((e, f), 2)
        # fs = self.flatten(ef)
        fs = self.pool(ef)
        cut = self.predictor(fs)
        return cut


def load_data(data_dir):
    # 读取原始数据
    data_size = 500000
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
    idx = np.where(temp>0)[0]

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

    # 释放部分内存
    del input_G_ndiag, input_B_ndiag

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
        A = np.abs(input_G[i, :, :]) + np.abs(input_B[i, :, :])
        A[A > 0.0001] = 1
        D = np.diag(np.sum(A, axis=0) ** (-0.5))
        kernel = np.dot(np.dot(D, A), D)
        k1[i, :, :] = kernel

    # 释放部分内存
    del input_G, input_B, input_G_diag, input_B_diag

    inputs = np.concatenate([input_Pd, input_Qd], axis=-1)

    return inputs, output_cut, k1, input_Pd_orginal


def data_generator(train_data, batch_size=15):
    (inputs, k1, output_cut) = train_data

    idx = np.random.permutation(inputs.shape[0])
    for k in range(int(np.ceil(inputs.shape[0] / batch_size))):
        from_idx = k * batch_size
        to_idx = (k + 1) * batch_size
        index = idx[from_idx:to_idx]
        # input_e, input_f, input_G, input_B, input_G_diag, input_B_diag, input_Pd, input_Qd
        inputs_ = [inputs[index, :, :],
                   k1[index, :, :]]
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


    mpc = case39()

    # 导入训练数据
    data_dir = "./dataset/case_39_loadshedding_samples_N-1_dist[1.0_0.5]_sample[35000].npz"
    inputs, output_cut, k1, input_Pd_original = preprocess(mpc, data_dir)
    inputs_eval, output_cut_eval, k1_eval, input_Pd_original_eval = preprocess(mpc, "./dataset/case_39_evaluation_testing_samples[30000].npz")

    # 对数据进行归一化处理
    inputs_, mean_, std_ = Z_Score(inputs)

    # 转化成tensor
    inputs_ = torch.tensor(inputs_, dtype=torch.float32).to(device)
    k1 = torch.tensor(k1, dtype=torch.float32).to(device)
    output_cut = torch.tensor(output_cut, dtype=torch.float32).to(device)

    train_data, test_data = (inputs_[:20000, :, :], k1[:20000, :, :], output_cut[:20000, :, :]), \
                            (inputs_[20000:30000, :, :], k1[20000:30000, :, :], output_cut[20000:30000, :, :])
    input_Pd_original_training, input_Pd_original_testing = input_Pd_original[:20000, :, 0],  input_Pd_original[20000:30000, :, 0]

    # 处理评估数据
    inputs_eval = (inputs_eval - mean_) / std_
    inputs_eval[np.isinf(inputs_eval)] = 0
    inputs_eval[np.isnan(inputs_eval)] = 0

    inputs_eval = torch.tensor(inputs_eval, dtype=torch.float32).to(device)
    k1_eval = torch.tensor(k1_eval, dtype=torch.float32).to(device)
    output_cut_eval = torch.tensor(output_cut_eval, dtype=torch.float32).to(device)
    eval_data = (inputs_eval, k1_eval, output_cut_eval)

    # 构建神经网络，并配置训练参数
    model = PIGCNN()
    lossfun = nn.L1Loss()
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    saved_folder = "./models/M1_new2"
    if (not os.path.exists(saved_folder)):
        os.mkdir(saved_folder)

    # 进行训练
    epoch_num = 2000
    losses_train = []
    losses_test = []
    for epoch in range(epoch_num):  # loop over the dataset multiple times
        epoch_loss = []
        for fs, labels in data_generator(train_data, batch_size=100):
            optimizer.zero_grad()
            outputs = model(fs[0], fs[1])
            loss = lossfun(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())

        # 训练完成之后使用测试集进行测试
        res = model(test_data[0], test_data[1])
        test_loss = lossfun(res, test_data[2])

        # 保存loss、accuracy值，可用于可视化
        losses_train.append(np.mean(epoch_loss))
        losses_test.append(test_loss.item())
        plt.plot(losses_train)
        plt.plot(losses_test)
        plt.legend(['Train', 'Test'])
        plt.savefig(saved_folder+ '/lossFig.png')
        plt.close()

        # 每一个epoch后打印Loss、Accuracy以及花费的时间
        print("Epoch {:03d}: Train Loss: {:.6f}, Test Loss: {:.6f}".format(epoch, np.mean(epoch_loss), test_loss.item()))

        if epoch % 10 == 0:
            #训练完成后更新保存模型
            model_path = saved_folder + "/M1_Model.pth"
            torch.save(model.state_dict(), model_path)

            # 保存训练过程的损失值
            temp = np.concatenate([np.array(losses_train)[:, np.newaxis], np.array(losses_test)[:, np.newaxis]], axis=-1)
            save_csv(temp, saved_folder+"/train_and_test_losses.csv")
    print(' Train finish!')

    # # 训练完成之后使用测试集进行测试
    #
    # model.load_state_dict(torch.load("./models/M1_new2/M1_Model.pth"))
    # res = model(test_data[0], test_data[1])
    # res = res.detach().cpu().numpy()
    # label = test_data[2].detach().cpu().numpy()
    # error = np.abs(res - label)[:, :, 0]
    #
    # temp = np.mean(input_Pd_original_testing, axis=0)
    # idx_0 = np.where(temp != 0)[0]
    # input_Pd_original_testing = input_Pd_original_testing[:, idx_0]
    # error = error[:, idx_0]
    #
    # # np.savez('./results/M1_results.npz', error=error)
    # print('平均绝对百分比误差(MAPE, Mean Absolute Percentage Error)：{}'.format(np.mean(error)))
    # print('平均绝对误差(MAE, Mean Absolute Error)：{}'.format(np.mean(error * input_Pd_original_testing)))
    # print('VAR：{}'.format(np.std(error * input_Pd_original_testing)))
    #
    # error = error * input_Pd_original_testing
    # error[error > 0.01] = 1
    # error[error < 0.01] = 0
    # print('概率精度(PA, Probability accuracy)：{}'.format(1 - np.mean(error)))
    #
    # res = res[:, idx_0]
    # label = label[:, idx_0]
    #
    # def evaluate(res, label):
    #     # 计算切负荷的精确率，定义切负荷为1, positive，不切为0, negative
    #     thre = 0.001
    #     label[label > thre] = 1
    #     label[label <= thre] = 0
    #
    #     res[res > thre] = 1
    #     res[res <= thre] = 0
    #
    #     label = label.reshape(res.shape[0] * res.shape[1], )
    #     res = res.reshape(label.shape[0], )
    #
    #     TP, FP, TN, FN = 0, 0, 0, 0
    #
    #     for i in range(res.shape[0]):
    #         if label[i] == 1:
    #             if res[i] == 1:
    #                 TP += 1
    #             else:
    #                 FN += 1
    #         else:
    #             if res[i] == 0:
    #                 TN += 1
    #             else:
    #                 FP += 1
    #     recall = TP / (TP + FN + 1)
    #     precision = TP / (TP + FP + 1)
    #
    #     return recall, precision
    #
    #
    # recall, precision = evaluate(res, label)
    # print('Recall：{}'.format(recall))
    # print('Precision：{}'.format(precision))
    # print("test finish!")
    #
    # res = model(eval_data[0], eval_data[1])
    # res = res.detach().cpu().numpy()
    # res[res<0.0001] = 0
    # output_cut = res * input_Pd_original_eval
    # p = np.mean(output_cut, axis=1)
    # idx = np.where(p > 0.0001)[0]
    #
    # EENS1 = np.sum(output_cut) * 100 / input_Pd_original_eval.shape[0] * 8760
    #
    # print(idx.shape[0]/p.shape[0], EENS1)
    #
    # print(idx.shape[0] / p.shape[0], EENS1)
    # print("test finish!")

