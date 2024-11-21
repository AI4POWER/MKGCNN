# -*- coding: utf-8 -*-
"""
Created on Tuesday, January 9 10:18:04 2024

@author: maosheng
"""
import time

import numpy as np
from pypower.ext2int import ext2int
from pypower.idx_gen import GEN_BUS, PG, QG, VG, QMAX, QMIN, GEN_BUS, GEN_STATUS, PMAX, PMIN
import os
import torch
import torch.nn as nn
import torch.optim as optim
import csv, copy
# import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.sparse import csr_matrix, hstack, vstack


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


def attentionDot(a, x):
    b = int(x.shape[0] / 661)
    F = x.shape[1]
    x = torch.reshape(x, ((b, 661, F)))
    x = a * x
    return torch.reshape(x, ((b * 661, F)))


class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, is_output=False):
        super(GCNLayer, self).__init__()
        self.is_output = is_output
        # self.fc_v1 = nn.Linear(input_dim * 3 * 2, input_dim).to(device)
        # output_dim = int(output_dim/4)
        self.fc_v1 = nn.Linear(input_dim * 5, output_dim).to(device)
        self.fc_v2 = nn.Linear(input_dim * 5, output_dim).to(device)

        self.atten_e = nn.Linear(input_dim, 1).to(device)
        self.atten_f = nn.Linear(input_dim, 1).to(device)

        self.pool = PoolLayer2().to(device)

    def forward(self, e, f, k1, k2, G_ndiag, B_ndiag, G_diag, B_diag, Pd, Qd):
        base = e * e + f * f + 0.1

        e_G_ndiag = torch.sparse.mm(G_ndiag, e)
        f_G_ndiag = torch.sparse.mm(G_ndiag, f)
        e_B_ndiag = torch.sparse.mm(B_ndiag, e)
        f_B_ndiag = torch.sparse.mm(B_ndiag, f)

        alpha = Pd * e / base + Qd * f / base - e_G_ndiag - f_B_ndiag
        beta = Qd * e / base - Pd * f / base + f_G_ndiag + e_B_ndiag

        base_gb = G_diag * G_diag + B_diag * B_diag
        e3 = alpha * G_diag / base_gb + beta * B_diag / base_gb
        f3 = beta * G_diag / base_gb - alpha * B_diag / base_gb

        base1 = e_G_ndiag - f_B_ndiag
        base2 = f_G_ndiag + e_B_ndiag
        # base4 = base1 * base1 + base2 * base2
        base4 = base_gb # G_diag * G_diag + B_diag * B_diag

        new_e = ((Pd - (e * e + f * f) * G_diag) * base1 + (Qd + (e * e + f * f) * B_diag) * base2) / base4
        new_f = ((Pd - (e * e + f * f) * G_diag) * base2 - (Qd + (e * e + f * f) * B_diag) * base1) / base4

        e1 = torch.sparse.mm(k1, e)
        f1 = torch.sparse.mm(k1, f)

        e2 = torch.sparse.mm(k2, e)
        f2 = torch.sparse.mm(k2, f)

        a1 = torch.sigmoid(self.atten_e(self.pool(e3)))
        a2 = torch.sigmoid(self.atten_e(self.pool(new_e)))
        a3 = torch.sigmoid(self.atten_e(self.pool(e1)))
        a4 = torch.sigmoid(self.atten_e(self.pool(e2)))
        a_base = a1 + a2 + a3 + a4 + 0.0001
        a1 = a1 / a_base
        a2 = a2 / a_base
        a3 = a3 / a_base
        a4 = a4 / a_base

        e_new = torch.cat((attentionDot(a1, e3), attentionDot(a2, new_e), attentionDot(a3, e1), attentionDot(a4, e2), e), -1)
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

        f_new = torch.cat((attentionDot(b1, f3), attentionDot(b2, new_f), attentionDot(b3, f1), attentionDot(b4, f2), f), -1)
        # e_new = torch.relu(self.fc_v1(e_new))
        # f_new = torch.relu(self.fc_v2(f_new))

        if self.is_output:
            e_new = torch.sigmoid(self.fc_v1(e_new))
            f_new = torch.sigmoid(self.fc_v2(f_new))
        else:
            e_new = torch.tanh(self.fc_v1(e_new))
            f_new = torch.tanh(self.fc_v2(f_new))

        return e_new, f_new


class PoolLayer2(nn.Module):
    def __init__(self, ):
        super(PoolLayer2, self).__init__()

    def forward(self, ef):
        b = int(ef.shape[0] / 661)
        F = ef.shape[1]
        ef = torch.reshape(ef, ((b, 661, F)))
        ef_pool = torch.mean(ef, dim=-2, keepdim=True)
        return ef_pool


class PoolLayer(nn.Module):
    def __init__(self, ):
        super(PoolLayer, self).__init__()

        self.trans = torch.ones((661, 1)).to(device)

    def forward(self, ef):
        b = int(ef.shape[0] / 661)
        F = ef.shape[1]
        ef = torch.reshape(ef, ((b, 661, F)))

        ef_pool = torch.mean(ef, dim=-2, keepdim=True)
        ef_pool = torch.matmul(self.trans, ef_pool)
        ef_new = torch.cat((ef, ef_pool), -1)

        ef_new = torch.reshape(ef_new, ((b*661, 2*F)))
        return ef_new


class PIGCNN(nn.Module):
    def __init__(self, ):
        super(PIGCNN, self).__init__()

        self.conv1 = GCNLayer(input_dim=1, output_dim=16)
        self.conv2 = GCNLayer(input_dim=16, output_dim=256)
        self.conv3 = GCNLayer(input_dim=256, output_dim=256)
        self.conv4 = GCNLayer(input_dim=256, output_dim=256)
        # self.conv5 = GCNLayer(input_dim=128, output_dim=128)
        # self.flatten = nn.Flatten()
        self.pool1 = PoolLayer()

        self.predictor = nn.Sequential(
            # nn.Linear(512, 500),
            # nn.ReLU(),
            nn.Linear(512*2, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            # nn.Sigmoid()
        ).to(device)

    def forward(self, inputs):
        e, f, k1, k2, G_ndiag, B_ndiag, G_diag, B_diag, Pd, Qd = inputs
        e, f = self.conv1(e, f, k1, k2, G_ndiag, B_ndiag, G_diag, B_diag, Pd, Qd)
        # e = PIGCNN.pool(e)
        # f = PIGCNN.pool(f)
        e, f = self.conv2(e, f, k1, k2, G_ndiag, B_ndiag, G_diag, B_diag, Pd, Qd)
        # e = PIGCNN.pool(e)
        # f = PIGCNN.pool(f)
        e, f = self.conv3(e, f, k1, k2, G_ndiag, B_ndiag, G_diag, B_diag, Pd, Qd)
        e, f = self.conv4(e, f, k1, k2, G_ndiag, B_ndiag, G_diag, B_diag, Pd, Qd)
        # e, f = self.conv5(e, f, k1, k2, G_ndiag, B_ndiag, G_diag, B_diag, Pd, Qd)

        fs = torch.cat((e, f), -1)
        fs = self.pool1(fs)
        # fs = (e+f)/2
        cut = self.predictor(fs)
        return cut

    @staticmethod
    def pool(x):
        b = int(x.shape[0] / 661)
        F = x.shape[1]
        x = torch.reshape(x, ((b, 661, F)))

        x_mean = torch.mean(x, dim=-2, keepdim=True)
        x_ = x - x_mean

        x_ = torch.reshape(x_, ((b * 661, F)))
        return x_




def load_data(data_dir, d=30000):
    # 读取原始数据
    data = np.load(data_dir, allow_pickle=True)
    input_Pd = data['input_Pd'][:d, :, :]
    input_Qd = data['input_Qd'][:d, :, :]
    input_Pg = data['input_Pg'][:d, :, :]
    input_Qg = data['input_Qg'][:d, :, :]
    output_cut = data['output_cut'][:d, :]

    input_Gen = data['input_Gen'][:d, :]
    input_G_diag = data['input_G_diag'][:d, :, :]
    input_B_diag = data['input_B_diag'][:d, :, :]
    input_G_ndiag = data['input_G_ndiag'][:d]
    input_B_ndiag = data['input_B_ndiag'][:d]
    k1 = data['k1'][:d]
    k2 = data['k2'][:d]

    print('完成数据读取。')

    node_num = input_Pd.shape[1]
    sample_num = input_Pd.shape[0]

    # 处理没有不能直接读取的数据
    input_e = np.ones((sample_num, node_num, 1))
    input_f = np.zeros((sample_num, node_num, 1))

    # 对数据进行预处理
    output_cut[output_cut < 0.000001] = 0
    output_cut = output_cut[:, :, np.newaxis]
    input_Gen = input_Gen
    input_Pd = input_Pd / 100
    input_Qd = input_Qd / 100

    # 对切负荷量进行归一化处理
    output_cut = output_cut / input_Pd
    output_cut[np.isnan(output_cut)] = 0
    output_cut[np.isinf(output_cut)] = 0
    input_Pd_orginal = copy.deepcopy(input_Pd)

    # 计算节点注入功率
    input_Pd = input_Pg - input_Pd
    input_Qd = input_Qg - input_Qd

    return (input_e, input_f, k1, k2, input_G_diag, input_B_diag, input_G_ndiag, input_B_ndiag,
            input_Pd, input_Qd, output_cut), input_Pd_orginal


def SparseConca(x):
    b_size = x.shape[0]

    if b_size==1:
        return x[0, :, :]
    else:
        temp = x[-1]
        m_shape = temp.shape[0]
        rows = temp.row + (b_size - 1)*m_shape
        cols = temp.col + (b_size - 1)*m_shape
        values = temp.data
        for i in range(b_size-1):
            temp = x[i]
            rows = np.concatenate([rows, temp.row + i * m_shape], axis=0)
            cols = np.concatenate([cols, temp.col + i * m_shape], axis=0)
            values = np.concatenate([values, temp.data], axis=0)
        x = torch.sparse_coo_tensor(indices=torch.LongTensor(np.array([rows, cols])),
                                    values=torch.tensor(values, dtype=torch.float32),
                                    size=(m_shape * b_size, m_shape * b_size)).to(device)
        return x
def deleteBatch(inputs, outputs):
    (e, f, k1, k2, G_ndiag, B_ndiag, G_diag, B_diag, Pd, Qd) = inputs
    batch_size = e.shape[0]

    e = torch.tensor(np.concatenate(np.split(e, batch_size, 0), axis=1)[0, :, :], dtype=torch.float32).to(device)
    f = torch.tensor(np.concatenate(np.split(f, batch_size, 0), axis=1)[0, :, :], dtype=torch.float32).to(device)
    G_diag = torch.tensor(np.concatenate(np.split(G_diag, batch_size, 0), axis=1)[0, :, :], dtype=torch.float32).to(device)
    B_diag = torch.tensor(np.concatenate(np.split(B_diag, batch_size, 0), axis=1)[0, :, :], dtype=torch.float32).to(device)
    Pd = torch.tensor(np.concatenate(np.split(Pd, batch_size, 0), axis=1)[0, :, :], dtype=torch.float32).to(device)
    Qd = torch.tensor(np.concatenate(np.split(Qd, batch_size, 0), axis=1)[0, :, :], dtype=torch.float32).to(device)
    outputs = torch.tensor(np.concatenate(np.split(outputs, batch_size, 0), axis=1)[0, :, :], dtype=torch.float32).to(device)

    k1 = SparseConca(k1)
    k2 = SparseConca(k2)
    G_ndiag = SparseConca(G_ndiag)
    B_ndiag = SparseConca(B_ndiag)

    inputs = (e, f, k1, k2, G_ndiag, B_ndiag, G_diag, B_diag, Pd, Qd)

    return inputs, outputs

def data_generator(train_data, input_Pd_original_training, batch_size=15, train=True):
    (e, f, k1, k2, G_diag, B_diag, G_ndiag, B_ndiag, Pd, Qd, output_cut) = train_data

    idx = np.random.permutation(e.shape[0])
    for k in range(int(np.ceil(e.shape[0] / batch_size))):
        from_idx = k * batch_size
        to_idx = (k + 1) * batch_size
        index = idx[from_idx:to_idx]
        # e, f, inputs, k1, k2, G_ndiag, B_ndiag, G_diag, B_diag, Pd, Qd
        outputs_ = output_cut[index, :, :]
        pd_ = input_Pd_original_training[index, :, :]

        if train:
            # 进行扰动添加新样本
            labels_temp = outputs_ * pd_
            disturbance = (np.random.random(outputs_.shape) - 0.5) / 0.5 * labels_temp
            labels_temp = labels_temp + disturbance
            pd_new = pd_ + disturbance
            input_pd_ = Pd[index, :, :] + pd_ - pd_new
            labels_temp = labels_temp / pd_new
            labels_temp[np.isnan(labels_temp)] = 0
            labels_temp[np.isinf(labels_temp)] = 0

            # 构建输入神经网络的输入特征和输出特征
            inputs = [np.concatenate([e[index, :, :], e[index, :, :]], axis=0),
                      np.concatenate([f[index, :, :], f[index, :, :]], axis=0),
                      np.concatenate([k1[index], k1[index]], axis=0),
                      np.concatenate([k2[index], k2[index]], axis=0),
                      np.concatenate([G_ndiag[index], G_ndiag[index]], axis=0),
                      np.concatenate([B_ndiag[index], B_ndiag[index]], axis=0),
                      np.concatenate([G_diag[index, :, :], G_diag[index, :, :]], axis=0),
                      np.concatenate([B_diag[index, :, :], B_diag[index, :, :]], axis=0),
                      np.concatenate([Pd[index, :, :], input_pd_], axis=0),
                      np.concatenate([Qd[index, :, :], Qd[index, :, :]], axis=0)
                      ]
            outputs_ = np.concatenate([outputs_, labels_temp], axis=0)
            pd_ = np.concatenate([pd_, pd_new], axis=0)
        else:
            inputs = [e[index, :, :],
                      f[index, :, :],
                      k1[index],
                      k2[index],
                      G_ndiag[index],
                      B_ndiag[index],
                      G_diag[index, :, :],
                      B_diag[index, :, :],
                      Pd[index, :, :],
                      Qd[index, :, :]]
        # 将数据中batch取消，然后在图节点的维度上拼接起来，然后转化为tensor
        inputs, outputs = deleteBatch(inputs, outputs_)

        yield inputs, outputs, pd_


def add_samples(train_data, input_Pd_original_training):
    # 扩充切负荷样本
    training_data_size = train_data[0].shape[0]
    (e, f, k1, k2, G_diag, B_diag, G_ndiag, B_ndiag, Pd, Qd, output_cut) = train_data
    temp = np.sum(output_cut[:, :, 0], axis=-1)
    idx = np.where(temp != 0)[0]
    labels = output_cut[idx, :, :] * input_Pd_original_training[idx, :, :]
    disturbance = (np.random.random(labels.shape) - 0.5) / 0.5 * labels
    labels = labels + disturbance
    pd_new = input_Pd_original_training[idx, :, :] + disturbance
    input_Pd_original_training = np.concatenate([input_Pd_original_training, pd_new], axis=0)

    labels = labels / pd_new
    labels[np.isnan(labels)] = 0
    labels[np.isinf(labels)] = 0
    PD_new = Pd[idx, :, :] + input_Pd_original_training[idx, :, :] - pd_new

    train_data = [np.concatenate([e, e[idx, :, :]], axis=0),
                  np.concatenate([f, f[idx, :, :]], axis=0),
                  np.concatenate([k1, k1[idx]], axis=0),
                  np.concatenate([k2, k2[idx]], axis=0),
                  np.concatenate([G_diag, G_diag[idx, :, :]], axis=0),
                  np.concatenate([B_diag, B_diag[idx, :, :]], axis=0),
                  np.concatenate([G_ndiag, G_ndiag[idx]], axis=0),
                  np.concatenate([B_ndiag, B_ndiag[idx]], axis=0),
                  np.concatenate([Pd, PD_new], axis=0),
                  np.concatenate([Qd, Qd[idx, :, :]], axis=0),
                  np.concatenate([output_cut, labels], axis=0)
                  ]
    return train_data, input_Pd_original_training


def separate_samples(train_data, input_Pd_original_training, scale):
    train_data_out = [train_data[0][:scale, :, :],
                      train_data[1][:scale, :, :],
                      train_data[2][:scale],
                      train_data[3][:scale],
                      train_data[4][:scale, :, :],
                      train_data[5][:scale, :, :],
                      train_data[6][:scale],
                      train_data[7][:scale],
                      train_data[8][:scale, :, :],
                      train_data[9][:scale, :, :],
                      train_data[10][:scale, :, :]]
    input_Pd_original_training_out = input_Pd_original_training[:scale, :]
    return train_data_out, input_Pd_original_training_out


if __name__ == '__main__':
    # 配置环境变量是否使用GPU
    from pypower.loadcase import loadcase

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    ################################# set device ##################################
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
    # from pypower.case118 import case118

    mpc = loadcase('./dataset/case661')
    # mpc['branch'][:, RATE_A] = 160

    # 导入训练数据
    print("导入训练数据")
    data_dir = "./dataset/case_661_sparse_Data_training.npz"
    train_data0, input_Pd_original_training0 = load_data(data_dir, d=30000)

    print("导入测试数据")
    data_dir = "./dataset/case_661_sparse_Data_testing.npz"
    test_data, input_Pd_original_testing = load_data(data_dir, d=20000)

    output_cut_eval = test_data[10] * input_Pd_original_testing
    output_cut_eval[output_cut_eval < 0.01] = 0
    p = np.sum(output_cut_eval, axis=1)
    idx = np.where(p > 0)[0]
    lolp0 = idx.shape[0] / p.shape[0]
    EENS0 = np.sum(output_cut_eval) * 100 / input_Pd_original_testing.shape[0] # * 8760

    print(lolp0, EENS0)

    # 构建神经网络，并配置训练参数
    model = PIGCNN()


    lossfun = nn.L1Loss()
    lossfun2 = nn.MSELoss()
    torch.save(model.state_dict(), './models/original_model.pth')

    testing_scale = [20000]

    testing_errors = []
    for scale in testing_scale:
        saved_folder = "./models/M5_final"
        if (not os.path.exists(saved_folder)):
            os.mkdir(saved_folder)

        # 初始化定义一个优化器
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)
        # model.load_state_dict(torch.load("./models/Model.pth"))
        model.load_state_dict(torch.load("./models/M5_final/Model_{}.pth".format(7)))

        # 切分训练样本
        train_data, input_Pd_original_training = separate_samples(train_data0, input_Pd_original_training0, scale)

        # 扩充样本集
        # train_data, input_Pd_original_training = add_samples(train_data, input_Pd_original_training)
        # train_data, input_Pd_original_training = add_samples(train_data, input_Pd_original_training)

        # 进行训练
        epoch_num = 1000
        losses_train = []
        losses_test = []
        best_model = 10000000
        for epoch in range(epoch_num):  # loop over the dataset multiple times
            epoch_loss = []
            if epoch == 800:
                optimizer = optim.Adam(model.parameters(), lr=0.0001)
            # 扰动学习
            # train_data_temp, input_Pd_original_training_temp = add_samples(train_data, input_Pd_original_training)
            # for fs, labels, pd_ in tqdm(data_generator(train_data_temp, input_Pd_original_training_temp, batch_size=50, train=False)):
            for fs, labels, pd_ in tqdm(data_generator(test_data, input_Pd_original_testing, batch_size=50, train=False)):
                optimizer.zero_grad()
                # t1 = time.time()
                outputs = model(fs)
                # t2 = time.time()
                # loss = torch.mean(torch.multiply(torch.abs(outputs - labels), labels + 1))
                loss = lossfun2(outputs*10, labels*10)/10 + lossfun(outputs, labels)
                # print(loss.item())
                loss.backward()
                optimizer.step()
                # t3 = time.time()
                # print(t2-t1, t3-t2)

                epoch_loss.append(loss.item())

            # 训练完成之后使用测试集进行测试
            reses = []
            pds = []
            label_output = []
            for fs, labels, pd_ in tqdm(data_generator(test_data, input_Pd_original_testing, batch_size=50, train=False)):
                res = model(fs)
                res = torch.reshape(res, (-1, 661, 1))
                res = res.detach().cpu().numpy()
                res[res > 1] = 1
                res[res < 0] = 0
                res[res < 0.0001] = 0
                reses.append(res)
                pds.append(pd_)

                labels = torch.reshape(labels, (-1, 661, 1))
                label_output.append(labels.detach().cpu().numpy())

            res = np.concatenate(reses, axis=0)[:, :, 0]
            pds = np.concatenate(pds, axis=0)[:, :, 0]
            label_output = np.concatenate(label_output, axis=0)[:, :, 0]
            res = res * pds
            label = label_output * pds
            res[res < 0.01] = 0
            label[label < 0.01] = 0

            error = np.abs(np.sum(res) - np.sum(label))
            error2 = np.sum(np.abs(res - label))

            p = np.sum(res, axis=1)
            idx = np.where(p >= 0.01)[0]
            lolp1 = idx.shape[0] / p.shape[0]
            EENS1 = np.sum(res) * 100 / pds.shape[0]  # * 8760

            e_lolp = abs(lolp1 - lolp0) / lolp0
            e_eens = abs(EENS1 - EENS0) / EENS0

            # 保存loss、accuracy值，可用于可视化
            losses_train.append(np.mean(epoch_loss))
            losses_test.append([np.mean(error), error2, lolp1, e_lolp, EENS1, e_eens])

            model_path = "./models/M5_final/Model_{}.pth".format(epoch)
            torch.save(model.state_dict(), model_path)
            # torch.save(model.state_dict(), './models/Model.pth')

            temp = np.concatenate([np.array(losses_train)[:, np.newaxis],
                                   np.array(losses_test)],
                                  axis=-1)
            save_csv(temp, "./models/M5_final/train_and_test_losses.csv")

            # 每一个epoch后打印Loss、Accuracy以及花费的时间
            print(
                "Epoch {:03d}: Train Loss: {:.6f}, Test Loss: {:.6f},{:.6f}, lolp:{:.6f}, e_lolp:{:.6f}, eens:{:.6f}, e_eens:{:.6f}:".format(
                    epoch,
                    np.mean(epoch_loss),
                    np.mean(error), np.mean(error2),
                    lolp1, e_lolp, EENS1, e_eens))
            #
            # if np.mean(error) < best_model:
            #     best_model = np.mean(error)
            #     # 训练完成后更新保存模型
            #     model_path = saved_folder + "/Model.pth"
            #     torch.save(model.state_dict(), model_path)
            #
            #     save_csv(np.array([[best_model]]), "./M5_test_mae_sparse3.csv")

        print('finish!')

    # 训练完成之后使用测试集进行测试
    model.load_state_dict(torch.load("./models/M5_final/Model_{}.pth".format(410)))

    reses = []
    pds = []
    label_output = []
    t1 = time.time()
    for fs, labels, pd_ in data_generator(test_data, input_Pd_original_testing, batch_size=100, train=False):
        res = model(fs)
        res = torch.reshape(res, (-1, 661, 1))
        res = res.detach().cpu().numpy()
        res[res > 1] = 1
        res[res < 0] = 0
        res[res < 0.0001] = 0
        reses.append(res)
        pds.append(pd_)

        labels = torch.reshape(labels, (-1, 661, 1))
        label_output.append(labels.detach().cpu().numpy())
    t2 = time.time()

    res = np.concatenate(reses, axis=0)[:, :, 0]
    pds = np.concatenate(pds, axis=0)[:, :, 0]
    label_output = np.concatenate(label_output, axis=0)[:, :, 0]
    res = res * pds
    label = label_output * pds
    res[res < 0.01] = 0
    label[label < 0.01] = 0

    error = np.abs(np.sum(res) - np.sum(label))
    error2 = np.sum(np.abs(res - label))

    p = np.sum(res, axis=1)
    idx = np.where(p >= 0.05)[0]
    lolp1 = idx.shape[0] / p.shape[0]
    EENS1 = np.sum(res) * 100 / pds.shape[0]  # * 8760

    e_lolp = abs(lolp1 - lolp0) / lolp0
    e_eens = abs(EENS1 - EENS0) / EENS0

    print(lolp0, lolp1, e_lolp, EENS0, EENS1, e_eens)
    print(t2-t1)

