
import numpy as np
import json
import copy, time
from pypower.loadcase import loadcase
from pypower.runopf import runopf
from pypower.runpf import runpf
from pypower.ext2int import ext2int
from pypower.int2ext import int2ext
from pypower.makePTDF import makePTDF
from pypower.makeYbus import makeYbus
from pypower.ppoption import ppoption
from pypower.idx_bus import PD, QD, VM, VA, GS, BUS_TYPE, PV, PQ, REF, LAM_P, LAM_Q, MU_VMAX, MU_VMIN, VMAX, VMIN
from pypower.idx_gen import PG, QG, VG, QMAX, QMIN, GEN_BUS, GEN_STATUS, PMAX, PMIN, MU_PMAX, MU_PMIN, MU_QMAX, MU_QMIN
from pypower.idx_brch import PF, QF, BR_STATUS, RATE_A, RATE_B, RATE_C, BR_R, BR_B, BR_X, T_BUS, F_BUS, MU_SF, MU_ST, MU_ANGMIN, MU_ANGMAX, PT
from pyomo.environ import value
from scipy.sparse import csr_matrix


def Nweibull(a, scale, size):
    temp = scale*np.random.weibull(a, size)
    return temp[0, 0]


def Monte_Carlo(ppc):
    bus = ppc['bus']
    windF = ppc['windF']
    PV = ppc['PV']
    # # 对负荷抽样，假定负荷均服从正态分布
    load_e_p = np.random.normal(loc=bus[:, PD], scale=np.abs(bus[:, PD]) * 0.3)
    load_e_q = np.random.normal(loc=bus[:, QD], scale=np.abs(bus[:, QD]) * 0.3)

    # 风速威布尔分布  %假设风电只有 有功
    Pwind = np.zeros((windF.shape[0], 1))
    Vwind = np.zeros((windF.shape[0], 1))

    W_pro = 0   # 风机故障率
    W_break = np.ones((windF.shape[0], 1))
    random_num = np.random.rand(windF.shape[0], 1)   # 生成连续均匀分布的随机数
    W_break[np.where(random_num-W_pro < 0)[0], 0] = 0
    # 判断风机是否被抽掉  风机出力
    for tt in range(windF.shape[0]):
        if windF[tt, 1] == 0:
            Pwind[tt, 0] = 0
        else:
            if W_break[tt, 0] == 0:
                windF[tt, 0] = 0
                Pwind[tt, 0] = 0
            else:
                Vwind[tt, 0] = Nweibull(windF[tt, 2], windF[tt, 3], (1, 1))  # 风速服从威布尔分布
                if Vwind[tt, 0] > windF[tt, 4] and Vwind[tt] <= windF[tt, 5]:
                    Pwind[tt] = windF[tt, 1]*(Vwind[tt]-windF[tt, 4])/(windF[tt, 5]-windF[tt, 4])  # 根据风速算功率
                else:
                    if Vwind[tt]> windF[tt, 5] and Vwind[tt] <= windF[tt, 6]:
                        Pwind[tt] = windF[tt, 1]
                    else:
                        Pwind[tt] = 0
    Ppv = np.zeros((PV.shape[0], 1))
    PV_pro = 0    # 光伏电站故障率
    PV_break = np.ones((PV.shape[0], 1))
    random_num = np.random.rand(PV.shape[0], 1)  # 生成连续均匀分布的随机数
    PV_break[np.where(random_num - PV_pro < 0)[0], 0] = 0

    # 光伏发电 出力
    for ii in range(PV.shape[0]):
        if PV[ii, 1] == 0:
            Ppv[ii] = 0
        else:
            if PV_break[ii] == 0:
                PV[ii, 1] = 0
                Ppv[ii] = 0
            else:
                Ppv[ii] = PV[ii,1] * np.random.beta(PV[ii, 2], PV[ii, 3], (1, 1))   # 光伏功率计算

    PQ_after = np.zeros((bus.shape[0], 1))  # 新能源的功率

    for ii in range(windF.shape[0]):
        PQ_after[int(windF[ii, 0]), 0] = Pwind[ii, 0] + PQ_after[int(windF[ii, 0]), 0]

    for ii in range(PV.shape[0]):
        PQ_after[int(PV[ii, 0]), 0] = Ppv[ii, 0] + PQ_after[int(PV[ii, 0]), 0]

    bus[:, PD] = load_e_p
    bus[:, QD] = load_e_q
    bus[:, PD] = bus[:, PD] - PQ_after[:, 0]

    # 返回更新之后的ppc
    ppc['bus'] = bus
    return ppc


def sampling(mpc, index, br_set_cut, br_set_not_cut):
    ppc = copy.deepcopy(mpc)
    ppc = Monte_Carlo(ppc)  # 对负荷和新能源进行抽样

    bus = ppc['bus']
    branch = ppc['branch']
    gen = ppc['gen']

    # 设置故障线路和机组
    branch[:, BR_STATUS] = 1
    gen[:, GEN_STATUS] = 1

    # temp = index % (len(br_set_cut) + gen.shape[0] + 20)
    # if temp < len(br_set_cut):
    #     branch[br_set_cut[temp], BR_STATUS] = 0
    # elif temp >= len(br_set_cut) and temp < len(br_set_cut) + gen.shape[0]:
    #     gen[temp - len(br_set_cut), GEN_STATUS] = 0
    #     gen[44, GEN_STATUS] = 1  # 平衡机不发生故障

    # 根据故障概率抽取样本
    # 线路状态抽样
    n_branch = branch.shape[0]
    ran = np.random.random((n_branch,))
    ran[ran > 0.01] = 1  # 设置线路的故障率为1%
    ran[ran <= 0.01] = 0
    branch[:, BR_STATUS] = ran
    branch[np.array(br_set_not_cut), BR_STATUS] = 1  # 第14条线路的故障率设置为0，避免系统解裂

    # 机组状态抽样
    n_gen = gen.shape[0]
    ran = np.random.random((n_gen,))
    ran[ran > 0.01] = 1  # 设置机组的故障率为1%
    ran[ran <= 0.01] = 0
    gen[:, GEN_STATUS] = ran
    gen[44, GEN_STATUS] = 1  # 平衡机故障率设置为0

    # 返回更新之后的ppc
    ppc['bus'] = bus
    ppc['gen'] = gen
    ppc['branch'] = branch
    sampled_mpc = ppc
    return sampled_mpc


def task(mpc, pro):
    # 定义存储样本不同变量
    input_Pd = []
    input_Qd = []
    input_G = []
    input_B = []

    input_Gen_status = []
    input_Br_status = []

    output_cut = []

    # 搜索断线导致系统解裂的线路索引
    br_set_not_cut = []
    br_set_cut = list(range(mpc['branch'].shape[0]))
    buses = list(np.concatenate([mpc['branch'][:, F_BUS], mpc['branch'][:, T_BUS]], axis=0))
    buses_ = np.unique(buses)
    for b in buses_:
        if buses.count(b) == 1:
            temp = np.concatenate(
                [np.where(mpc['branch'][:, F_BUS] == b)[0], np.where(mpc['branch'][:, T_BUS] == b)[0]], axis=0)
            # br_set_not_cut.extend(list(np.where(mpc['branch'][:, F_BUS]==b)[0]))
            # br_set_not_cut.extend(list(np.where(mpc['branch'][:, T_BUS]==b)[0]))
            br_set_cut.remove(temp[0])
            br_set_not_cut.append(temp[0])

    # 设置抽样参数
    sample_number = 200
    i = 0
    cut_s = 0
    index = 0
    while i < sample_number:
        index += 1
        # 根据负荷分布和故障率进行抽样
        sampled_mpc = sampling(copy.deepcopy(mpc), index, br_set_cut, br_set_not_cut)
        print("线程:", pro, "样本数:", i, cut_s)

        # 计算切负荷
        try:
            res, status = loadshedding_model(copy.deepcopy(sampled_mpc))
        except:
            continue
        if status:
            i += 1
            # 存储输入特征
            input_Pd.append(sampled_mpc['bus'][:, PD])
            input_Qd.append(sampled_mpc['bus'][:, QD])

            temp_mpc = ext2int(sampled_mpc)
            Ybus, _, _ = makeYbus(temp_mpc['baseMVA'], temp_mpc["bus"], temp_mpc["branch"])
            input_G.append(np.real(Ybus))
            input_B.append(np.imag(Ybus))

            input_Gen_status.append(sampled_mpc['gen'][:, GEN_STATUS])
            input_Br_status.append(sampled_mpc['branch'][:, BR_STATUS])

            # 存储输出特征
            output_cut.append(np.array([value(res.C[j]) for j in range(n_bus)]))

            temp = np.array([value(res.C[j]) for j in range(n_bus)])
            if np.sum(temp) > 0.00001:
                cut_s = cut_s + 1
    return input_Pd, input_Qd, input_G, input_B, output_cut, input_Br_status, input_Gen_status


if __name__ == '__main__':
    from pypower.case39 import case39
    from Loadshedding import loadshedding_model
    from joblib import Parallel, delayed
    from tqdm import tqdm
    import time

    mpc = loadcase('./dataset/case661')

    mpc['baseMVA'] = 100

    mpc['gen'][:, QMIN] = -1000
    mpc['branch'][:, RATE_A] = 1500
    mpc['windF'] = np.array([[2, 20, 2.016, 5.089, 3.5, 15, 25],
                             [3, 20, 2.016, 5.089, 3.5, 15, 25],
                             [7, 20, 2.016, 5.089, 3.5, 15, 25],
                             [14, 20, 2.016, 5.089, 3.5, 15, 25], ])
    mpc['PV'] = np.array([[15, 20, 2.06, 2.5],
                          [23, 20, 2.06, 2.5],
                          [38, 20, 2.06, 2.5],
                          [38, 20, 2.06, 2.5],
                          ])
    # 计算新能源的渗透率
    rate = (np.sum(mpc['windF'][:, 1]) + np.sum(mpc['PV'][:, 1]))/(np.sum(mpc['windF'][:, 1]) + np.sum(mpc['PV'][:, 1]) + np.sum(mpc['gen'][:, 1]))
    print('新能源的渗透率为：', rate)

    n_bus = mpc['bus'].shape[0]
    n_gen = mpc['gen'].shape[0]

    t1 = time.time()
    results = (
        Parallel(n_jobs=8)
        (delayed(task)(mpc, i)
         for i in tqdm(range(100)))
    )

    input_Pd = []
    input_Qd = []
    input_G = []
    input_B = []
    input_Gen = []
    input_Br = []
    output_cut = []
    for i in range(len(results)):
        input_Pd.extend(results[i][0])
        input_Qd.extend(results[i][1])
        input_G.extend(results[i][2])
        input_B.extend(results[i][3])
        output_cut.extend(results[i][4])
        input_Br.extend(results[i][5])
        input_Gen.extend(results[i][6])
    input_Pd = np.array(input_Pd)
    input_Qd = np.array(input_Qd)
    input_G = np.array(input_G)
    input_B = np.array(input_B)
    output_cut = np.array(output_cut)
    input_Br = np.array(input_Br)
    input_Gen = np.array(input_Gen)

    t2 = time.time()
    print('time:', t2-t1)
    print('sample number:', input_Pd.shape[0])

    np.savez('./dataset/case_661_testing_dist[1.0_0.5]_sample[{}].npz'.format(input_Pd.shape[0]),
             input_Pd=input_Pd, input_Qd=input_Qd, input_G=input_G, input_B=input_B, output_cut=output_cut,
             input_Gen=input_Gen, input_Br=input_Br)

