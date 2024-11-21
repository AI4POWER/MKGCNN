from pyomo.environ import *
import pyomo.environ as pe
# from cvxopt import solvers, matrix
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
import sympy
from sympy import Matrix


def loadshedding_model(mpc):
    # 确定系统的相关参数
    n_bus = mpc['bus'].shape[0]
    n_branch = mpc['branch'].shape[0]
    n_gen = mpc['gen'].shape[0]
    temp_mpc = ext2int(mpc)
    Ybus, Yf, Yt = makeYbus(temp_mpc['baseMVA'], temp_mpc['bus'], temp_mpc['branch'])
    G = np.real(Ybus).todense()
    B = np.imag(Ybus).todense()

    G_ = np.real(Ybus)
    B_ = np.imag(Ybus)

    # 针对线路故障的情况，由于G和B中已经为0了，在mpc还是可以把故障的线路加上,机组故障则直接保留
    temp_mpc = copy.deepcopy(mpc)
    temp_mpc["gen"][:, GEN_STATUS] = temp_mpc["gen"][:, GEN_STATUS] + 1
    temp_mpc["branch"][:, BR_STATUS] = 1
    temp_mpc = ext2int(temp_mpc)
    temp_mpc["gen"][:, GEN_STATUS] = temp_mpc["gen"][:, GEN_STATUS] - 1


    # 确定平衡节点的编号
    slack_bus = np.where(temp_mpc['bus'][:, BUS_TYPE]==3)[0][0]

    # 定义优化模型
    model = ConcreteModel(name='loadshedding')

    # 定义优化变量
    model.Vm = Var(list(range(n_bus)), domain=NonNegativeReals)
    model.Va = Var(list(range(n_bus)), bounds=(-3.14, 3.14))

    model.Pg = Var(list(range(n_gen)), domain=NonNegativeReals)
    model.Qg = Var(list(range(n_gen)))
    model.C = Var(list(range(n_bus)), domain=NonNegativeReals) # 切负荷量的下限为0， 不需要单独设置约束

    # 平衡节点相角限制
    model.Va[slack_bus].fix(0.0)

    # # 设定切负荷的上线
    # pd_max = abs(mpc['bus'][:, PD])/100
    # pd_max[pd_max < 0.0001] = 0
    # pd_max[pd_max >= 0.0001] = 1  # 一个节点最多可以切100MW

    # 切负荷量的上限
    def loadshedding_rule_upper(model, node):
        return  mpc['bus'][node, PD] / 100 - model.C[node] >= 0
    model.loadshedding_upper = Constraint(list(range(n_bus)), rule=loadshedding_rule_upper)

    # 机组有功出力上限
    def pg_generation_rule_upper(model, node):
        return mpc['gen'][node, PMAX] * mpc['gen'][node, GEN_STATUS]/ 100 - model.Pg[node] >= 0
    model.pg_generation_upper = Constraint(list(range(n_gen)), rule=pg_generation_rule_upper)

    # 机组有功出力下限
    def pg_generation_rule_lower(model, node):
        return model.Pg[node] - mpc['gen'][node, PMIN] * mpc['gen'][node, GEN_STATUS] / 100 >= 0
    model.pg_generation_lower = Constraint(list(range(n_gen)), rule=pg_generation_rule_lower)

    # 机组无功出力上限
    def qg_generation_rule_upper(model, node):
        return mpc['gen'][node, QMAX] * mpc['gen'][node, GEN_STATUS] / 100 - model.Qg[node] >= 0
    model.qg_generation_upper = Constraint(list(range(n_gen)), rule=qg_generation_rule_upper)

    # 机组无功出力下限
    def qg_generation_rule_lower(model, node):
        return model.Qg[node] - mpc['gen'][node, QMIN] * mpc['gen'][node, GEN_STATUS] / 100 >= 0
    model.qg_generation_lower = Constraint(list(range(n_gen)), rule=qg_generation_rule_lower)

    # 电压幅值上限约束
    def voltage_magnitude_rule_upper(model, node):
        return mpc['bus'][node, VMAX] - model.Vm[node] >= 0
    model.voltage_magnitude_upper = Constraint(list(range(n_bus)), rule=voltage_magnitude_rule_upper)

    # 电压幅值下限约束
    def voltage_magnitude_rule_lower(model, node):
        return model.Vm[node] - mpc['bus'][node, VMIN] >= 0
    model.voltage_magnitude_lower = Constraint(list(range(n_bus)), rule=voltage_magnitude_rule_lower)

    # 线路潮流FROM TO的约束
    def branch_branch_power_rule_from_to(model, i):
        a, b = int(temp_mpc['branch'][i, F_BUS]), int(temp_mpc['branch'][i, T_BUS])
        r = temp_mpc['branch'][i, BR_R]
        x = temp_mpc['branch'][i, BR_X]
        gij = r / (r * r + x * x)
        bij = -x / (r * r + x * x)
        tij = model.Va[a] - model.Va[b]  # 计算线路上的相角差
        Pij = gij * (model.Vm[a] * model.Vm[a] - model.Vm[a] * model.Vm[b] * cos(tij)) - bij * model.Vm[a] * model.Vm[b] * sin(tij)
        return temp_mpc['branch'][i, RATE_A] / 100 - Pij >= 0
    model.voltage_branch_power_from_to = Constraint(list(range(n_branch)), rule=branch_branch_power_rule_from_to)

    # 线路潮流TO FROM的约束
    def branch_branch_power_rule_to_from(model, i):
        a, b = int(temp_mpc['branch'][i, F_BUS]), int(temp_mpc['branch'][i, T_BUS])
        r = temp_mpc['branch'][i, BR_R]
        x = temp_mpc['branch'][i, BR_X]
        gji = r / (r * r + x * x)
        bji = -x / (r * r + x * x)
        tji = model.Va[b] - model.Va[a]  # 计算线路上的相角差
        Pji = gji * (model.Vm[b] * model.Vm[b] - model.Vm[b] * model.Vm[a] * cos(tji)) - bji * model.Vm[b] * model.Vm[a] * sin(tji)
        return temp_mpc['branch'][i, RATE_A] / 100 - Pji >= 0
    model.voltage_branch_power_to_from = Constraint(list(range(n_branch)), rule=branch_branch_power_rule_to_from)

    # 节点有功平衡约束
    def node_power_rule_p(model, i): # 系数矩阵的写法
        # 计算第i列每行中非零元素的索引
        index = B_.indices[B_.indptr[i]:B_.indptr[i+1]]
        Pin_r = sum(model.Vm[i] * model.Vm[j] * (G[i, j] * cos(model.Va[i] - model.Va[j]) + B[i, j] * sin(model.Va[i] - model.Va[j])) for j in list(index))
        # Pin_r = sum(model.Vm[i] * model.Vm[j] * (G[i, j] * cos(model.Va[i] - model.Va[j]) + B[i, j] * sin(model.Va[i] - model.Va[j])) for j in list(range(n_bus)))
        if i in temp_mpc['gen'][:, GEN_BUS]:
            idx = np.where(i == temp_mpc['gen'][:, GEN_BUS])[0][0]
            Pin_l = model.Pg[idx] - temp_mpc['bus'][i, PD] / 100 + model.C[i]
        else:
            Pin_l = - temp_mpc['bus'][i, PD] / 100 + model.C[i]
        return Pin_r == Pin_l
    model.node_power_p = Constraint(list(range(n_bus)), rule=node_power_rule_p)

    # 节点无功平衡约束
    def node_power_rule_q(model, i):
        index = B_.indices[B_.indptr[i]:B_.indptr[i + 1]]
        Qin_r = sum(model.Vm[i] * model.Vm[j] * (G[i, j] * sin(model.Va[i] - model.Va[j]) - B[i, j] * cos(model.Va[i] - model.Va[j])) for j in list(index))
        # Qin_r = sum(model.Vm[i] * model.Vm[j] * (
        #             G[i, j] * sin(model.Va[i] - model.Va[j]) - B[i, j] * cos(model.Va[i] - model.Va[j])) for j in
        #             list(range(n_bus)))
        if i in temp_mpc['gen'][:, GEN_BUS]:
            idx = np.where(i == temp_mpc['gen'][:, GEN_BUS])[0][0]
            Qin_l = model.Qg[idx] - (temp_mpc['bus'][i, QD] / 100)
        else:
            Qin_l = - temp_mpc['bus'][i, QD] / 100
        return Qin_r == Qin_l
    model.node_power_q = Constraint(list(range(n_bus)), rule=node_power_rule_q)

    # 目标函数：最小化总发电成本
    # def objective_rule(model):
    #     return sum(temp_mpc['gencost'][i, -3] * model.Pg[i] * model.Pg[i]*10000.0 + temp_mpc['gencost'][i, -2] * model.Pg[i]*100 +
    #                temp_mpc['gencost'][i, -1] for i in list(range(n_gen)))
    # 最小切负荷成本
    def objective_rule(model):
        return sum(10000*model.C[i] for i in list(range(n_bus)))
    model.obj = Objective(rule=objective_rule, sense=minimize)

    # 配置优化的模型的求解器
    model.dual = pe.Suffix(direction=pe.Suffix.IMPORT)
    opt = SolverFactory('ipopt')

    # 求解模型
    results = opt.solve(model)
    #使用gams的求解器求解
    # import os
    # os.environ['PATH'] = r'D:\GAMS\win64\24.5' + os.pathsep + os.environ['PATH']
    # solver = SolverFactory("gams")
    # solver.options["solver"] = "CONOPT"
    #
    # print('开始求解')
    # t1 = time.time()
    # results = solver.solve(model)
    # t2 = time.time()
    # print(t2 - t1)


    # def branch_branch_power_rule_from_to(model, i):
    #     a, b = int(temp_mpc['branch'][i, F_BUS]), int(temp_mpc['branch'][i, T_BUS])
    #     r = temp_mpc['branch'][i, BR_R]
    #     x = temp_mpc['branch'][i, BR_X]
    #     gij = r/(r*r + x*x)
    #     bij = -x/(r*r + x*x)
    #
    #     # gij, bij = -G_[a, b], -B_[a, b]  # 支路潮流方程用的是支路电导电纳
    #     tij = model.Va[a] - model.Va[b]  # 计算线路上的相角差
    #     Pij = gij * (model.Vm[a] * model.Vm[a] - model.Vm[a] * model.Vm[b] * cos(tij)) - bij * model.Vm[a] * model.Vm[b] * sin(tij)
    #     print(Pij)
    #     print(model.Vm[a], value(model.Vm[a]))
    #     print(model.Vm[b], value(model.Vm[b]))
    #     print(model.Va[a], value(model.Va[a]))
    #     print(model.Va[b], value(model.Va[b]))
    #     return value(Pij)
    # print(branch_branch_power_rule_from_to(model, 651))

    # print(results)
    # print("节点电压幅值：", [value(model.Vm[i]) for i in range(n_bus)])
    # # print("机组出力：", [value(model.Pg[i]) for i in range(n_gen)])
    # print("切负荷量：", [value(model.C[i]) for i in range(n_bus)])
    # print("最优目标值：", value(model.obj))

    # print(model.dual.display())

    if results.solver.termination_condition == TerminationCondition.optimal:
    # if results.solver.termination_condition == TerminationCondition.locallyOptimal:
        return model, 1
    else:
        return model, 0


if __name__ == '__main__':
    from pypower.case24_ieee_rts import case24_ieee_rts
    from pypower.case118 import case118

    mpc = case118()
    mpc['branch'][:, RATE_A] = 160

    ppopt = ppoption(None, PF_DC=False, OPF_ALG_DC=200)
    ppopt['OPF_FLOW_LIM'] = 1  # 设置线路有功潮流约束
    ppopt['VERBOSE'] = 0  # 不输出结果
    ppopt['OUT_ALL'] = -1  # 关闭所有输出
    res = runopf(mpc, ppopt)

    mpc['bus'][0, PD] = 3000
    print("load", mpc['bus'][0, PD])
    model, _ = loadshedding_model(copy.deepcopy(mpc))

    mpc['bus'][0, PD] = 3500
    print("load", mpc['bus'][0, PD])
    model, _ = loadshedding_model(copy.deepcopy(mpc))
    # mpc['bus'][0, PD] = 800
    # print("load", mpc['bus'][0, PD])
    # model, _ = loadshedding_model(copy.deepcopy(mpc))
    #
    # mpc['bus'][0, PD] = 600
    # print("load", mpc['bus'][0, PD])
    # model, _ = loadshedding_model(copy.deepcopy(mpc))
    # mpc['bus'][0, PD] = 450
    # print("load", mpc['bus'][0, PD])
    # model, _ = loadshedding_model(copy.deepcopy(mpc))

    # for k in model.qg_generation_lower.keys():
    #     print(k, model.qg_generation_lower[k], model.dual[model.qg_generation_lower[k]])
    #
    # for k in model.qg_generation_upper.keys():
    #     print(k, model.qg_generation_upper[k], model.dual[model.qg_generation_upper[k]])
    #
    # for k in model.voltage_magnitude_lower.keys():
    #     print(k, model.voltage_magnitude_lower[k], model.dual[model.voltage_magnitude_lower[k]])
    #
    # print(model.dual.display())
    print(123)

