'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-05-28 10:43:29
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-05-28 10:51:48
FilePath: \HPBO-V6\parallel-Case3.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from bayes_opt_adaptive  import BayesianOptimization
from bayes_opt_adaptive import UtilityFunction
from scipy.optimize import minimize, NonlinearConstraint, Bounds, SR1
from bayes_opt_adaptive.logger import JSONLogger
from bayes_opt_adaptive.event import Events
from bayes_opt_adaptive.util import load_logs
from cases import Case1,Case2,Case3
import numpy as np
import xlwt


sol = 'ASBO-parallel'
case_name = 'Case4'
runs_number = 10
sol = 'ASBO'
case_name = 'Case3'
runs_number = 100
list_realvalue = []

def func(x1,x2,x3,x4,x5):
    x = np.array([x1,x2,x3,x4,x5])
    obj, c1, c2, c3, c4= Case3(x)
    cons = [c1,c2,c3,c4]
    print('obj',obj,cons)
    return -obj,cons

nc_lb = np.array([-1,-1,-2,0,])
nc_ub = np.array([2,  4, 3,1.5])
cons = NonlinearConstraint(func, nc_lb, nc_ub)
pbounds = {'x1': (-5, 12),'x2': (-20, 22),'x3': (-10, 10),'x4': (-12, 20),'x5': (-15, 18),}
acquisition_function = UtilityFunction(kind="ucb", kappa=2.5, xi=0, kappa_decay=1, kappa_decay_delay=0)


if __name__ == "__main__":    
    for i in range(1):
        optimizer = BayesianOptimization(f=func, pbounds=pbounds,constraint=cons,verbose=2,random_state=i)
        optimizer.maximize(init_points=100, n_iter=200,p_hyper=[[5, 2, 0.78, 0.95,1e-3],[5, 2, 0.75, 0.95,1e-5],[5, 2, 0.78, 0.95,1e-5],[6, 2, 0.78, 0.95,1e-5]],acquisition_function=acquisition_function)
        print(optimizer.max)
        list_realvalue.extend([optimizer.max['target']])
        f = xlwt.Workbook('encoding = utf-8')
        sheet1 = f.add_sheet('sheet1',cell_overwrite_ok=True)
        for i in range(len(list_realvalue)):
            res = list_realvalue[i]
            sheet1.write(i,0,res)
