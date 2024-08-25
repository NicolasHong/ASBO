from bayes_opt_adaptive  import BayesianOptimization
from bayes_opt_adaptive import UtilityFunction
from scipy.optimize import minimize, NonlinearConstraint, Bounds, SR1
from bayes_opt_adaptive.logger import JSONLogger
from bayes_opt_adaptive.event import Events
from bayes_opt_adaptive.util import load_logs
from cases import Case1,Case2,Case3
import numpy as np
import time
import xlwt
from pathlib import Path
import subprocess
import os
import win32com.client as win32
import time
import random
import subprocess
import win32gui
from multiprocessing import Pool, cpu_count, current_process,Lock
import sys
BASE_DIR = Path().absolute()
DIR = BASE_DIR/'Results'
file_location = BASE_DIR/'extractive_distillation'

runs_number = 100
list_realvalue = []

sol = 'ASBO-parallel'
case_name = 'Case4'
runs_number = 10

flowsheet_pass = 0  
list_realvalue = []
def Case4(x, Application):
    global flowsheet_pass,list_realvalue  
    x = [round(num, 3) for num in x]

    max_attempts = 1
    attempts = 0
    result = None
    try:
        while attempts < max_attempts:
            try:
                Application.Tree.FindNode(r"\Data\Streams\SOLVENT\Input\TOTFLOW\MIXED").Value = x[0]
                Application.Tree.FindNode(r"\Data\Blocks\COLUMN\Input\BASIS_D").Value = x[1]
                Application.Tree.FindNode(r"\Data\Blocks\COLUMN\Input\BASIS_RR").Value = x[2]
                Application.Tree.FindNode(r"\Data\Blocks\COL-REC\Input\BASIS_D").Value = x[3]
                Application.Tree.FindNode(r"\Data\Blocks\COL-REC\Input\BASIS_RR").Value = x[4]
                Application.Reinit()
                Application.Engine.Run2(1)
                while Application.Engine.IsRunning == 1:
                    time.sleep(1)
                status_code = Application.Tree.FindNode("\Data\Results Summary\Run-Status").AttributeValue(12)
                if (status_code & 1) == 1:     
                    status =  'Available'
                elif (status_code & 4) == 4:   
                    status =  'Warning'
                elif (status_code & 32) == 32:  
                    status =  'Error'
                else:
                    status =  'Error'
                if status == 'Available' or status == 'Warning':
                    column_con_duty =Application.Tree.FindNode("\Data\Blocks\COLUMN\Output\COND_COST").Value
                    column_reb_duty = Application.Tree.FindNode("\Data\Blocks\COLUMN\Output\REB_COST").Value
                    rec_con_duty = Application.Tree.FindNode("\Data\Blocks\COL-REC\Output\COND_COST").Value
                    rec_reb_duty = Application.Tree.FindNode("\Data\Blocks\COL-REC\Output\REB_COST").Value
                    obj = column_con_duty+rec_con_duty+column_reb_duty+rec_reb_duty
                    purity_heptan = Application.Tree.FindNode("\Data\Streams\C7\Output\MOLEFRAC\MIXED\\N-HEPTAN").Value
                    purity_tolue = Application.Tree.FindNode("\Data\Streams\TOLUENE\Output\MOLEFRAC\MIXED\TOLUE-01").Value
                    result = -obj,[purity_heptan,purity_tolue]
                    print('constraint:',purity_heptan,purity_tolue,obj)
                    break
                else:
                    raise Exception("Convergence failed")
            except Exception as e:
                print(f"Run failed: {e}")
                attempts += 1
                for i in range(len(x)):
                    epsilon = 0.01*int("%d" % attempts)*x[i]
                    x[i] = x[i] + random.choice((-1,1))* epsilon
                # print(f"Number of restarts: {attempts}/{max_attempts}")

    except Exception as e: 
        print("Exception: ", e.args) 
        print("Aspen error, try to close Aspen!")
        Application.Close()

    if result is not None:
        result = result
    else:
        result = -10000,[0,0]
    flowsheet_pass += 1
    return result

def init_aspen():
    version = "37.0"
    Application = win32.Dispatch('Apwn.Document.' + version)

    # 打开 Aspen 文件
    path = os.path.join(file_location, 'extractive_distillation.bkp')
    Application.InitFromArchive2(path)  # 打开文件

    # 设置 AP 用户界面的可见性，1为可见，0为不可见
    Application.Visible = 0
    # 压制对话框的弹出，1为压制；0为不压制
    Application.SuppressDialogs = 1
    return Application
    
def kill_aspen_process():
    """终止 AspenPlus.exe 进程并隐藏错误信息"""
    try:
        subprocess.run('taskkill /F /IM AspenPlus.exe', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        pass  
def close_aspen(application):
    """关闭 AspenPlus 文件"""
    try:
        application.Close()
    except Exception as e:
        print("Error while closing AspenPlus:", e)


def obj_fun(x0,x1,x2,x3,x4):
    aspen_instance = init_aspen()
    Application = aspen_instance
    x = [x0,x1,x2,x3,x4,]
    result = Case4([x0, x1, x2, x3, x4], Application)  
    # close_aspen(aspen_instance)
    return result

if __name__ == "__main__":    

    nc_lb = [0.98,0.98]
    nc_ub = [1,   1 ]
    cons = NonlinearConstraint(obj_fun, nc_lb, nc_ub, hess=SR1(), finite_diff_rel_step=0.0001)

    pbounds = {'x0':(40,100),'x1': (30, 80),'x2': (3,10), 'x3': (20,80),'x4': (1,5)}
    for i in range(1,2):
        start = time.time()
        optimizer = BayesianOptimization(f=obj_fun, pbounds=pbounds,constraint=cons,verbose=2,random_state=i)
        acquisition_function = UtilityFunction(kind="ucb", kappa=2.5, xi=0, kappa_decay=1, kappa_decay_delay=0)
        optimizer.maximize(init_points=100, n_iter=50,p_hyper=[[5, 2, 0.78, 0.95,1e-3],[5, 2, 0.78, 0.95,1e-5],[6, 2, 0.78, 0.95,1e-5],[6.5, 2, 0.80, 0.95,1e-5]],acquisition_function=acquisition_function,allow_duplicate_points=True)
        print(optimizer.max)
        end = time.time()
        print('time:',end-start)
        # subprocess.run('taskkill /F /IM AspenPlus.exe')
