import math 
import numpy as np
import os
import win32com.client as win32
import time
import random
import psutil 

'''Case1'''

def Case1(x):
    def objective(x):
        x1 = x[0]
        x2 = x[1]
        return ((x1+10)/15 - 0.4)**2 + ((x2+15)/30 - 0.3)**2
    def constraint1(x):
        return -2*x[0] + x[1] - 15
    def constraint2(x):
        return (x[0]**2)/2 + 4*x[0] - x[1] - 5
    def constraint3(x):
        return -(x[0] - 4)**2 / 5 - 2*x[1]**2 + 10
    obj = objective(x)
    c1 = constraint1(x)
    c2 = constraint2(x)
    c3 = constraint3(x)

    return obj,c1,c2,c3

    
'''Case2'''

def Case2(x):
    def objective(x):
        x1 = x[0]
        x2 = x[1]
        return -(x1 - 10)**2 - (x2 - 15)**2
    def constraint(x):
        x1 = x[0]
        x2 = x[1]
        return ((x2 - 5.1/(4*np.pi**2)*x1**2 + 5/np.pi*x1 - 6)**2 + 10*(1 - 1/(8*np.pi))*np.cos(x1) + 5)
    obj = objective(x)
    c1 = constraint(x)
    return obj,c1



'''Case3''' # Not the Case 3 in AIChE Paper

def Case3(x):
    def Ackley(x):
        # xx = [x1, x2]
        a = 20
        b = 0.2
        c = 2 * math.pi
        sum1 = 0
        sum2 = 0
        for ii in range(1):
            xi = x[ii]
            sum1 = sum1 + xi**2
            sum2 = sum2 + math.cos(c*xi)
        term1 = -a * math.exp(-b * math.sqrt(sum1/2))
        term2 = -math.exp(sum2/2)
        y = term1 + term2 + a + math.exp(1)
        return y
    def branin_constraint(x):
        branin_val = (
        1 * (x[1] - (5.1 / (4 * np.pi ** 2)) * x[0]**2 + (5 / np.pi) * x[0] - 6)**2 +
        10 * (1 - (1 / (8 * np.pi))) * np.cos(x[0]) + 10 +
        1 * (x[4] - (5.1 / (4 * np.pi ** 2)) * x[3]**2 + (5 / np.pi) * x[3] - 6)**2 +
        10 * (1 - (1 / (8 * np.pi))) * np.cos(x[3]) + 10 - 100)
        return branin_val
    obj = Ackley(x)
    c1 = branin_constraint(x)
    c2 = np.sum(np.sin(2 * np.pi * x)) - np.prod(np.cos(3 * np.pi * x))
    c3 = np.sum(np.exp(-x**2)) + np.prod(np.sin(x))
    c4 = 0.5 * np.sin(2 * np.pi * (x[0] - 2 * x[1])) + x[0] + 2 * x[1] - 1.5 + 3 * x[2] - 2 * x[3] +0.4*x[4]**2
    
    return obj, c1, c2, c3, c4

'''Case3T''' # True Case 3

def Case3T(x):
    def Ackley(x):
        # xx = [x1, x2]
        a = 20
        b = 0.2
        c = 2 * math.pi
        sum1 = 0
        sum2 = 0
        for ii in range(5):
            xi = x[ii]
            sum1 = sum1 + xi**2
            sum2 = sum2 + math.cos(c*xi)
        term1 = -a * math.exp(-b * math.sqrt(sum1/2))
        term2 = -math.exp(sum2/2)
        y = term1 + term2 + a + math.exp(1)
        return y
    def branin_constraint(x):
        branin_val = (
        1 * (x[1] - (5.1 / (4 * np.pi ** 2)) * x[0]**2 + (5 / np.pi) * x[0] - 6)**2 +
        10 * (1 - (1 / (8 * np.pi))) * np.cos(x[0]) + 10 +
        1 * (x[4] - (5.1 / (4 * np.pi ** 2)) * x[3]**2 + (5 / np.pi) * x[3] - 6)**2 +
        10 * (1 - (1 / (8 * np.pi))) * np.cos(x[3]) + 10 - 100)
        return branin_val
    obj = Ackley(x)
    c1 = branin_constraint(x)
    c2 = np.sum(np.sin(2 * np.pi * x)) - np.prod(np.cos(3 * np.pi * x))
    c3 = np.sum(np.exp(-x**2)) + np.prod(np.sin(x))
    c4 = 0.5 * np.sin(2 * np.pi * (x[0] - 2 * x[1])) + x[0] + 2 * x[1] - 1.5 + 3 * x[2] - 2 * x[3] +0.4*x[4]**2
    
    return obj, c1, c2, c3, c4

def Case4(x):
    version = "37.0"
    Application = win32.Dispatch('Apwn.Document.' + version)
    # Application.InitFromArchive2(path)  
    # Application.InitFromArchive2(r'E:\test\extractive_distillation.bkp')  
    # Application.InitFromArchive2(r'E:\test\extractive_distillation_BOBYQA.bkp')  
    # Application.InitFromArchive2(r'E:\test\extractive_distillation_SQP.bkp')  
    Application.Visible = 0 # 1 visable
    Application.SuppressDialogs = 1
    x = [round(num, 3) for num in x]
    for i in range(len(x)):
        globals()[f'x{i}'] = x[i]
    max_attempts = 1
    attempts = 0
    result = None
    try:
        while attempts < max_attempts:
            try:
                Application.Tree.FindNode("\Data\Streams\SOLVENT\Input\TOTFLOW\MIXED").Value = x0
                Application.Tree.FindNode("\Data\Blocks\COLUMN\Input\BASIS_D").Value = x1
                Application.Tree.FindNode("\Data\Blocks\COLUMN\Input\BASIS_RR").Value = x2 
                Application.Tree.FindNode("\Data\Blocks\COL-REC\Input\BASIS_D").Value = x3
                Application.Tree.FindNode("\Data\Blocks\COL-REC\Input\BASIS_RR").Value = x4
                Application.Reinit()
                Application.Engine.Run2()
                # Application.Engine.Run2(1)
                # while Application.Engine.IsRunning == 1:
                #     time.sleep(1)
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
                    result = obj,[purity_heptan,purity_tolue]
                    break
                else:
                    raise Exception("Convergence failed")
            except Exception as e:
                # print(f"Run failed: {e}")
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
        result = 10000,[0,0]
    Application.Close()
    kill_by_pid('AspenPlus.exe')
    time.sleep(1) 
    return result

def kill_by_pid(process_name):
   for proc in psutil.process_iter():
       if proc.name() == process_name:
          p = psutil.Process(proc.pid)
          p.terminate()