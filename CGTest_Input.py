# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 14:51:35 2019

@author: Marcezar
"""

import os
import pickle

# Import Numpy and Scipy
import numpy as np
import numpy.random as npr
import scipy.linalg as spl

# Import self implemented code
from matrices import Preconditioner,Prolongator
from functions import cg
from operators import ADEF1,ADEF2,BNN

# Import time measurement
import time

import pandas as pd

from IPython.display import clear_output





def computeCG(A,x_sol,mode,prol,k=6,tol=1e-10,maxiter=None,omg=1.0):
    if mode not in ['ADEF1','ADEF2','BNN1','BNN2','None']:
        raise NameError("Invalid input for mode.")
    
    if prol not in ['def','prec','I','100','121',
                    '2332','11v','opt','rand','optSI']:
        raise NameError("Invalid input for prolongator.")
    
    n = A.shape[0]
    b = A@x_sol

    if prol == 'def':
        print("Computing CG.")
        t_0           = time.perf_counter()
        [x, res, err] = cg(A, b, tol=tol, maxiter=maxiter ,x_exact=x_sol)
        t_cg          = time.perf_counter() - t_0
        k             = None
        t_U           = 0
    
    else:
        
        # Construct preconditioner
        Prec = Preconditioner(A, form='Diag', omg=omg)
        if prol == 'prec':
            P   = Prec
            k   = None
            t_U = 0
            print("Computing preconditioned CG.")
            

            
        else:
            print("Constructing Deflation Operator P"+prol+".")
            
            t_0 = time.perf_counter()
            
            # Construct prolongation matrix U
            if   prol == 'I':
                U = Prolongator.U_I(n, k)
            elif prol == '100':
                U = Prolongator.U_100(n)
            elif prol == '121':
                U = Prolongator.U_121(n)
            elif prol == '2332':
                U = Prolongator.U_2332(n)
            elif prol == '11v':
                U = Prolongator.U_111v(n, k)
            elif prol == 'opt':
                U = Prolongator.U_opt(n, k, M=Prec@A)
            elif prol == 'optSI':
                U = Prolongator.U_optSI(n, k, M=Prec@A)
            elif prol == 'rand':
                U = Prolongator.U_rand(n, k)
            t_U = time.perf_counter() - t_0
            
            # Construct adepted deflation operator P
            if   mode == 'ADEF1':
                P = ADEF1(A, U, Prec)
            elif mode == 'ADEF2':
                P = ADEF2(A, U, Prec)
            elif mode in ['BNN1','BNN2']:
                P = BNN(A, U, Prec)
                
            k = U.shape[1]
            print("Computing CG with P_%s (k=%i, ω=%4.2f)." %(prol,k,omg))     
            
        x0 = np.zeros(n)
        t_0           = time.perf_counter()
        
        # testing equality of iterations of ADEF2 and BNN with same 
        # starting vector
        if mode == 'ADEF2' or mode == 'BNN2':
            x0 = P.get_starting_vector(b,x0)
    
        [x, res, err] = cg(A, b, x0, tol=tol, maxiter=maxiter,
                           M1=P ,x_exact=x_sol)
        
        t_cg          = time.perf_counter() - t_0
            
            
    return dict(MatrixName    = None,
                MatrixShape   = A.shape,
                DeflationMode = mode,
                Prolongation  = prol,
                Dimension     = k   ,
                x             = x   ,
                Residual      = res ,
                Error         = err ,
                TimeSetup     = t_U , 
                TimeCG        = t_cg)











def computeStats(A,prol,k,mode,omg=1.0):
    
    
    
    n = A.shape[0]
    Prec = Preconditioner(A, form='Diag', omg=omg).toarray()
    
    if   prol == 'I':
        U = Prolongator.U_I(n, k)
    elif prol == '100':
        U = Prolongator.U_100(n)
    elif prol == '121':
        U = Prolongator.U_121(n)
    elif prol == '2332':
        U = Prolongator.U_2332(n)
    elif prol == '11v':
        U = Prolongator.U_111v(n, k)
    elif prol == 'opt':
        U = Prolongator.U_opt(n, k, M=Prec@A)
    elif prol == 'optSI':
        U = Prolongator.U_optSI(n, k, M=Prec@A)
    elif prol == 'rand':
        U = Prolongator.U_rand(n, k)

    if   mode == 'ADEF1':
        P = ADEF1(A, U, Prec)
    elif mode == 'ADEF2':
        P = ADEF2(A, U, Prec)
    elif mode in ['BNN1','BNN2']:
        P = BNN(A, U, Prec)


    #ACinv = spl.inv(np.conj(U.T)@A@U)
    
    # Compute Error Propagation matrix and its A-norm
    E = np.eye(n)-P@A
    Anorm = spl.norm(spl.sqrtm(A)@E@spl.inv(spl.sqrtm(A)),2)
    
    #P2 = Prec@(np.eye(n)-A@U@ACinv@np.conj(U.T)) + U@ACinv@np.conj(U.T)
    #E2 = (np.eye(n)-Prec@A)@(np.eye(n)-U@ACinv@np.conj(U.T)@A)
    
    #Anorm2 = spl.norm(spl.sqrtm(A)@E2@spl.inv(spl.sqrtm(A)),2)
    
    # Compute singular values and condition number 
    sv = np.linalg.svd(P@A,compute_uv=False)
    condN = sv[0]/sv[-1]
    
    eigv = np.sort(spl.eig(Prec@A)[0])
    condNopt = np.real(eigv[n-1]/eigv[k])
    
    return Anorm, condN, condNopt #, P1, P2Anorm2,








def CG_gather(Matrix,testlist,T,tol=1e-10,maxiter=None):
    A, name = Matrix
    n       = A.shape[0]
    x_sol   = npr.randn(n)
    x_sol  /= np.linalg.norm(x_sol)
    time_setup = np.empty((T,len(testlist)))
    time_comCG = np.empty((T,len(testlist)))
    
    if not os.path.exists("results\\" + name):
        os.makedirs("results\\" + name)
        pickle.dump(A,open('results\\' + name + ".p","wb"))
    
    k = 6
    for item in testlist:
        time_setup = np.empty(T)
        time_comCG = np.empty(T)     
        if item[2] == '!':
            for i in range(T):
                print("Matrix",name,"with",item[0],item[1],":")
                print("Sequence",i+1,"of",str(T)+":")
                result = computeCG(A,x_sol,mode=item[0],prol=item[1],k=k,
                                   tol=tol,maxiter=maxiter,omg=item[4])
                time_setup[i] = result['TimeSetup']
                time_comCG[i] = result['TimeCG']
                clear_output()
            k = result['Dimension']
        else:
            for i in range(T):
                print("Matrix",name,"with deflation operator",
                      item[0],item[1],":")
                print("Sequence",i+1,"of",str(T)+":")
                result = computeCG(A,x_sol,mode=item[0],prol=item[1],k=item[2],
                                   tol=tol,maxiter=maxiter,omg=item[4])
                time_setup[i] = result['TimeSetup']
                time_comCG[i] = result['TimeCG']
                clear_output()
                
        result['MatrixName']         = name
        result['TimeSetup']          = time_setup
        result['TimeCG']             = time_comCG
        result['Colour']             = item[3]
        result['MaxIterations']      = maxiter
        result['Tolerance']          = tol
        result['RelaxationFactor']   = item[4]
        
        
        kString = str(result['Dimension'])
        kString = (4-len(kString))*str(0) + kString
        TString = str(len(result['TimeCG']))
        TString = (4-len(TString))*str(0) + TString
        omgStri = str(result['RelaxationFactor'])
        omgStri = omgStri + (4-len(omgStri))*str(0)
        
        
        pickle.dump(result,open("results\\" + name + "\\"
                                + result['DeflationMode'] + "_"
                                + result['Prolongation']
                                + "_k" + kString
                                + "_T" + TString
                                + "_omg" + omgStri
                                + ".p","wb"))



def AnormCondgather():
    
    matrixlist = [A_name for A_name in \
                  os.listdir('results') if A_name[-2:] != ".p"]
    
    data_Anorm = pd.DataFrame()
    data_condN = pd.DataFrame()
    data_condNopt = pd.DataFrame()
    
    for i,A_name in enumerate(matrixlist):
        
        # load matrix
        A = pickle.load(open('results\\'+A_name+'.p','rb')).toarray()
        
        tests = os.listdir('results\\'+A_name)
        
        
        dict_Anorm = {}
        dict_condN = {}
        dict_condNopt = {}
        
        for j,resultname in enumerate(tests):
            
            print("matrix",i+1,"of",len(matrixlist)+1)
            print("result",j+1s,"of",len(tests)+1)
            
            result = pickle.load(open('results\\'+A_name+'\\'+resultname,'rb'))
            
            
            # load properties
            k    = result['Dimension']
            #mode = result['DeflationMode']
            prol = result['Prolongation']
            
            Anorm, condN, condNopt = computeStats(A,prol,k,'ADEF1',omg=1.0)
            
            # set different k to vdim
            if k <= 15:
                dict_Anorm[resultname[:-8]] = Anorm #16
                dict_condN[resultname[:-8]] = condN
                dict_condNopt[resultname[:-8]] = condNopt
            else:
                dict_Anorm[resultname[:-12]+'vdim'] = Anorm #20
                dict_condN[resultname[:-12]+'vdim'] = condN
                dict_condNopt[resultname[:-12]+'vdim'] = condNopt
            
            clear_output()
        data_Anorm[A_name] = pd.Series(dict_Anorm)
        data_condN[A_name] = pd.Series(dict_condN)
        data_condNopt[A_name] = pd.Series(dict_condNopt)
        del(dict_Anorm)
        del(dict_condN)
        del(dict_condNopt)
        
    return data_Anorm, data_condN, data_condNopt
            






            
            
            
# Update directorys and results in obsolete format
def Patch_results():
    
    #Patch directorys name
    for A_name in os.listdir('results'):
        if A_name[-2:] !=".p":
            if A_name[-7:] == '.mtx.gz':
                os.rename('results\\'+A_name,'results\\'+A_name[:-7])
        elif A_name[-2:] ==".p":
            if A_name[-9:] == '.mtx.gz.p':
                os.rename('results\\'+A_name,'results\\'+A_name[:-9]+'.p')
    
    matrixlist = [A_name for A_name in \
                  os.listdir('results') if A_name[-2:] !=".p"]
    for A_name in matrixlist:
        
        if A_name[:13] == 'Poissonmatrix':
            tests = os.listdir('results\\'+A_name)
            A_name2 = 'Poisson'+A_name[13:]
            os.makedirs("results\\" + A_name2)
            for resultname in tests:
                result = pickle.load(open('results\\' + A_name + '\\'
                                          + resultname,'rb'))
                
                
                
                # Patch k and T in files name
                #kString = str(result['Dimension'])
                #kString = (4-len(kString))*str(0) + kString
                #TString = str(len(result['TimeCG']))
                #TString = (4-len(TString))*str(0) + TString
                
                # Patch result entry 'MatrixShape'
                #result['MatrixShape'] = pickle.load(open('results\\'+A_name+'.p','rb')).shape
    
                # Patch result entry 'MatrixName'
                #if result['MatrixName'][-7:] == '.mtx.gz':
                #    result['MatrixName'] = result['MatrixName'][:-7] 
                #if 'Residuum' in result.keys():
                #    result['Residual'] = result.pop('Residuum')
                
                result['MatrixName']=A_name2
                
                # Patch Colour
                #if result['Prolongation'] == 'optSI':
                #    result['Colour'] = 'black'
                
                pickle.dump(result,open("results\\" + A_name2 + "\\"
                                + resultname,"wb"))
                
            os.remove('results\\' + A_name + '\\' + resultname)