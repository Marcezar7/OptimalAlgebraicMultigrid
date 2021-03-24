# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 12:26:13 2019

@author: Marcezar
"""

import numpy as np
import numpy.linalg as npl





def A_norm(x,A):
    return np.sqrt(np.vdot(x,A@x))





def cg(A,b,x0=None,tol=1e-5,maxiter=None,M1=None,M3=None,x_exact=None):
    n = A.shape[0]
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0
    if maxiter is None:
        maxiter = 3*n
    r = b-A@x
    res_arr = np.zeros(maxiter)
    res_arr[0] = npl.norm(r)
    if x_exact is not None:
        err_arr = np.zeros(maxiter)
        err_arr[0] = A_norm(x-x_exact,A)
    i = 0
    
    # unpreconditioned CG-method
    if M1 is None  and M3 is None:
        p = r
        
        while res_arr[i] > tol*res_arr[0] and i < maxiter-1:
            i = i+1
            w       = A@p
            aph     = np.vdot(r,r)/np.vdot(p,w)
            x       = x + aph*p
            rnew    = r - aph*w
            bta     = np.vdot(rnew,rnew)/np.vdot(r,r)
            p       = rnew + bta*p
            r       = rnew
            res_arr[i] = npl.norm(r)
            if x_exact is not None:
                err_arr[i] = A_norm(x-x_exact,A) 

    # CG method for using a preconditioner
    elif M1 is not None and M3 is None:
        y = p = M1@r
        while res_arr[i] > tol*res_arr[0] and i < maxiter-1:
            i = i+1
            w       = A@p
            aph     = np.vdot(y,r)/np.vdot(p,w)
            x       = x + aph*p
            rnew    = r - aph*w
            ynew    = M1@rnew
            bta     = np.vdot(ynew,rnew)/np.vdot(r,y)
            p       = ynew + bta*p
            r       = rnew
            y       = ynew
            res_arr[i] = npl.norm(r)
            if x_exact is not None:
                err_arr[i] = A_norm(x-x_exact,A)
    
    # CG method for using the P_DEF-Operator
    elif M1 is not None and M3 is not None:
        y = p = M1@r
        while res_arr[i] > tol*res_arr[0] and i < maxiter-1:
            i = i+1
            w       = M3@(A@p)
            aph     = np.vdot(y,r)/np.vdot(p,w)
            x       = x + aph*p
            rnew    = r - aph*w
            ynew    = M1@rnew
            bta     = np.vdot(ynew,rnew)/np.vdot(r,y)
            p       = ynew + bta*p
            r       = rnew
            y       = ynew
            res_arr[i] = npl.norm(r)
            if x_exact is not None:
                err_arr[i] = A_norm(x-x_exact,A)
        Qb = np.conj(M3.U.T)@b
        Qb = np.linalg.solve(M3.E,Qb)
        Qb = M3.U@Qb
        x = Qb + M3.H@x

    else:
        raise NotImplementedError("no 'M1' given")

    res_arr = np.delete(res_arr,range(i+1,maxiter))
    
    if x_exact is not None:
        err_arr = np.delete(err_arr,range(i+1,maxiter))
        return (x,res_arr,err_arr)
    else:
        return (x,res_arr)
    
    