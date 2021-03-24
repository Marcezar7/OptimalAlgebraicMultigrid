# -*- coding: utf-8 -*-
"""



Matrix constructor
^^^^^^^^^^^^^^^^^^

This module contains functions that construct the following matrices:
    * a poissonmatrix of given dimension n
    * preconditioner for a given matrix A
    * a prolongation operator for multigrid in various forms
"""

import numpy as np
import numpy.random as npr
import scipy.sparse as sps
import scipy.sparse.linalg as spsl





def Poissonmatrix(n):
    """Creates a Poissonmatrix of given size.
    
    :param n:
        
        dimension of the constructed matrix
    :type n: int
    
    
    :return:
        
        poissonmatrix of shape :math:`n \\times n` in sparse format
    :rtype: scipy.sparse.csr.csr_matrix
    
    
    """
    Diagonals = [-1,2,-1]
    A = sps.diags(Diagonals,[-1,0,1],[n,n]).tocsr()
    return A




def Preconditioner(A,form='Diag',omg=1.0):
    """Creates a Preconditioner for a given matrix in different forms.
    
    :param A: matrix that the preconditioner is for
    :type A: array_like
    
    :param form: possible forms are 'Diag' and 'LTriag'
    :type form: str, optional
    
    :param omg: relaxation parameter
    :type omg: float, optional
    
        
    :return: preconditioner of shape nxn in sparse format
    :rtype: scipy.sparse.dia.dia_matrix, scipy.sparse.csc.csc_matrix
    
    
    """
    if not 0 < omg < 2:
        print("Omega has to be between 0 and 2, setting omega to 1...")
        omg=1.0

    if   form == 'Diag':
        return omg*sps.diags(1/A.diagonal())

    elif form == 'LTriag':
        n = A.shape[0]
        L = sps.diags([A.diagonal(-k) for k in range(n)],
                      [-i for i in range(n)])
        return omg*spsl.inv(L.tocsc())

    else:
        raise NameError("Possible choices for 'form' are 'Diag' and 'LTriag'.")




class Prolongator(): # p!rolongator
    # =========================================================================
    # Create a prolongation matrix in a specified form
    # =========================================================================
    
    def U_I(n,k):
        return np.vstack((np.eye(k),np.zeros((n-k,k))))



    def U_100(n):
        k = (n+1)//2
        U = np.zeros((n,k))
        for j in range(k):
            U[2*j,j] = 1
        return U



    
    def U_121(n):
        k=(n-1)//2
        U=np.zeros((n,k))
        for j in range(k):
            U[2*j  ,j] = 1
            U[2*j+1,j] = 2
            U[2*j+2,j] = 1
        if n%2 == 0:
            fill = np.zeros((n,1))
            fill[-2:]=np.array([1,3]).reshape(2,1)
            return np.hstack((U,fill))/4
        else:
            return U/4



    
    def U_2332(n):
        k=n//2-1
        U=np.zeros((n,k))
        for j in range(k):
            U[2*j  ,j] = 2
            U[2*j+1,j] = 3
            U[2*j+2,j] = 3
            U[2*j+3,j] = 2
        if n%2 == 1:
            fill = np.zeros((n,1))
            fill[-3:]=np.array([2,3,5]).reshape(3,1)
            return np.hstack((U,fill))/10
        else:
            return U/10



    
    def U_111v(n,k):
        U=np.zeros((n,k))
        m = int(n/k)
        for j in range(k):
            U[m*j:m*j+m,j] = np.ones(m)
        return U/(n//k)




    
    def U_opt(n,k,M,max_iter=None,tol=0):
        """
        ----------------------------------------------------------------------
    
        OPTIMAL PROLANGATOR for ADEPTED DEFLATION OPERATOR
        
        ----------------------------------------------------------------------
        """
        U = spsl.eigs(M,k,which="SM",maxiter=max_iter,tol=tol)[1]
        return U
   



    
    def U_optSI(n,k,M,max_iter=None,tol=0):
        """
        ----------------------------------------------------------------------
    
        OPTIMAL PROLANGATOR for ADEPTED DEFLATION OPERATOR
        with shift invert mode
     
        ----------------------------------------------------------------------
        """
        U = spsl.eigs(M,k,sigma=0+0j,maxiter=max_iter,tol=tol)[1]
        return U   
    

    
    
    def U_rand(n,k,seed=None,max_iter=10):
        """
        ----------------------------------------------------------------------
    
        random prolongator
     
        ----------------------------------------------------------------------
        """
        npr.seed(seed)
        for i in range(max_iter):
            U = npr.normal(loc=4, scale=5, size=(n,k))
            rank = np.linalg.matrix_rank(U)
            if k == rank:
                for line in U.T:
                    line /= np.linalg.norm(line)
                return U
        raise RuntimeError("No random Prolongator with rank r found within",
                           max_iter,"iterations.")
