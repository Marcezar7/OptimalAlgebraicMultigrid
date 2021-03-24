# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 14:55:07 2020

@author: Marcezar
"""

import numpy as np
import scipy.linalg as spl
import numpy.random as npr

class KrylovSubSpace():

    # tol_ROG = np.inf -> niemals
    #         = beliebige Toleranz    
    #         = 0      -> immer
    def __init__(self,A,v0=None,maxdim=25,tol_stop=1e-15,tol_ROG=1e-15):
        self.A         = A
        self.m         = maxdim
        self.tol_stop  = tol_stop
        self.tol_ROG   = tol_ROG
        
        self.V         = np.zeros((self.A.shape[0],self.m+1),
                                  dtype=np.complex128)
        if v0 is None:
            self.V[:,0]  = npr.randn(self.A.shape[0])
        else:
            self.V[:,0]  = v0
        self.V[:,0] /= spl.norm(self.V[:,0])
        self.dim     = 1
        
        self.H = np.zeros((self.m+1,self.m),dtype=np.complex128)
        
        self.ArnoldiMGS()



    def ArnoldiMGS(self):
        timer = 0
        for j in range(self.dim-1,self.m):
            wj_init = self.A@self.V[:,j]
            wj      = wj_init.copy()
            for i in range(j+1):
                self.H[i,j] = np.vdot(self.V[:,i],wj)
                wj    -= self.H[i,j]*self.V[:,i]
                
            v = wj
            
            #Reorthogonalization
            if abs(spl.norm(wj) - spl.norm(wj_init)) > self.tol_ROG:
                timer += 1
                for i in range(j):
                    v -= np.vdot(self.V[:,i],wj)*self.V[:,i]
                    
            self.H[j+1,j] = spl.norm(v)
            
            if abs(self.H[j+1,j]) < self.tol_stop:
                print("Toleranz erreicht")
                self.dim = j+1
                break
                #return self.H,self.V
            
            self.V[:,j+1] = v/self.H[j+1,j]
        #print("V wurde",timer,"mal reorthogonalisiert")
        #return H,V
        self.dim = self.m



    def implicitRestart(self,shifts):
        k    = self.m - len(shifts)
        beta = self.H[-1,-1]
        Q    = np.eye(self.m)
        
        for shift in shifts:
            Q_temp, R     = spl.qr(self.H[:-1,:] - shift*np.eye(self.m))
            self.H[:-1,:] = R@Q_temp + shift*np.eye(self.m)
            Q             = Q@Q_temp
        
        v    = (self.H[k,k-1]*self.V[:,:-1]@Q[:,k]       # Nochmal prüfen
                + beta*Q[self.m-1,k-1]*self.V[:,self.m])
        beta = spl.norm(v)
        v   /= beta

        self.V[:,:k]  = self.V[:,:-1]@Q[:,:k]
        self.V[:,k]   = v
        self.H[k,k-1] = beta
        self.dim = k

    # Hier wird Vk+1 weggelassen und stattdessen die Dimension des kleineren
    # Unterraums um 1 verkleinert
    def implicitRestart2(self,shifts):
        k = self.m - len(shifts)
        Q = np.eye(self.m)
        
        for shift in shifts:
            Q_temp, R = spl.qr(self.H[:-1,:] - shift*np.eye(self.m))
            self.H[:-1,:] = R@Q_temp + shift*np.eye(self.m)
            Q             = Q@Q_temp
            
        #v = self.H[k,k-1]*
        self.V[:,:k] = self.V[:,:-1]@Q[:,:k]
        self.dim = k - 1
        

# k -> shifts
def eigs(A,k=6,maxdim=None,v0=None,maxiter=None,tol=1e-15): # ,which='SM'
    if maxdim  is None: maxdim  = 4*k+1
    if maxiter is None: maxiter = 10*A.shape[0]
    KS = KrylovSubSpace(A,v0,maxdim)
    
    res_norm = np.empty(k)
    
    for i in range(maxiter):

        
        
        # Compute Eigenpairs of H and sort them
        # from smallest to largest abs(eigenvalue):
        eigval_m, eigvec_m = spl.eig(KS.H[:-1,:])
        ind    = np.abs(eigval_m).argsort()
        eigval_m = eigval_m[ind]
        eigvec_m = eigvec_m[:,ind]
        
        # Choose the desired Eigenpairs of H:
        eigval = eigval_m[:k]
        eigvec = eigvec_m[:,:k]

        # Calculate the Residuum and check if
        # the stopping criterion is fullfilled:
        res_norm = np.abs(KS.H[-1,-1]*eigvec[-1])
        veconv   = res_norm <= tol
        if veconv.sum() == k:
            return eigval, eigvec
        
        # Choose the largest m-k eigenvalues
        # of H as shifts for implicit restart:
        shifts = eigval_m[k:]
        KS.implicitRestart(shifts)
        
        # Enlarge the KrylovSubspace to maxdim = m
        KS.ArnoldiMGS()
    
    raise RuntimeError(veconv.sum(),"/",k,"eigenvectors converged")

