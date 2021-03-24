# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:51:12 2019

@author: Marcezar
"""
import numpy as np
from scipy.sparse import eye
from scipy.sparse.linalg import LinearOperator

class DeflationOperator(LinearOperator):


    def __init__(self,A,U,M=None):
        super().__init__(A.dtype,A.shape)
        self._A = A
        self._U = U
        if M is None:
            M = eye(self.shape[0])
        self._M = M
        self._E = np.conj(U.T)@A@U
        


    def get_starting_vector(self,b,x0):
        b = np.conj(self._U.T)@b
        b = np.linalg.solve(self._E,b)
        b = self._U@b
        
        x = self._A@x0
        x = np.conj(self._U.T)@x
        x = np.linalg.solve(self._E,x)
        x = self._U@x
        
        return b + x0 - x


    def get_A(self):
        return self._A
    def set_A(self,A):
        if not A.shape[0] == A.shape[1] == self._U.shape[0]:
            raise ValueError("Wrong dimensions")
        self._A = A
        self._E = np.conj(self._U.T)@self._A@self._U
    def del_A(self):
        del(self._A)
        del(self._E)
    A = property(get_A,set_A,del_A)


    def get_U(self):
        return self._U
    def set_U(self,U):
        if not self._A.shape[0] == U.shape[0] == self._M.shape[0]:
            raise ValueError("Wrong dimensions")
        self._U = U
        self._E = np.conj(self._U.T)@self._A@self._U
    def del_U(self):
        del(self._U)
        del(self._E)
    U = property(get_U,set_U,del_U)


    def get_M(self):
        return self._M
    def set_M(self,M):
        if not self._A.shape == M.shape:
            raise ValueError("Wrong dimensions")
        self._M = M
    def del_M(self):
        del(self._M)
    M = property(get_M,set_M,del_M)


    def get_E(self):
        return self._E
    def set_E(self,_):
        raise AttributeError("This Attribute is protected, change A or U instead.")
    def del_E(self):
        raise AttributeError("This Attribute is protected, change A or U instead.")
    E = property(get_E,set_E,del_E)
    


class ADEF1(DeflationOperator):
    def __init__(self,A,U,M=None):
        super().__init__(A,U,M=None)
        
    def _matvec(self,v1):
        v2 = np.conj(self._U.T)@v1
        v2 = np.linalg.solve(self._E,v2)
        v2 = self._U@v2
        
        v3 = self._A@v2
        
        return self.M@(v1-v3)+v2
    
    def get_starting_vector(self,b,x0):
        raise NotImplementedError("Not implemented for ADEF1.")
    
    
        
class ADEF2(DeflationOperator):
    def __init__(self,A,U,M=None):
        super().__init__(A,U,M=None)
        
    def _matvec(self,v1):
        v2 = self._M@v1
        v3 = self._A@v2
        v3 = np.conj(self._U.T)@v3
        v3 = np.linalg.solve(self._E,v3)
        v3 = self._U@v3
        
        v4 = np.conj(self._U.T)@v1
        v4 = np.linalg.solve(self._E,v4)
        v4 = self._U@v4

        return v2 - v3 + v4
    
    
        
class BNN(DeflationOperator):
    def __init__(self,A,U,M=None):
        super().__init__(A,U,M=None)
        
    def _matvec(self,v1):
        v2 = np.conj(self._U.T)@v1
        v2 = np.linalg.solve(self._E,v2)
        v2 = self._U@v2
        
        v3 = self._A@v2
        v3 = self._M@v3
        v3 = self._A@v3
        v3 = np.conj(self._U.T)@v3
        v3 = np.linalg.solve(self._E,v3)
        v3 = self._U@v3
        
        v4 = self._A@v2
        v4 = self._M@v4
        
        v5 = self._M@v1
        
        v6 = self._A@v5
        v6 = np.conj(self._U.T)@v6
        v6 = np.linalg.solve(self._E,v6)
        v6 = self._U@v6
        
        return v2 + v3 - v4 + v5 - v6