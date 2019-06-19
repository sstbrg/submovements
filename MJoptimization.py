import scipy.io as sio
import pandas as pd
import numpy as np
import math
from scipy.optimize import least_squares


class MJxy:
    """
    Creates an MSE function for jerk polynom. 
    df = processed data frame
    num_block = number of block
    num_r = number of repetition
    t0 = movement start time
    t = vector of times per rep
    D  = movement duration
    Ax = displacement resulting from the movement (x)
    Ay = displacement resulting from the movement (y)
 
    """
    def __init__(self, id, num_block, num_r, t0, t, D, Ax, Ay):
        """Initialise the trajectory with starting state."""
        self._id = id
        self._num_block = num_block 
        self._num_r = num_r
        self._t0 = t0
        self._t = t
        self._D = D
        self._Ax = Ax
        self._Ay = Ay
        #self.reset()

    #def rep_for_fit(self, num_block, num_r,  )

    def create_predicted_func(self)
        """Creates the predicted jerk functions for x and y"""
        nt = (self.t-self.t0)./self.D  #nt is normalized time (0 <= nt <= 1)
        self.Px = self.Ax/self.D * (-60 * nt.**3 + 30 * nt.**4 + 30 * nt.**2)
        self.Py = self.Ay/self.D * (-60 * nt.**3 + 30 * nt.**4 + 30 * nt.**2)
        
        A_tang = math.sqrt((self.Ax/self.D)**2 + (self.Ay/self.D)**2)
        self.B = A_tang * (-60 * nt**3 + 30 * nt**4 + 30 * nt**2)

        return self.Px,self.Py,self.B

    def create_observed_func(self,Vx,Vy)
        """Creates the observed functions for x and y"""
        self.Ox = Vx
        self.Oy = Vy
        return self.Ox, self.Oy

    def create_cost_func(self)
        """Creates the cost jerk function for x and y"""
        self.epsilon = np.sum((self.Px-self.Ox)**2 + (self.Py-self.Oy)**2 + (self.B-math.sqrt(self.Ox**2 + self.Oy**2))**2/2*(self.Ox**2+self.Oy**2)
        return self.epsilon

    def optimize_jerk(self)
        """Creates the cost jerk function for x and y"""
        res_1 = least_squares(self.epsilon, )

    



