import scipy.io as sio
import attr
import pandas as pd
import numpy as np
import math
from scipy.optimize import least_squares
from DataProcessing import Trial


class MJxy:
    """
    Creates an MSE function for jerk polynom. 
    df = processed data frame
    num_block = number of block
    num_r = number of repetition
    t0 = movement start time
    t = vector of times per rep
    D  = movement duration
    Ax = displacement resulting from the movement (x): PosX(t=max) - PosX(t=1)
    Ay = displacement resulting from the movement (y): PosY(t=max) - PosY(t=1)
 
    """
    def __init__(self, id, num_block, num_r, t0, t, D, Ax, Ay):
        """Initialise the trajectory with starting state."""
        self.trial = trial
        self.num_block = self.trial.block
        self.rep = self.trial.rep
        self.condition = self.trial.condition
        self.Ax = self.trial.filtered_position_data[x][-1]  -  self.trial.filtered_position_data[x][0]) #for 1 J
        self.Ay = self.trial.filtered_position_data[y][-1]  -  self.trial.filtered_position_data[y][0]) #for 1 J
        self.t = self.trial.time
        self.D = max(self.t)-min(self.t)) #for 1 J
        self.t0 = t0
        

    def create_predicted_func(self,t, Ax, Ay, D, t0):
        """Creates the predicted jerk functions for x and y"""
        nt = (self.t-self.t0)./self.D  #nt is normalized time (0 <= nt <= 1)
        self.Px = self.Ax/self.D * (-60 * nt**3 + 30 * nt.**4 + 30 * nt**2)
        self.Py = self.Ay/self.D * (-60 * nt**3 + 30 * nt.**4 + 30 * nt**2)
        
        A_tang = math.sqrt((self.Ax/self.D)**2 + (self.Ay/self.D)**2)
        self.B = A_tang * (-60 * nt**3 + 30 * nt**4 + 30 * nt**2)

        return self.Px,self.Py,self.B

    def create_observed_func(self,Vx,Vy):
        """Creates the observed functions for x and y"""
        self.Ox = Vx
        self.Oy = Vy
        return self.Ox, self.Oy

    def create_cost_func(self, t, Ax, Ay, D, t0):
        """Creates the cost jerk function for x and y"""
        Px, Py, B = self.create_predicted_func(t, Ax, Ay, D, t0)
        Ox, Oy = self.create_observed_func(Vx, Vy)
        self.epsilon = np.sum((Px-Ox)**2 + (Py-Oy)**2 + (B-math.sqrt(Ox**2 + Oy**2))**2/2*(Ox**2+Oy**2))
        return self.epsilon

    def optimize_jerk(self):
        """Creates the cost jerk function for x and y"""
        res_1 = least_squares(self.epsilon, [t0, D, Ax, Ay], bounds=([0, max(t)], [1, max(t)-min(t)], [-(posx[-1]-posx[0]), posx[-1]-posx[0]], [-(posy[-1]-posy[0]), posy[-1]-posy[0]]))

    



