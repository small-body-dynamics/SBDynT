import sys
import numpy as np
import pandas as pd
import tools
import run_reb
import tno
import resonances
import machine_learning


from os import path
from datetime import date
from pickle import dump
from pickle import load
from importlib import resources as impresources



class SSSB:
    # class that stores the information for a Solar System Small Body for dynamical analysis.
    def __init__(self,des='',clones=0):
        self.clones = clones
        self.des = des

        
        self.sim = make(des,rocky_planets=False,clones=0,filename='Single')
        
        self.ml_data = tno_classifier.TNO_ML_outputs(clones=clones)

        self.pes = [0,0,0]
        self.pes_err = [0,0,0]

        self.prop_omega = 0
        self.prop_Omega = 0
        
        self.g = 0
        self.s = 0
        
        self_rese = False
        self_resI = False


    def init_pe(des,rocky_planets=False,clones=0,filename='Single'):

        
        self.sim = sim