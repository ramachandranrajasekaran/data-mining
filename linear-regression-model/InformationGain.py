# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 19:41:14 2020

@author: Ramachandran R
"""

import numpy as np
from math import log2
from xlwt import Workbook

def calcEnthropy(x, y):
    pX = x/(x+y) if x+y != 0 else 0
    pY = y/(x+y) if x+y != 0 else 0
    eX = log2(pX) if pX != 0 else 0
    eY = log2(pY) if pY != 0 else 0
    
    enthropy = -((pX*eX)+(pY*eY))
    return enthropy

def informationGain(x, y, eB):
    mx = np.mean(x)
    
    inl = np.where(x<=mx)
    inh = np.where(x>mx)
    
    yl = y[inl[0]]
    yh = y[inh[0]]
    
    nl0 = len(np.where(yl == 0)[0])
    nl1 = len(np.where(yl > 0)[0])
    
    enpl = calcEnthropy(nl0, nl1)
    
    nh0 = len(np.where(yh == 0)[0])
    nh1 = len(np.where(yh > 0)[0])
    
    enph = calcEnthropy(nh0, nh1)
    
    ig = eB - (((nl0+nl1)/(nl0+nl1+nh0+nh1))*enpl) - (((nh0+nh1)/(nl0+nl1+nh0+nh1))*enph) 
    return ig
    
wb = Workbook() 
sheet1 = wb.add_sheet('InformationGain') 
ig=np.zeros((57,20))
for j in range(1,57):
    fname='C:/Users/Ramji/BITS/SEMESTER_2/Data Mining (ISZC415)/Assignment/data/'+str(j)+'.csv'
    df = np.genfromtxt(fname,delimiter=',')
    features=df[:,0:-1]
    bugs=df[:,-1]
    
    c1 = len(np.where(bugs==0)[0])
    c2 = len(np.where(bugs>0)[0])
    
    eB = calcEnthropy(c1, c2)
    
    projNum = 'Project_'+str(j)
    sheet1.write(j-1, 0, projNum)
    
    for i in range(0,20):
        ig[j,i]=informationGain(features[:,i], bugs, eB) 
        
        sheet1.write(j-1, i+1, ig[j,i]) 
    
wb.save('2019ht12107.xls')
