# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 07:45:53 2016

@author: ankit
"""

import scipy.io
import pickle as pkl
from random import shuffle

#import numpy as np

inpmat = scipy.io.loadmat('/home/ankit/Downloads/Indian_pines.mat')['indian_pines']
targetmat = scipy.io.loadmat('/home/ankit/Downloads/Indian_pines_gt.mat')['indian_pines_gt']
width = inpmat.shape[1]
height = inpmat.shape[0]
print height, width
Train,Test = [],[]
Classes = []
for i in range(16):
    Classes.append([])
for i in range(height):
    for j in range(width):
        curr_inp = inpmat[i,j,:]
        curr_tar = targetmat[i,j]
        if(curr_tar>0):
            Classes[curr_tar-1].append(curr_inp)
 
for i in range(16):
    shuffle(Classes[i])
    for j in range(len(Classes[i])/5):
        Train.append((Classes[i][j],i))

    for j in range(len(Classes[i])/5,len(Classes[i])):
        Test.append((Classes[i][j],i))
        
#shuffle(Train)

f_out = open('Data_new.pkl','wb')
pkl.dump({'train_data':Train}, f_out)
pkl.dump({'test_data':Test}, f_out)
f_out.close()