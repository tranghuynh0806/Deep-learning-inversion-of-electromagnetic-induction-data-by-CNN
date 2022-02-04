# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script filefr
"""

import numpy as np
import discretize
import matplotlib as mpl
import matplotlib.pyplot as plt
from pymatsolver import Solver
from SimPEG import (
    maps,
    utils,
    data_misfit,
    regularization,
    optimization,
    inversion,
    inverse_problem,
    directives,
    data,
)
import random
import csv
from random import uniform
from SimPEG.electromagnetics import frequency_domain as FDEM, time_domain as TDEM
# cs- cell size, ncx-number of core cell in x-direction, ncz-number of core cell in z-direction
cs, ncx, ncz, npad = 1.0, 20.0, 20.0, 20 # 20 padding cells
hx = [(cs, ncx), (cs, npad, 1.3)] # 1.3 is the exansion factor (don't want to change
# the cell sizes due to the wave equation)
#hx is the width of the cell in x-direction and start counting from the origin


npad = 15
temp = np.ones(39) * 2.5 #There are 10 layers but divided into 20 cells
temp_1 = np.logspace(np.log10(2.5), np.log10(15))
temp_pad = temp_1[-1] * 1.3 ** np.arange(npad)
#temp = np.logspace(np.log10(1), np.log10(12), 40) # return numbers spaced evenly on a log scale
#print("temp", temp.shape) # input the number cells for z direction
#temp_pad = temp[-1] * 1.3 ** np.arange(npad) # return evenly space values within a given interval 0-11
#print("temp_pad", temp_pad.shape)
hz = np.r_[temp_pad[::-1], temp_1[::-1], temp[::-1], temp, temp_1, temp_pad]
#hz = np.r_[temp_pad[::-1], temp[::-1], temp, temp_pad] #translate slice objects to join a sequence array along the first axis
mesh = discretize.CylMesh([hx, 1, hz], "00C") #use flag of 1 to denote perfect rotational symmetry
#00C is the orgin of the mesh (x starts from zero, y does not matter in this case and z in the center)
#print("hz", hz.shape)
#print("hz_value", hz )
mesh.plotGrid()


# Step2: Set the mapping and a starting model
# Note: this sets our inversion model as 1D log conductivity
# below subsurface

# vectorCCz is the cell-centered grid vector (1D) in the z direction
active = mesh.vectorCCz < 0#size of 62 with 31 trues and the rest is false 


# active map include the air cell
layer1 =(mesh.vectorCCz < 0) & (mesh.vectorCCz >= -5) # size of 62 with 3 trues and the rest false
layer2 =(mesh.vectorCCz < -5) & (mesh.vectorCCz >= -10)
layer3 =(mesh.vectorCCz < -10) & (mesh.vectorCCz >= -15)
layer4 =(mesh.vectorCCz < -15) & (mesh.vectorCCz >= -20)
layer5 =(mesh.vectorCCz < -20) & (mesh.vectorCCz >= -25)
layer6 =(mesh.vectorCCz < -25) & (mesh.vectorCCz >= -30)
layer7 =(mesh.vectorCCz < -30) & (mesh.vectorCCz >= -35)
layer8 =(mesh.vectorCCz < -35) & (mesh.vectorCCz >= -40)
layer9 =(mesh.vectorCCz < -40) & (mesh.vectorCCz >= -45)
layer10 =(mesh.vectorCCz < -45) & (mesh.vectorCCz >= -50)
layer11 =(mesh.vectorCCz < -50) & (mesh.vectorCCz >= -55)
layer12 =(mesh.vectorCCz < -55) & (mesh.vectorCCz >= -60)
layer13 =(mesh.vectorCCz < -60) & (mesh.vectorCCz >= -65)
layer14 =(mesh.vectorCCz < -65) & (mesh.vectorCCz >= -70)
layer15 =(mesh.vectorCCz < -70) & (mesh.vectorCCz >= -75)
layer16 =(mesh.vectorCCz < -75) & (mesh.vectorCCz >= -80)
layer17 =(mesh.vectorCCz < -80) & (mesh.vectorCCz >= -85)
layer18 =(mesh.vectorCCz < -85) & (mesh.vectorCCz >= -90)
layer19 =(mesh.vectorCCz < -90) & (mesh.vectorCCz >= -95)
layer20 =(mesh.vectorCCz < -95) 
#half_space = (mesh.vectorCCz < -100)

actMap = maps.InjectActiveCells(mesh, active, np.log(1e-8), nC=mesh.nCz) 
#InjectActiveCells (62,31) -> 31 is the number of cell that below the subsurface
mapping = maps.ExpMap(mesh) * maps.SurjectVertical1D(mesh) * actMap

# SurjectVertical1D map takes the 1D vector that we find vertically and put it horizontally
# map.ExpMap(mesh) -> true vertical conductivity map
# actMap.shape -> (62, 31) -> take vector of length 31 and output the vector of length 62
# maps.SurjectVertical1D(mesh).shape ->(1860,62) -> the vector of length 62 and output
#the vector of length 1860

sig_1 = round(uniform(0.001, 1.001),3)
sig_2 = round(uniform(0.001, 1.001),3)
sig_3 = round(uniform(0.001, 1.001),3)
sig_4 = round(uniform(0.001, 1.001),3)
sig_5 = round(uniform(0.001, 1.001),3)
sig_6 = round(uniform(0.001, 1.001),3)
sig_7 = round(uniform(0.001, 1.001),3)
sig_8 = round(uniform(0.001, 1.001),3)
sig_9 = round(uniform(0.001, 1.001),3)
sig_10 = round(uniform(0.001, 1.001),3)
sig_11 = round(uniform(0.001, 1.001),3)
sig_12 = round(uniform(0.001, 1.001),3)
sig_13 = round(uniform(0.001, 1.001),3)
sig_14 = round(uniform(0.001, 1.001),3)
sig_15 = round(uniform(0.001, 1.001),3)
sig_16 = round(uniform(0.001, 1.001),3)
sig_17 = round(uniform(0.001, 1.001),3)
sig_18 = round(uniform(0.001, 1.001),3)
sig_19 = round(uniform(0.001, 1.001),3)
sig_20 = round(uniform(0.001, 1.001),3)
sig_half = 1e-4
sig_air = 1e-8
sigma = np.ones(mesh.nCz) * sig_air #return a new array with the shape of mesh.nCz, fill with ones
#sigma[active] = sig_half# the conductivity of the last layer

sigma[layer1] = sig_1
sigma[layer2] = sig_2
sigma[layer3] = sig_3
sigma[layer4] = sig_4
sigma[layer5] = sig_5
sigma[layer6] = sig_6
sigma[layer7] = sig_7
sigma[layer8] = sig_8
sigma[layer9] = sig_9
sigma[layer10] = sig_10
sigma[layer11] = sig_11
sigma[layer12] = sig_12
sigma[layer13] = sig_13
sigma[layer14] = sig_14
sigma[layer15] = sig_15
sigma[layer16] = sig_16
sigma[layer17] = sig_17
sigma[layer18] = sig_18
sigma[layer19] = sig_19
sigma[layer20] = sig_20
print("sigma of layer 1", sigma[layer1])




# Plot the intial model
#plt.colorbar(mesh.plotImage(np.log10(mapping*m0))[0])


###################################################################################




# Print out the model, real magnetic values and imaginary magnetic values
# Initial and reference model
x = 0
for x in range(30001):
    # Conductivity values for each layer
    sig_1 = round(uniform(0.001, 1.001),3)
    sig_2 = round(uniform(0.001, 1.001),3)
    sig_3 = round(uniform(0.001, 1.001),3)
    sig_4 = round(uniform(0.001, 1.001),3)
    sig_5 = round(uniform(0.001, 1.001),3)
    sig_6 = round(uniform(0.001, 1.001),3)
    sig_7 = round(uniform(0.001, 1.001),3)
    sig_8 = round(uniform(0.001, 1.001),3)
    sig_9 = round(uniform(0.001, 1.001),3)
    sig_10 = round(uniform(0.001, 1.001),3)
    sig_11 = round(uniform(0.001, 1.001),3)
    sig_12 = round(uniform(0.001, 1.001),3)
    sig_13 = round(uniform(0.001, 1.001),3)
    sig_14 = round(uniform(0.001, 1.001),3)
    sig_15 = round(uniform(0.001, 1.001),3)
    sig_16 = round(uniform(0.001, 1.001),3)
    sig_17 = round(uniform(0.001, 1.001),3)
    sig_18 = round(uniform(0.001, 1.001),3)
    sig_19 = round(uniform(0.001, 1.001),3)
    sig_20 = round(uniform(0.001, 1.001),3)
    # Set them for each layer
    sigma[layer1] = sig_1
    sigma[layer2] = sig_2
    sigma[layer3] = sig_3
    sigma[layer4] = sig_4
    sigma[layer5] = sig_5
    sigma[layer6] = sig_6
    sigma[layer7] = sig_7
    sigma[layer8] = sig_8
    sigma[layer9] = sig_9
    sigma[layer10] = sig_10
    sigma[layer11] = sig_11
    sigma[layer12] = sig_12
    sigma[layer13] = sig_13
    sigma[layer14] = sig_14
    sigma[layer15] = sig_15
    sigma[layer16] = sig_16
    sigma[layer17] = sig_17
    sigma[layer18] = sig_18
    sigma[layer19] = sig_19
    sigma[layer20] = sig_20
    
    #m0 = np.log(sigma[layer1], sigma[layer2], sigma[layer3], sigma[layer4], sigma[layer5], sigma[layer6], sigma[layer7], sigma[layer8], sigma[layer9], sigma[layer10], sigma[layer11] ) # only from sigmal 1 to sigma 20 (only 20 value in the log of sigma)
    m0 = np.log(sigma[active])
    
    array = [m0[0], m0[66], m0[68], m0[70], m0[72], m0[74], m0[76], m0[78], m0[80], m0[82], m0[84], m0[86], m0[88], m0[90], m0[92], m0[94], m0[96], m0[98], m0[100], m0[102]]
    #print(m0[0], m0[66], m0[68], m0[70], m0[72], m0[74], m0[76], m0[78], m0[80], m0[82], m0[84], m0[86], m0[88], m0[90], m0[92], m0[94], m0[96], m0[98], m0[100], m0[102])
    array_conversion = np.array(array)
    print(array_conversion)
    
    with open('model_30000.csv', 'a') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(array_conversion)
        csv_file.close()
   
   
    #np.savez("modeling.npz", array)
    #npzfile = np.load("modeling.npz")
        

        
    # Step3: Invert Resolve data
    # Bird height from the surface

    src_height_resolve = 30
    
    # Set Rx (In-phase and Quadrature)
    rxOffset = 7.86
    bzr = FDEM.Rx.PointMagneticFluxDensitySecondary(
       np.array([[rxOffset, 0.0, src_height_resolve]]),
       orientation="z",
       component="real",
    )
    
    bzi = FDEM.Rx.PointMagneticFluxDensitySecondary(
       np.array([[rxOffset, 0.0, src_height_resolve]]),
       orientation="z",
       component="imag",
    )
    
    
    # Set Source (In-phase and Quadrature)
    #frequency_cp = resolve["frequency_cp"][()]
    freqs = [380, 1792, 8180, 41060, 128520]
    srcLoc = np.array([0.0, 0.0, src_height_resolve])
    
    srcList = [
        FDEM.Src.MagDipole([bzr, bzi], freq, srcLoc, orientation="Z") for freq in freqs
    ]
    
    # Set FDEM survey (In-phase and Quadrature)
    survey = FDEM.Survey(srcList)
    prb = FDEM.Simulation3DMagneticFluxDensity(mesh, sigmaMap=mapping)
    prb.survey = survey
    prb.solver = Solver
    data = prb.dpred(m0)
    bz_real = data[0:len(data):2]
    bz_ima = data[1:len(data):2]

    
    mu= 4 * np.pi * 10**(-7)
    bp = -mu / (4*np.pi*rxOffset**3) 
    bz_real = bz_real*(1e6) / bp # ppm real survey
    bz_ima = bz_ima *(1e6) / bp
    arr1= np.array(bz_real)
    arr2= np.array(bz_ima)
    bz=[]
    #print("Bz_real: ", bz_real)
    #print("Bz_image: ", bz_ima)
    bz = np.concatenate((arr1, arr2))
    print("Bz", bz)
    
    '''
    with open('bz_final.csv', 'a') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(bz)
        csv_file.close()
    '''
    with open('bz_30000.csv', 'a') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(bz)
        csv_file.close()
  
