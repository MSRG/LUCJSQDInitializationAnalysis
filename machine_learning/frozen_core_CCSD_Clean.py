#!/usr/bin/env python
# coding: utf-8
import psutil
import sys
# !{sys.executable} --version
# !{sys.executable} -m pip install shap --upgrade 
import joblib
import time
from shutil import copy
import numpy as np
import pandas as pd
#import tensorflow as tf
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from tqdm.notebook import tqdm
import seaborn as sns
from collections import Counter

from glob import glob
import psi4
from helper_CC_ML_spacial import *

# SHAP
import shap

availablememory = psutil.virtual_memory().available / (1024.0 ** 3)
threads_count = int(psutil.cpu_count() * 0.75)

print(f"Running on {threads_count} threads using {availablememory} Gb")
properties=['Evir1', 'Hvir1', 'Jvir1', 'Kvir1', 'Evir2', 'Hvir2', 'Jvir2', 'Kvir2', 'Eocc1', 'Jocc1', 'Kocc1', 'Hocc1','Eocc2', 'Jocc2', 'Kocc2', 'Hocc2', 'Jia1', 'Jia2', 'Kia1', 'Kia2','diag', 'orbdiff', 'doublecheck', 't2start', 't2mag', 't2sign', 'Jia1mag', 'Jia2mag','Kia1mag', 'Kia2mag','t2']




data_dir = 'PrasadData'
basis_sets = ['STO-3G','cc-pVDZ','aug-cc-pVDZ']




for basis in basis_sets:
    print(basis)
    basisdirs = os.path.join(data_dir,basis)
    try:
        os.makedirs(basisdirs)
    except FileExistsError:
        # directory already exists
        pass
        
    for path in glob('./ddcc-voglab2019/watertest/*water')+['./ddcc-voglab2019/water']:
        print(path)
        struct_names = os.path.basename(path)
        structdirs = os.path.join(basisdirs,struct_names)
        copiedstructures = os.path.join(structdirs,'structures')
        gzdata = os.path.join(structdirs,'alldata')
        mldata = os.path.join(structdirs,'MLData')
        
        try:
            os.makedirs(structdirs)
            os.makedirs(copiedstructures)
            os.makedirs(gzdata)
            os.makedirs(mldata)
        except FileExistsError:
            # directory already exists
            pass        
            
        structures = glob(os.path.join(path,'*xyz'))

        data_dict={}
        for struct in structures:
            k = os.path.basename(struct.split('_')[0])
            if os.path.exists(os.path.join(gzdata,f"{os.path.basename(k.replace('.xyz',''))}.pkl.gz"))==False:

                print(struct)
                copy(struct,copiedstructures)
                
                with open(struct,'r') as f:
                    text=f.read()
                
                xyz=False
                if xyz==True: 
                    qmol = psi4.qcdb.Molecule.from_string(text, dtype='xyz')
                    mol = psi4.geometry(qmol.create_psi4_string_from_molecule()+ 'symmetry c1')                
                else:                                
                    mol = psi4.geometry(text)  
        
                psi4.core.clean()
                psi4.core.be_quiet()
                psi4.set_memory(f'{int(np.floor(availablememory) * 0.8)} GB')
                psi4.core.set_output_file('output.dat', False)
                psi4.core.set_num_threads(threads_count)
                psi4.set_options({'basis': basis,
                                  'scf_type':     'pk',
                                  'reference':    'rhf',
                                  'mp2_type':     'conv',
                                  'e_convergence': 1e-8,
                                  'd_convergence': 1e-8})
                try:
                    rhf_e, scf_wfn = psi4.energy('scf', return_wfn=True)
                    scf_e, scf_wfn = psi4.energy('scf', return_wfn=True)
                    A=HelperCCEnergy(mol, rhf_e, scf_wfn,freeze_core=True)
                    
                    A.compute_energy()
                    
                    data=pd.DataFrame(np.array([getattr(A,attr).flatten() for attr in properties]).T,columns=properties)
                    data.to_pickle(os.path.join(gzdata,f"{k.replace('.xyz','')}.pkl.gz"), compression='gzip')
                    psi4.core.clean()
                except:
                    pass        
