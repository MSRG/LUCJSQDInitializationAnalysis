import psutil
import sys
# !{sys.executable} --version
# !{sys.executable} -m pip install shap --upgrade 
import joblib
import time
from shutil import copy
import numpy as np
import pandas as pd

import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from tqdm.notebook import tqdm
import seaborn as sns
from collections import Counter

from glob import glob
import psi4
from helper_CC_ML_spacial import *

import random
random.seed(0)
print(os.getcwd())
os.chdir(os.path.join(os.getcwd(), 'machine_learning'))

# Most important features (top 5 by SHAP):
# doublecheck: Numerator of the MP2 t2-amplitude, two-electron integral <ik || ab>
# t2start: Initial MP2 t2-amplitude
# t2mag: Magnitude of the MP2 t2-amplitude
# orbdiff: Denominator of the MP2 t2-amplitude
# diag: Binary feature denoting whether a=b (virtual orbits are the same)
feat = ['doublecheck', 't2start', 't2mag', 'orbdiff', 'diag']

# Feature order in X
properties=['Evir1', 'Hvir1', 'Jvir1', 'Kvir1', 'Evir2', 'Hvir2', 'Jvir2', 'Kvir2', 'Eocc1', 'Jocc1', 'Kocc1', 'Hocc1','Eocc2', 'Jocc2', 'Kocc2', 'Hocc2', 'Jia1', 'Jia2', 'Kia1', 'Kia2','diag', 'orbdiff', 'doublecheck', 't2start', 't2mag', 't2sign', 'Jia1mag', 'Jia2mag','Kia1mag', 'Kia2mag','t2']

basis = 'aug-cc-pVDZ'   # change for a different level of theory


psi4.set_num_threads(12)
psi4.set_memory('8 GB')
data_dir = 'PrasadData'
basisdirs = os.path.join(data_dir,basis)

try:
    os.makedirs(basisdirs)
except FileExistsError:
    #   directory already exists
    pass

root = 'data'
struct_names = os.path.basename(root)
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


# get training molecules
with open("train_names.txt", "r") as f:
    lines = f.readlines()

train = [element[:-1] for element in lines]

# get testing molecules
with open("test_names.txt", "r") as f:
    lines = f.readlines()

test = [element[:-1] for element in lines]
structures = train + test # all structures

all_strucs = glob(os.path.join(root, '*.xyz'))

# save allstrucs
with open("all_strucs.txt", "a") as f:
    f.write(f"{all_strucs}")


data_dict = {}
for fn in structures:
    struct = f"data/{fn}"

    with open("curr_strucs_aug.txt", "a") as f:
        f.write(f"{struct}")
    
    copy(struct,copiedstructures)


    with open(struct, 'r') as f:
        text = f.read()

    mol = psi4.geometry(text)
    
    psi4.core.clean()
    psi4.core.be_quiet()
    
    psi4.set_options({'basis': basis,
                        'scf_type':     'pk',
                        'reference':    'rohf',
                        'mp2_type':     'conv',
                        'e_convergence': 1e-8,
                        'd_convergence': 1e-8})
    try:
        rhf_e, scf_wfn = psi4.energy('scf', return_wfn=True)
        scf_e, scf_wfn = psi4.energy('scf', return_wfn=True)
        A=HelperCCEnergy(mol, rhf_e, scf_wfn,freeze_core=False)
        A.compute_energy()
        
        data=pd.DataFrame(np.array([getattr(A,attr).flatten() for attr in properties]).T,columns=properties)
        data_dict[struct.split('_')[0]]=data
    except:
        pass   
        
    psi4.core.clean()
    
# Save gzip files for data processing
for k,v in sorted(data_dict.items()):
    v.to_pickle(os.path.join(gzdata,f"{os.path.basename(k.replace('.xyz',''))}.pkl.gz"), compression='gzip')


# Get files
data_dict = {
    os.path.basename(v).split('.')[0]: pd.read_pickle(v, compression='gzip')
    for v in glob('./PrasadData/aug-cc-pVDZ/data/alldata/*.pkl.gz') 
}


