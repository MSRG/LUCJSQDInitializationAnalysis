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
from sklearn.pipeline import Pipeline

from tqdm.notebook import tqdm
import seaborn as sns
from collections import Counter

from glob import glob
import psi4
from helper_CC_ML_spacial import *

import random
random.seed(0)


# Most important features (top 5 by SHAP):
# doublecheck: Numerator of the MP2 t2-amplitude, two-electron integral <ik || ab>
# t2start: Initial MP2 t2-amplitude
# t2mag: Magnitude of the MP2 t2-amplitude
# orbdiff: Denominator of the MP2 t2-amplitude
# diag: Binary feature denoting whether a=b (virtual orbits are the same)
feat = ['doublecheck', 't2start', 't2mag', 'orbdiff', 'diag']

basis = 'STO-3G'   # change for a different level of theory


psi4.set_num_threads(12)

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

all_strucs = glob(os.path.join(root, '*.xyz'))


# get training molecules
with open("train_names.txt", "r") as f:
    lines = f.readlines()

train = [element[:-1] for element in lines]

# get testing molecules
with open("test_names.txt", "r") as f:
    lines = f.readlines()

test = [element[:-1] for element in lines]
structures = train + test # all structures

data_dict = {}
for fn in structures:
    struct = f"data/{fn}"
    
    copy(struct,copiedstructures)
    
    with open(struct,'r') as f:
        lines=f.readlines()

    clean_lines = lines[:-2] # drop symmetry line
    num_atoms = len(clean_lines)
    text = f"{num_atoms}\nGenerated\n" + ''.join(clean_lines)
        
    qmol = psi4.qcdb.Molecule.from_string(text, dtype='xyz')
    mol = psi4.geometry(qmol.create_psi4_string_from_molecule()+ 'symmetry c1')                

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
    print(k)
    v.to_pickle(os.path.join(gzdata,f"{os.path.basename(k.replace('.xyz',''))}.pkl.gz"), compression='gzip')


# Get files
data_dict = {
    os.path.basename(v).split('.')[0]: pd.read_pickle(v, compression='gzip')
    for v in glob('./PrasadData/STO-3G/data/alldata/*.pkl.gz') 
}

#
X_train=[]
y_train=[]
X_test=[]
y_test=[]

X_train=np.vstack([data_dict[i].drop(columns=['t2']).to_numpy() for i in train])
y_train=np.hstack([data_dict[i]['t2'].to_numpy() for i in train])

X_test=np.vstack([data_dict[i].drop(columns=['t2']).to_numpy() for i in test])
y_test=np.hstack([data_dict[i]['t2'].to_numpy() for i in test])


#####
inf_mask = np.isinf(X_train).any(axis=1)

inf_indices = np.where(inf_mask)[0]

X_train = X_train[~inf_mask]
y_train = y_train[~inf_mask]


#####
inf_mask = np.isinf(X_test).any(axis=1)

inf_indices = np.where(inf_mask)[0]

X_test = X_test[~inf_mask]
y_test = y_test[~inf_mask]


#####


properties = data_dict[train[0]].drop(columns='t2').columns.tolist()  # all features

selected_indices = [properties.index(f) for f in feat]  # get top 5 features
X_train = X_train[:, selected_indices]
X_test = X_test[:, selected_indices]

# Instantiate model

rfr=RandomForestRegressor()
model = RandomForestRegressor(**{'bootstrap': True,
    'ccp_alpha': 0.0,
    'criterion': 'squared_error',
    'max_depth': None,
    'max_features': 1.0,
    'max_leaf_nodes': None,
    'max_samples': None,
    'min_impurity_decrease': 0.0,
    'min_samples_leaf': 1,
    'min_samples_split': 2,
    'min_weight_fraction_leaf': 0.0,
    'monotonic_cst': None,
    'n_estimators': 300,
    'n_jobs': -1,
    'oob_score': False,
    'random_state': None,
    'verbose': 0,
    'warm_start': False}
)

model_pipeline = Pipeline([
    ('scaler', MinMaxScaler(feature_range=(-1,1))),
    ('regressor', model)
])

# fit model
model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)

# compute performance metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# save metrics
with open("sto_3g_performance.txt", "a") as f:
    f.write(f"MAE: {mae}, MSE: {mse}, R2: {r2}")


# save model and train test splits
joblib.dump(model_pipeline, 'sto_3g_model_07_07_2025.pkl')

splits = {
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train,
    'y_test': y_test,
}
joblib.dump(splits, 'sto_3g_data_splits.pkl')



