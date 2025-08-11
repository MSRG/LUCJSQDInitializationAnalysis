import os
import random
import pandas as pd
import numpy as np
from glob import glob
import time
import joblib

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

random.seed(0)

# Most important features (top 5 by SHAP):
# doublecheck: Numerator of the MP2 t2-amplitude, two-electron integral <ik || ab>
# t2start: Initial MP2 t2-amplitude
# t2mag: Magnitude of the MP2 t2-amplitude
# orbdiff: Denominator of the MP2 t2-amplitude
# diag: Binary feature denoting whether a=b (virtual orbits are the same)
feat = ['doublecheck', 't2start', 't2mag', 'orbdiff', 'diag']

# Feature order in X
properties=['Evir1', 'Hvir1', 'Jvir1', 'Kvir1', 'Evir2', 'Hvir2', 'Jvir2', 'Kvir2', 'Eocc1', 'Jocc1', 'Kocc1', 'Hocc1','Eocc2', 'Jocc2', 'Kocc2', 'Hocc2', 'Jia1', 'Jia2', 'Kia1', 'Kia2','diag', 'orbdiff', 'doublecheck', 't2start', 't2mag', 't2sign', 'Jia1mag', 'Jia2mag','Kia1mag', 'Kia2mag','t2']

# get training molecules
with open("train_names.txt", "r") as f:
    lines = f.readlines()

train = [element[:-1] for element in lines]


# get testing molecules
with open("test_names.txt", "r") as f:
    lines = f.readlines()

test = [element[:-1] for element in lines]

train = [name.removesuffix('.xyz') for name in train]
test = [name.removesuffix('.xyz') for name in test]

data_dict = {
    os.path.basename(v).split('.')[0]: pd.read_pickle(v, compression='gzip')
    for v in glob('./PrasadData/aug-cc-pVDZ/data/alldata/*.pkl.gz') 
}

X_test=np.vstack([data_dict[i].drop(columns=['t2']).to_numpy() for i in test])
y_test=np.hstack([data_dict[i]['t2'].to_numpy() for i in test])

inf_mask = np.isinf(X_test).any(axis=1)

inf_indices = np.where(inf_mask)[0]

X_test = X_test[~inf_mask]
y_test = y_test[~inf_mask]

selected_indices = [properties.index(f) for f in feat]  # get top 5 features
X_test = X_test[:, selected_indices]

out_dir = "out"
try:
    os.makedirs(out_dir)
except FileExistsError:
    #   directory already exists
    pass

steps = [100]
for n in steps:
    t1 = time.time()
    # get training molecules
    with open(f"out/train_names_{n}.txt", "r") as f:
        lines = f.readlines()

    train = [element[:-1] for element in lines]
    
    X_train=np.vstack([data_dict[i].drop(columns=['t2']).to_numpy() for i in train])
    y_train=np.hstack([data_dict[i]['t2'].to_numpy() for i in train])

    inf_mask = np.isinf(X_train).any(axis=1)
    
    inf_indices = np.where(inf_mask)[0]
    
    X_train = X_train[~inf_mask]
    y_train = y_train[~inf_mask]

    X_train = X_train[:, selected_indices]
    
    # Instantiate model
    rfr=RandomForestRegressor()
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,                 
        min_samples_split=10,      
        min_samples_leaf=4,    
        max_features='sqrt',
        n_jobs=-1,
        random_state=42
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
    print(f"N: {n}, MAE: {mae}, MSE: {mse}, R2: {r2}")
    with open("out/faster_aug_cc_pVDZ_performance.txt", "a") as f:
        f.write(f"N: {n}, MAE: {mae}, MSE: {mse}, R2: {r2}\n")
    
    
    # save model and train test splits
    joblib.dump(model_pipeline, f"out/faster_aug_cc_pVDZ_model_{n}.pkl")
    
    splits = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
    }
    joblib.dump(splits, f"out/faster_aug_cc_pVDZ_data_splits_{n}.pkl")
    t2 = time.time()
    print(f"{n} samples took {t2-t1} seconds\n")

