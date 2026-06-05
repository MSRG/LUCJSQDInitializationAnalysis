#!/usr/bin/env python
# coding: utf-8

# In[13]:


#!/usr/bin/env python
# coding: utf-8
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from tqdm.notebook import tqdm
import pandas as pd
import os, sys, time
import numpy as np
from dlucj.classical import run_dice_addon



# In[3]:


os.environ["OPENBLAS_CORETYPE"] = "generic"
os.environ["OPENBLAS_NUM_THREADS"] = "64"
os.environ["OMP_NUM_THREADS"] = "64"


# In[4]:


basis_sets = ['STO-3G','cc-pVDZ','aug-cc-pVDZ']

active_spaces = pd.read_csv('../DDLUCJ_active_spaces_unfrozen.csv').dropna(axis=1)


df=pd.read_csv('energies.csv',index_col=0)


# In[15]:


help(run_dice_addon)


# In[21]:


# All molecules are uncharged and closed-shell
open_shell = False
spin_sq = 0
energies = {}

#Iterate over basis sets
for b in basis_sets:
    energies[b] = {}
    # Iterate over DataFrame rows and find the structures in the directory
    for row in list(active_spaces.itertuples(index=False)):
        structure_dict = row._asdict()
        molname = structure_dict['molecule']
        molfroz = structure_dict['n_frozen']
        molelec = structure_dict['n_electrons']
        molorb = structure_dict['num_orbitals']

        # Find path
        if 'GDB' in molname:
            structpath = f"./structures/{molname}.xyz"
        else:
            structpath = glob(f"./structures/{molname}*.xyz")[0]

        if os.path.exists(structpath):
            print(structpath)
            print(b,molname,(molelec,molorb))
            print(molname)
            df = run_dice_addon(structpath, b, range(molorb), 'C1', 0, 0, molfroz,n_jobs=8)
            energies[b][molname] = df





# # Flatten the dictionary
# records = []
# for basis_set, molecules in energies.items():
#     for molecule, methods in molecules.items():
#         for method, energy in methods.items():
#             records.append((basis_set, molecule, method, energy))

# # Create DataFrame
# df = pd.DataFrame(records, columns=['Basis Set', 'Molecule', 'Method', 'Energy'])

# # Set MultiIndex
# # df.set_index(['Basis Set', 'Molecule', 'Method'], inplace=True)






# df.to_csv('energies.csv')









# In[35]:


stackeddf=[]
for k,v in energies.items():
    for k1,v1 in v.items():
        df = v1.copy()
        df['Name']=k1
        stackeddf.append(df)


# In[37]:


pd.concat(stackeddf).to_csv('energies.csv')


# In[ ]:




