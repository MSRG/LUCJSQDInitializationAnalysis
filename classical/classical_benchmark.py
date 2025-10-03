#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyscf
import pyscf.cc
import pyscf.mcscf

import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from tqdm.notebook import tqdm
import pandas as pd
import os, sys, psutil



# In[2]:


os.environ['OMP_NUM_THREADS'] = "64" 




# In[3]:


# for structure in glob("structures/*xyz"):
#     with open(structure,'r') as f:
#         structlines = f.readlines()
#     xyzlines = []    
#     for line in structlines:
#         if 'units'  not in line and 'symmetry'  not in line:
#             split = line.split()
#             if len(split)>1:
#                 xyzlines.append(split)

#     # with open(structure,'w') as g:
#         g.write(f"{len(xyzlines)}\n\n")
#         for a,x,y,z in xyzlines:
#             x,y,z = float(x),float(y),float(z)
#             g.write(f"{a} {x:>13.6f} {y:>13.6f} {z:>13.6f}\n")


# In[4]:


basis_sets = ['STO-3G','cc-pVDZ','aug-cc-pVDZ']


# In[5]:


active_spaces = pd.read_csv('../DDLUCJ_active_spaces_unfrozen.csv').dropna(axis=1)


# In[6]:


active_spaces


# In[ ]:


# All molecules are uncharged and closed-shell
open_shell = False
spin_sq = 0
energies = {}
#Iterate over basis sets
for b in basis_sets:
    energies[b] = {}
    # Iterate over DataFrame rows and find the structures in the directory
    for row in active_spaces.itertuples(index=False):
        structure_dict = row._asdict()
        molname = structure_dict['molecule']
        molfroz = structure_dict['n_frozen']
        molelec = structure_dict['n_electrons']
        molorb = structure_dict['num_orbitals']

        if 'GDB' in molname:
            structpath = f"./structures/{molname}.xyz"
        else:
            structpath = glob(f"./structures/{molname}*.xyz")[0]
        print(structpath)

        print(b,molname,(molelec,molorb))
        if os.path.exists(structpath):
            print(molname)


            # Build molecule
            mol = pyscf.gto.Mole()
            mol.build(
                atom=structpath,
                basis=b
            )


            # RHF
            scf = pyscf.scf.RHF(mol).run()
            RHF_energy = scf.e_tot
            # CCSD
            ccsd = pyscf.cc.CCSD(scf, frozen=molfroz).run()   
            CCSD_energy = ccsd.run().e_tot    
            # CASCI
            cas = pyscf.mcscf.CASCI(scf, molorb, molelec)
            CASCI_energy = cas.run().e_tot




            print(RHF_energy)
            print(CCSD_energy)
            print(CASCI_energy)

            energies[b][molname] = {"HF":RHF_energy,"CCSD":CCSD_energy,"CASCI":CASCI_energy}


# In[ ]:


# Flatten the dictionary
records = []
for basis_set, molecules in energies.items():
    for molecule, methods in molecules.items():
        for method, energy in methods.items():
            records.append((basis_set, molecule, method, energy))

# Create DataFrame
df = pd.DataFrame(records, columns=['Basis Set', 'Molecule', 'Method', 'Energy'])

# Set MultiIndex
# df.set_index(['Basis Set', 'Molecule', 'Method'], inplace=True)



# In[ ]:


df.to_csv('energies.csv')


# In[ ]:


df = pd.read_csv('energies.csv',index_col=0)
# df.set_index(['Basis Set', 'Molecule','Method'], inplace=True)
# df.loc[:,:,'HF'] = df.loc[:,:,'HF'] - df.loc[:,:,'CASCI']
# df.loc[:,:,'CCSD'] = df.loc[:,:,'CCSD'] - df.loc[:,:,'CASCI']
# df.loc[:,:,'CASCI'] = df.loc[:,:,'CASCI'] - df.loc[:,:,'CASCI']


# In[ ]:


df


# In[ ]:





# In[ ]:


g = sns.catplot(data=df.sort_values(by='Molecule'),    x="Molecule",    y="Energy",    hue="Method",    col="Basis Set",    kind="bar",    height=4,    aspect=1.2)
g.set_titles("{col_name}")
g.set_axis_labels("Molecule", "Energy")

# Move legend outside the plot
g._legend.set_bbox_to_anchor((1.05, 0.75))  # (x, y) position relative to the plot
g._legend.set_frame_on(True)               # Optional: adds a frame around the legend
g.set_xticklabels(rotation=45)
plt.tight_layout()
plt.show()


# In[ ]:




