#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pyscf
# import pyscf.cc
# import pyscf.mcscf
from pyscf import gto, scf, mcscf, cc
from pyscf.shciscf import shci


import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from tqdm.notebook import tqdm
import pandas as pd
import os, sys, time


# In[ ]:


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


# In[ ]:


basis_sets = ['STO-3G','cc-pVDZ','aug-cc-pVDZ']


# In[ ]:


active_spaces = pd.read_csv('../DDLUCJ_active_spaces_unfrozen.csv').dropna(axis=1)


# In[ ]:


structure_dict = active_spaces.query("molecule == 'ammonia'").iloc[0].to_dict()
molname = structure_dict['molecule']
molfroz = structure_dict['n_frozen']
molelec = structure_dict['n_electrons']
molorb = structure_dict['num_orbitals']
structpath = glob(f"./structures/{molname}*.xyz")[0]
print(molelec,molorb)


# In[ ]:


# t0 = time.time()
# # Build N2 molecule
# mol = gto.Mole()
# mol.build(
# verbose=3,
# atom=structpath,
# basis='sto-3g'
# )


# # RHF
# RHF = scf.RHF(mol).run()
# RHF_energy = RHF.e_tot
# # CCSD
# ccsd = cc.CCSD(RHF, frozen=molfroz).run()   
# CCSD_energy = ccsd.run().e_tot    
# # CASCI
# print(molorb, molelec)
# cas = mcscf.CASCI(RHF, molorb, molelec)
# CASCI_energy = cas.run().e_tot


# #
# # Multireference
# #

# mc = shci.SHCISCF(RHF, molorb, molelec)

# # mc.fcisolver.runtimeDir = "runtime"
# mc.fcisolver.nroots = 1
# # mc.fcisolver.davidsonTol = 1e-5
# # mc.fcisolver.dE = 1e-10
# # # mc.fcisolver.scratchDirectory = "scratch"
# # mc.fcisolver.nPTiter = 0
# # mc.fcisolver.DoRDM = True

# if not os.path.exists(mc.fcisolver.runtimeDir):
#     os.mkdir(mc.fcisolver.runtimeDir)

# if not os.path.exists(mc.fcisolver.scratchDirectory):
#     os.mkdir(mc.fcisolver.scratchDirectory)    
# mc.kernel()
# SHCI_energy = mc.e_tot

# print("Total Time:    ", time.time() - t0)

# print(RHF_energy)
# print(CCSD_energy)
# print(CASCI_energy)
# print(SHCI_energy)

# # File cleanup
# mc.fcisolver.cleanup_dice_files()
# if not os.path.exists(mc.fcisolver.scratchDirectory):
#     os.rmdir(mc.fcisolver.scratchDirectory)

# if not os.path.exists(mc.fcisolver.runtimeDir):    
#     os.rmdir(mc.fcisolver.runtimeDir)


# energies[b][molname] = {"HF":RHF_energy,"CCSD":CCSD_energy,"CASCI":CASCI_energy,"SHCI":CASCI_energy}


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
    
            t0 = time.time()
            # Build N2 molecule
            mol = gto.Mole()
            mol.build(
            verbose=3,
            atom=structpath,
            basis=b
            )
            
            # RHF
            RHF = scf.RHF(mol)
            RHF_energy = RHF.run().e_tot
            # CCSD
            ccsd = cc.CCSD(RHF, frozen=molfroz)
            CCSD_energy = ccsd.run().e_tot  

            # CASCI
            print(molorb, molelec)
            try:
                cas = mcscf.CASCI(RHF, molorb, molelec)
                CASCI_energy = cas.run().e_tot
            except MemoryError:
                CASCI_energy = None
            
            #
            # Multireference
            #
            
            mc = shci.SHCISCF(RHF, molorb, molelec)
            
            # mc.fcisolver.runtimeDir = "runtime"
            mc.fcisolver.nroots = 1
            mc.fcisolver.davidsonTol = 1e-5
            mc.fcisolver.dE = 1e-10
            # mc.fcisolver.scratchDirectory = "scratch"
            mc.fcisolver.nPTiter = 0
            mc.fcisolver.DoRDM = True
            
            if not os.path.exists(mc.fcisolver.runtimeDir):
                os.mkdir(mc.fcisolver.runtimeDir)
            
            if not os.path.exists(mc.fcisolver.scratchDirectory):
                os.mkdir(mc.fcisolver.scratchDirectory)    
            mc.kernel()
            SHCI_energy = mc.e_tot
            
            print("Total Time:    ", time.time() - t0)
            
            print(RHF_energy)
            print(CCSD_energy)
            print(CASCI_energy)
            print(SHCI_energy)
            
            # File cleanup
            mc.fcisolver.cleanup_dice_files()
            if not os.path.exists(mc.fcisolver.scratchDirectory):
                os.rmdir(mc.fcisolver.scratchDirectory)
            
            if not os.path.exists(mc.fcisolver.runtimeDir):    
                os.rmdir(mc.fcisolver.runtimeDir)
            
            
            energies[b][molname] = {"HF":RHF_energy,"CCSD":CCSD_energy,"CASCI":CASCI_energy,"SHCI":CASCI_energy}


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


# df = pd.read_csv('energies.csv',index_col=0)
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




