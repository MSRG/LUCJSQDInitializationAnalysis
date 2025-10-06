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
import numpy as np


# In[ ]:


os.environ["OPENBLAS_CORETYPE"] = "generic"
os.environ["OPENBLAS_NUM_THREADS"] = "64"
os.environ["OMP_NUM_THREADS"] = "64"


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


df=pd.read_csv('energies.csv',index_col=0)


# In[ ]:


def RunClassical(StructurePath,BasisSet,NElec,NOrb,NFroz,verbose=True):
    """
    parameters
    ----------
    StructurePath: str
        Path to XYZ structure

    BasisSet: str
        Specify basis set

    NElec: int
        Number of electrons in the active space

    NOrb:
        Number of spatial orbitals in the active space

    NFroz:
        Number of frozen orbitals

    returns
    -------
    EnergyDict: dict
        Dictionary containing energies (keys: HF, CCSD, CASCI, SHCI)

    """
    t0 = time.time()
    # Build N2 molecule
    mol = gto.Mole()
    mol.build(
    atom=StructurePath,
    basis=BasisSet
    )

    # RHF
    RHF = scf.RHF(mol)
    RHF_energy = RHF.run().e_tot
    # CCSD
    ccsd = cc.CCSD(RHF, frozen=NFroz)
    CCSD_energy = ccsd.run().e_tot  

    # CASCI
    if verbose:
        print(f"CAS({NElec},{NOrb})")

    try:
        cas = mcscf.CASCI(RHF, NOrb, NElec)
        CASCI_energy = cas.run().e_tot
    except MemoryError:
        CASCI_energy = None

    #
    # Multireference
    #

    mc = shci.SHCISCF(RHF, NOrb, NElec)

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

    if verbose:
        print("Total Time:    ", time.time() - t0)
        print(f"HF: {RHF_energy:.6f} Eh")
        print(f"CCSD: {CCSD_energy:.6f} Eh")
        try:
            print(f"CASCI: {CASCI_energy:.6f} Eh")
        except TypeError: 
            print(f"CASCI: {None} Eh")
        print(f"SHCI: {SHCI_energy:.6f} Eh\n")

    # File cleanup
    mc.fcisolver.cleanup_dice_files()
    if not os.path.exists(mc.fcisolver.scratchDirectory):
        os.rmdir(mc.fcisolver.scratchDirectory)

    if not os.path.exists(mc.fcisolver.runtimeDir):    
        os.rmdir(mc.fcisolver.runtimeDir)

    EnergyDict = {"HF":RHF_energy,"CCSD":CCSD_energy,"CASCI":CASCI_energy,"SHCI":SHCI_energy}
    return EnergyDict


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

        # Find path
        if 'GDB' in molname:
            structpath = f"./structures/{molname}.xyz"
        else:
            structpath = glob(f"./structures/{molname}*.xyz")[0]

        if os.path.exists(structpath):
            print(structpath)
            print(b,molname,(molelec,molorb))
            print(molname)
            energies[b][molname] = RunClassical(structpath,b,molelec,molorb,molfroz)


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




