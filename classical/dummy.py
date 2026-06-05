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


active_spaces = pd.read_csv('../DDLUCJ_active_spaces_unfrozen.csv').dropna(axis=1)
molname = 'GDB04_53'
structure_dict = active_spaces.query("molecule =='GDB04_53'").iloc[0].to_dict()
molfroz = structure_dict['n_frozen']
molelec = structure_dict['n_electrons']
molorb = structure_dict['num_orbitals']
b = 'STO-3G'

# if 'GDB' in molname:
structpath = f"./structures/{molname}.xyz"
# else:
#     structpath = glob(f"./structures/{molname}*.xyz")[0]
#     print(structpath)

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