from shutil import copy
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from glob import glob
import psi4
import sys
sys.path.insert(0, '/home/amousso3/DDLUCJ/check_amplitudes/')
from helper_CC_ML_spacial import *


# === 1. Molecule Setup ===
with open('/home/amousso3/DDLUCJ/check_amplitudes/diatomics/NN.xyz','r') as f:
    text=f.read()

qmol = psi4.qcdb.Molecule.from_string(text, dtype='xyz')
mol = psi4.geometry(qmol.create_psi4_string_from_molecule()+ 'symmetry c1')   


psi4.core.clean()
psi4.core.be_quiet()

# === 2. Set Options and Run RHF ===

psi4.set_options({'basis': 'STO-3G',
                  'scf_type':     'pk',
                  'reference':    'rhf',
                  'mp2_type':     'conv',
                  'e_convergence': 1e-8,
                  'd_convergence': 1e-8})

rhf_e, scf_wfn = psi4.energy('scf', return_wfn=True)

# === 3. Use HelperCCEnergy ===
A = HelperCCEnergy(mol, rhf_e, scf_wfn, freeze_core=True)
A.compute_energy()

# === 4. Define Active Space ===
n_frozen = A.nfzc  # automatically determined by HelperCCEnergy if freeze_core=True
nmo = A.nmo
active_orbitals = list(range(n_frozen, nmo))

num_orbitals = len(active_orbitals)
n_elec_total = A.ndocc * 2
active_elec = n_elec_total - 2 * n_frozen
num_elec_a = int((active_elec + mol.multiplicity() - 1) // 2)
num_elec_b = int((active_elec - mol.multiplicity() + 1) // 2)

# === 5. Extract Active-Space Integrals ===
hcore = A.H1[np.ix_(active_orbitals, active_orbitals)]

# Get full MO ERI tensor and slice to active space
# No slicing — this is already the active-space ERI
eri = A.get_MO('aaaa')


# === 6. Use CCSD Final Energy as "exact" reference ===
exact_energy = A.FinalEnergy + rhf_e

# === 7. Optionally run Psi4’s built-in CCSD for comparison ===
psi4.set_options({'freeze_core': 'true'})
ccsd_e, ccsd_wfn = psi4.energy('ccsd', return_wfn=True)

# === 8. Extract Amplitudes ===
t1 = A.t1
t2 = A.t2

# === 9. Print Summary ===
print("CCSD correlation energy (Helper):", A.ccsd_corr_e)
print("Total CCSD energy (Helper):      ", A.FinalEnergy)
print("H1 shape:", hcore.shape)
print("ERI shape:", eri.shape)
print("Active electrons: (α, β) =", (num_elec_a, num_elec_b))
print("Exact (CCSD) Energy:", exact_energy)
