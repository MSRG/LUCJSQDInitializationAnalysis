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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pickle
import xgboost as xgb
import json
from glob import glob
import psi4
from helper_CC_ML_spacial import *

import pyscf
import pyscf.cc
import pyscf.mcscf
import ffsim
import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler

home = os.path.expanduser("~")
molecules_dir = os.path.join(home, "DDLUCJ", "machine_learning", "injection","sqd-experiment-pipeline", "molecules.csv")

basis_sets = ['STO-3G','cc-pVDZ','aug-cc-pVDZ']
basis = basis_sets[2]

# Retrieve molecule filenames, formulas, and active spaces in minimal basis set
molecules = pd.read_csv(molecules_dir)
results = pd.DataFrame(index=molecules['molecule'], columns=basis_sets, dtype=float)
for basis in basis_sets:
    for molecule in molecules['molecule']:

        mol_row = molecules[molecules['molecule'] == molecule].iloc[0]
        mol_file =  mol_row['mol_filename']

        # Load Molecule in Python and prepare Helper Object for CCSD calculations
        with open(os.path.join(os.path.expanduser('~'),'DDLUCJ', 'machine_learning', 'data', mol_file),'r') as f:
            text=f.read()

        lines = text.split('\n')

        # Slice the list to exclude the last two lines
        modified_lines = lines[:-3]

        # Join the remaining lines back into a single string
        text_cleaned = '\n'.join(modified_lines)

        # Specify molecule properties
        open_shell = False   
        spin_sq = 0          # singlet

        mol = pyscf.gto.Mole()
        mol.build(
        atom=text_cleaned,
        basis=basis,
        symmetry="c1",
        )

        # Get molecular integrals
        scf = pyscf.scf.RHF(mol).run()

        # Define active space
        nmo = scf.mo_coeff.shape[1]
        num_orbitals = mol_row['num_orbitals']
        n_frozen = nmo - num_orbitals
        n_electrons = mol_row['n_electrons']     # all valence electrons

        # Define active space
        active_space = range(n_frozen, mol.nao_nr())
        num_elec_a = (n_electrons + mol.spin) // 2
        num_elec_b = (n_electrons - mol.spin) // 2
        cas = pyscf.mcscf.CASCI(scf, num_orbitals, (num_elec_a, num_elec_b))
        mo = cas.sort_mo(active_space, base=0)
        hcore, nuclear_repulsion_energy = cas.get_h1cas(mo)
        eri = pyscf.ao2mo.restore(1, cas.get_h2cas(mo), num_orbitals)

        # Compute exact energy
        exact_energy = cas.run().e_tot
        results.loc[molecule, basis] = exact_energy

        out_dir = os.path.join(home, "DDLUCJ", "machine_learning", "injection","sqd-experiment-pipeline", "data", basis, "integrals")
        os.makedirs(out_dir, exist_ok=True)
        fname = f"{molecule}_{basis}_cas_integrals.npz"
        path  = os.path.join(out_dir, fname)

        np.savez_compressed(
            path,
            hcore=hcore.astype(np.float64),
            eri=eri.astype(np.float64),
            mo=mo.astype(np.float64),           # keep the MO coeffs you used for CAS
            comment="MO-basis CAS pack; mo matches active-space order"
        )
out_dir = os.path.join(home, "DDLUCJ", "machine_learning", "injection","sqd-experiment-pipeline", "data")
os.makedirs(out_dir, exist_ok=True)
out_csv = os.path.join(out_dir, "casci_energies_by_basis.csv")
results.to_csv(out_csv)