import os
import numpy as np
import pandas as pd

import pyscf
from pyscf import gto, scf, mcscf, cc
import ffsim

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.primitives import StatevectorSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from ansatzmap import get_zigzag_physical_layout
from DDLUCJ import DDLUCJ, GrabAmps

ampdict = GrabAmps("methanol", "STO-3G")
t1, t2 = ampdict["ML"]

initDDLUCJ = DDLUCJ(
    StructurePath="../../../classical/structures/methanol22.xyz",
    BasisSet="STO-3G",
    NElec=int(18),
    NOrb=int(14),
    Symmetry=False,  # avoid buggy orbsym path (same fix as block2/DMRG)
    injected=True,
    t1=t1,
    t2=t2,
    n_reps=int(1),
    backend="statevector",
    optimization_level=3,
    shots=int(100000000),
    temp_dir="./",
    clean_temp_dir=True,
    n_jobs=1,
    num_batches=10,
    max_iterations=5,
    samples_per_batch=1000,
    verbose=True
)

total_energy, subspace_dim = initDDLUCJ(postprocess=True, usefulqrum=True)
energy_val = float(np.squeeze(total_energy))

EnergyPath = "../energies/methanol_LUCJ_L1_STO-3G_ML_StateVector_Shots100000000.txt"
with open(EnergyPath, 'w') as f:
    f.write(f"Basis Set: STO-3G\n")
    f.write(f"Molecule: methanol\n")
    f.write(f"Method: LUCJ(L=1)/ML\n")
    f.write(f"Energy: {energy_val}\n")
    f.write(f"Subspace Dimension: {subspace_dim}\n")
