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

ampdict = GrabAmps("buta-1,3-diene", "cc-pVDZ")
t1, t2 = ampdict["MP2"]

initDDLUCJ = DDLUCJ(
    StructurePath="../../../classical/structures/GDB04_53.xyz",
    BasisSet="cc-pVDZ",
    NElec=int(30),
    NOrb=int(26),
    Symmetry=False,  # avoid buggy orbsym path (same fix as block2/DMRG)
    injected=True,
    t1=t1,
    t2=t2,
    n_reps=int(5),
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

EnergyPath = "../energies/buta-1,3-diene_LUCJ_L5_cc-pVDZ_MP2_StateVector_Shots100000000.txt"
with open(EnergyPath, 'w') as f:
    f.write(f"Basis Set: cc-pVDZ\n")
    f.write(f"Molecule: buta-1,3-diene\n")
    f.write(f"Method: LUCJ(L=5)/MP2\n")
    f.write(f"Energy: {energy_val}\n")
    f.write(f"Subspace Dimension: {subspace_dim}\n")
