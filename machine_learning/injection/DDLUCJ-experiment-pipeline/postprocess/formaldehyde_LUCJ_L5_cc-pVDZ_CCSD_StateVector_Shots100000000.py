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

ampdict = GrabAmps("formaldehyde", "cc-pVDZ")
t1, t2 = ampdict["CCSD"]

initDDLUCJ = DDLUCJ(
    StructurePath="../../../classical/structures/formaldehyde138.xyz",
    BasisSet="cc-pVDZ",
    NElec=int(16),
    NOrb=int(12),
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

EnergyPath = "../energies/formaldehyde_LUCJ_L5_cc-pVDZ_CCSD_StateVector_Shots100000000.txt"
with open(EnergyPath, 'w') as f:
    f.write(f"Basis Set: cc-pVDZ\n")
    f.write(f"Molecule: formaldehyde\n")
    f.write(f"Method: LUCJ(L=5)/CCSD\n")
    f.write(f"Energy: {energy_val}\n")
    f.write(f"Subspace Dimension: {subspace_dim}\n")
