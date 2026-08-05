import os
import pandas as pd
from classical import run_DMRG

structure = '../../structures/methane50.xyz'
basis = 'STO-3G'
n_electrons = 10
num_orbitals = 9
sym = None
spin_sq = 0
charge = 0
n_jobs = 8
out_csv = "dmrg_Energy.csv"

energy = run_DMRG(structure, basis, n_electrons, num_orbitals, sym, spin_sq, charge, n_jobs=n_jobs)
df = pd.DataFrame([{"energy": energy}])
df.to_csv(out_csv, index=False)
print("Done writing dmrg_Energy.csv")
