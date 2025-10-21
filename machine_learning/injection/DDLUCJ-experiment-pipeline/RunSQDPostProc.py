#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import psutil
from functools import partial
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
# import xgboost as xgb

from glob import glob
# import psi4
# from helper_CC_ML_spacial import *

import pyscf
from pyscf import gto, scf, mcscf, cc

import ffsim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.primitives import StatevectorSampler, BitArray
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler

from qiskit_addon_sqd.fermion import SCIResult, diagonalize_fermionic_hamiltonian
from qiskit_addon_sqd.counts import bit_array_to_arrays


from ansatzmap import get_zigzag_physical_layout

from tqdm import tqdm

from DDLUCJ import DDLUCJ, GrabAmps


# In[ ]:


BasisDirs=glob('data/*')


# In[ ]:


energyDF=pd.read_csv("../../../classical/energies.csv",index_col=0)


# In[ ]:


moldf = pd.read_csv('molecules.csv')
activespacedf = pd.read_csv("active_spaces.csv")


# In[ ]:





# In[ ]:





# In[ ]:


BasisSets = ['STO-3G','cc-pVDZ','aug-cc-pVDZ']


# In[ ]:


# os.mkdir('counts')
service = QiskitRuntimeService()


# In[ ]:


"False"


# In[ ]:


def run(pathxyz,name,basis,n_electrons,num_orbitals,L,k):
    filecontents=f"""
import psutil
from functools import partial
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
# import xgboost as xgb

from glob import glob
# import psi4
# from helper_CC_ML_spacial import *

import pyscf
from pyscf import gto, scf, mcscf, cc

import ffsim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler

from qiskit_addon_sqd.fermion import SCIResult, diagonalize_fermionic_hamiltonian
from qiskit_addon_sqd.counts import bit_array_to_arrays
from qiskit.primitives import StatevectorSampler, BitArray


from ansatzmap import get_zigzag_physical_layout

from tqdm import tqdm

from DDLUCJ import DDLUCJ, GrabAmps    

ampdict = GrabAmps("{name}","{basis}")

t1, t2 = ampdict["{k}"]
initDDLUCJ = DDLUCJ(StructurePath="{pathxyz}", 
                    BasisSet="{basis}", 
                    NElec=int({n_electrons}),
                    NOrb=int({num_orbitals}),
                    injected=True,
                    t1=t1, 
                    t2=t2,
                    n_reps = int({L}),
                    optimization_level=3,
                    temp_dir="./",
                    clean_temp_dir=True,
                    n_jobs=64,                            
                    verbose=False)

counts = np.load(f"../counts/{name}_LUCJ_L{L}_{basis}_{k}.npz")
bitstrings = counts['bitstrings']
probarr = counts['probarr']
bitstrings = BitArray.from_bool_array(bitstrings)

result_history, result = initDDLUCJ(postprocess=True,BitArray=bitstrings) 

new_energy = result.energy + initDDLUCJ.nuclear_repulsion_energy
EnergyPath = f"../energies/{name}_LUCJ_L{L}_{basis}_{k}.txt" 
with open(EnergyPath,'w') as f:
    new_row={{"Basis Set": "{basis}", "Molecule": "{name}", "Method": f"LUCJ(L={L})/{k}", "Energy": new_energy}}
    for k,v in new_row.items():
        f.write(f"{{v}}\\n")

"""
    with open(f"./postprocess/{name}_LUCJ_L{L}_{basis}_{k}.py",'w') as f:
        f.write(filecontents)

    runfile=f"""#!/bin/bash
#SBATCH --time=0-12:00:00
#SBATCH -J {name}_LUCJ_L{L}_{basis}_{k}
#SBATCH --account=rrg-jacobsen-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mem-per-cpu=5000M
#SBATCH --error=job.e%J
#SBATCH --output=job.o%j



echo 'About to run python file'
module load python/3.10
module load openmpi
module load symengine rust
module load hdf5
module load openblas
export UCX_VFS_ENABLE=no
source /lustre09/project/6004825/gjones/ENV/bin/activate
export LD_LIBRARY_PATH=$EBROOTOPENBLAS/lib:$LD_LIBRARY_PATH
echo "Running in directory: $(pwd)"
python "{name}_LUCJ_L{L}_{basis}_{k}.py" 
echo "File run"    
"""
    with open(f"./postprocess/{name}_LUCJ_L{L}_{basis}_{k}.sh",'w') as f:
            f.write(runfile)


# In[ ]:


# os.mkdir('energies')


# In[ ]:


postprocessed = []
for i in tqdm(sorted(glob("./jobids/*txt")),desc='Running'):
                                                  
    with open(i,'r') as f:
        name,basis,k,L,JobID = [i.strip() for i in f.readlines()]
    print(name,basis,k,L,JobID)
    moldict = moldf[moldf['molecule']==name]

    n_electrons=moldict['n_electrons'].values[0]
    num_orbitals=moldict['num_orbitals'].values[0]
    xyzname = moldict['mol_filename'].values[0]
    pathxyz = os.path.join("../../../../classical/structures/",xyzname)



    JobPath = f"./jobids/{name}_LUCJ_L{L}_{basis}_{k}.txt" 
    EnergyPath = f"./energies/{name}_LUCJ_L{L}_{basis}_{k}.txt" 

    if os.path.exists(JobPath)==True and os.path.exists(EnergyPath)==False:
        print(f"Running {name}_LUCJ_L{L}_{basis}_{k}")
        run(pathxyz,name,basis,n_electrons,num_orbitals,L,k)


# In[ ]:




