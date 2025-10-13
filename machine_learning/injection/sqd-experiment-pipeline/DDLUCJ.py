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

from ansatzmap import get_zigzag_physical_layout


class DDLUCJ:
    def __init__(self,StructurePath, 
                 BasisSet, 
                 NElec,
                 NOrb,
                 NFroz=0,
                 Symmetry="C1",
                 Spin=0,
                 injected=False,
                 t1=None, 
                 t2=None,
                 n_reps = 1,
                 channel = None,
                 instance = None,
                 backend = None,         
                 optimization_level=3,
                 shots = 10_000,
                 energy_tol = 1e-08,
                 occupancies_tol = 1e-05,
                 max_iterations = 100,
                 num_batches = 1,
                 samples_per_batch = 300,
                 symmetrize_spin = True,
                 carryover_threshold = 1e-4,
                 max_cycle = 200,
                 temp_dir="./",
                 clean_temp_dir=False,
                 n_jobs=None,
                 verbose=False
                ):
        """
        Initialize the method
        
        parameters
        ----------
        StructurePath: str
            Path to xyz structure
        
        BasisSet: str
            Basis set
        
        NElec: int
            Number of electrons in the active space
        
        NOrb: int
            Number of spatial orbitals in the active space
        
        NFroz: int
            Number of frozen orbitals 
            (default = 0)
        
        Symmetry: str
            Molecular point group 
            (default = C1; I don't think symmetry is implemented in DDCC...)

        Spin: int
            Number of unpaired electrons (2S)
            (default = 0; singlet)
        
        injected: bool
            Flag to say we are injecting t1/t2-amplitudes
            (default = False; run PySCF)
        
        t1: np.ndarray
            Injected t1-amplitudes
            (default = None; run PySCF)
            
        t2: np.ndarray
            Injected t1-amplitudes
            (default = None; run PySCF)            

        n_reps: int
            Number of layers/repetitions in the LUCJ circuit
            (default = 1)
            
        channel: str
            Name of IBM Quantum channel
            (default = None)
         
         instance: str
            IBM Quantum instance
            (default = None)
         
         backend: str
            IBM Quantum backend
            (default = None)        
         
         optimization_level: int
             Circuit optimization level
             (default = 3)
         
         shots: int
             Number of evaluations on device
             (default = 10_000)
         
         energy_tol: float
             Tolerance for the recovered energy 
             (default = 1e-08)
         
         occupancies_tol:
             Tolerance for the occupation numbers
             (default = 1e-05)
         
         max_iterations: int
             (default = 100)
         
         num_batches: int
             (default = 1)
         
         samples_per_batch: int
             (default = 300)
         
         symmetrize_spin: bool
             (default = True)
         
         carryover_threshold: float
             (default = 1e-4)
         
         max_cycle: int
             (default = 200)
         
         temp_dir: str
             (default = "./")
         
         clean_temp_dir: bool
             (default = False)
         
         n_jobs: int
             (default = None)
         
         verbose: bool
             (default = False)
        """
        # PySCF options
        self.StructurePath=StructurePath
        self.BasisSet=BasisSet
        self.Spin=Spin
        self.Symmetry=Symmetry
        self.NElec=NElec
        self.NOrb=NOrb
        self.NFroz=NFroz

        # Circuit setup
        self.injected = injected
        self.t1=t1
        self.t2=t2
        self.n_reps = n_reps

        # Runtime args
        self.channel = channel
        self.instance = instance 
        self.backend = backend
        self.optimization_level = optimization_level
        self.shots = shots

        # SQD and configuration recovery
        self.energy_tol = energy_tol
        self.occupancies_tol = occupancies_tol
        self.max_iterations = max_iterations
        self.num_batches = num_batches
        self.samples_per_batch = samples_per_batch
        self.symmetrize_spin = symmetrize_spin
        self.carryover_threshold = carryover_threshold
        self.max_cycle = max_cycle

        # Dice plugin options
        self.temp_dir=temp_dir
        self.clean_temp_dir=clean_temp_dir
        self.n_jobs=n_jobs

        self.verbose = verbose
        
    def Initialize(self):
        """
        Initialize PySCF to return integrals, active space, etc.
        """
        mol = gto.Mole()
        # mol.build()
        # mol.symmetry = False
        mol.build(
            atom=self.StructurePath,
            basis=self.BasisSet,
            symmetry=self.Symmetry,
            spin=self.Spin
        )
        
        RHF = scf.RHF(mol).run()
        cas = mcscf.CASCI(RHF, self.NOrb, self.NElec,ncore=self.NFroz)
    
        # cas = pyscf.mcscf.CASCI(scf, num_orbitals, num_elec_a+num_elec_b)
        active_space = list(range(cas.ncore,cas.ncore+cas.ncas))
        if self.verbose:
            print(self.NOrb, self.NElec,self.NFroz)
            print(active_space)
        # print(num_orbitals, (num_elec_a, num_elec_b))
        self.mo = cas.sort_mo(active_space, base=0)
        self.hcore, self.nuclear_repulsion_energy = cas.get_h1cas(self.mo)
        self.eri = pyscf.ao2mo.restore(1, cas.get_h2cas(self.mo), self.NOrb)   

    def Circuit(self):
        # Add size safety check for the amplitudes!
        if self.injected == False and self.t1==None and self.t2==None:
            # Get CCSD t2 amplitudes for initializing the ansatz
            ccsd = pyscf.cc.CCSD(scf, frozen=range(self.NFroz)).run()
            self.t1 = ccsd.t1
            self.t2 = ccsd.t2
        print(self.t1.shape, self.t2.shape)
        
        Nocc, NVirt = self.t1.shape 
        Nact = self.NOrb - self.NFroz
        NVirtSlice= Nact - Nocc
        self.t1 = self.t1[self.NFroz:self.NOrb,:NVirtSlice]
        self.t2 = self.t2[self.NFroz:self.NOrb,self.NFroz:self.NOrb,:NVirtSlice,:NVirtSlice]
        
        print(self.t1.shape, self.t2.shape)
        alpha_alpha_indices = [(p, p + 1) for p in range(self.NOrb - 1)]
        alpha_beta_indices = [(p, p) for p in range(0, self.NOrb, 4)]
         
         
        ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
            t2=self.t2,
            t1=self.t1,
            n_reps=self.n_reps,
            interaction_pairs=(alpha_alpha_indices, alpha_beta_indices),
            # Setting optimize=True enables the "compressed" factorization
            optimize=True,
            # Limit the number of optimization iterations to prevent the code cell from running
            # too long. Removing this line may improve results.
            options=dict(maxiter=1000),
        )
         
        # create an empty quantum circuit
        qubits = QuantumRegister(2 * self.NOrb, name="q")
        circuit = QuantumCircuit(qubits)
        
        # prepare Hartree-Fock state as the reference state and append it to the quantum circuit
        circuit.append(ffsim.qiskit.PrepareHartreeFockJW(int(self.NOrb), (int(self.NElec//2),int(self.NElec//2))), qubits)
         
        # apply the UCJ operator to the reference state
        circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), qubits)
        circuit.measure_all()            
        self.circuit = circuit
        

    def Transpile(self):

        self.service = QiskitRuntimeService(channel=self.channel,instance=self.instance)

            
            
        if self.backend==None:
            self.backend = self.service.least_busy(operational=True, simulator=False)
        else:
            self.backend = self.service.backend(self.backend)
            
        if self.verbose:
            print(f"Using backend {self.backend.name}")
            
        initial_layout, _ = get_zigzag_physical_layout(self.NOrb, backend=self.backend)
         
        pass_manager = generate_preset_pass_manager(
            optimization_level=self.optimization_level, backend=self.backend, initial_layout=initial_layout
        )
         
        # with PRE_INIT passes
        # We will use the circuit generated by this pass manager for hardware execution
        pass_manager.pre_init = ffsim.qiskit.PRE_INIT
        self.isa_circuit = pass_manager.run(self.circuit)
        if self.verbose:
            print(f"Gate counts (w/ pre-init passes): {self.isa_circuit.count_ops()}")

    def RunDevice(self):
        if self.JobID==None:
            sampler = Sampler(mode=self.backend)
            job = sampler.run([self.isa_circuit], shots=self.shots)
            primitive_result = job.result()
            pub_result = primitive_result[0]
            self.bit_array = pub_result.data.meas
            if self.verbose:
                print(f"Qiskit Runtime Job ID: {job.job_id()}")
                
            self.runtimejob = job.job_id()
        else:
            if self.verbose:
                print(f"{self.JobID}")            
            job = self.service.job(self.JobID)
            primitive_result = job.result()
            pub_result = primitive_result[0]
            self.bit_array = pub_result.data.meas

    def Postprocess(self):
    
    
        # Pass options to the built-in eigensolver. If you just want to use the defaults,
        # you can omit this step, in which case you would not specify the sci_solver argument
        # in the call to diagonalize_fermionic_hamiltonian below.
        if self.n_jobs == 1 or self.n_jobs == None:
            from qiskit_addon_sqd.fermion import solve_sci_batch
            
            sci_solver = partial(solve_sci_batch, spin_sq=self.Spin, max_cycle=self.max_cycle)
        else:
            from qiskit_addon_dice_solver import solve_sci_batch
            sci_solver = partial(solve_sci_batch, spin_sq=self.Spin, max_cycle=self.max_cycle,mpirun_options= ["-quiet", "-n", f"{self.n_jobs}"],temp_dir=self.temp_dir,clean_temp_dir=self.clean_temp_dir)
        # List to capture intermediate results
        result_history = []
        
        
        def callback(results: list[SCIResult]):
            result_history.append(results)
            iteration = len(result_history)
            print(f"Iteration {iteration}")
            for i, result in enumerate(results):
                print(f"\tSubsample {i}")
                print(f"\t\tEnergy: {result.energy + self.nuclear_repulsion_energy}")
                print(f"\t\tSubspace dimension: {np.prod(result.sci_state.amplitudes.shape)}")
        
        
        self.result = diagonalize_fermionic_hamiltonian(
            self.hcore,
            self.eri,
            self.bit_array,
            samples_per_batch=self.samples_per_batch,
            norb=self.NOrb,
            nelec=(self.NElec//2,self.NElec//2),
            num_batches=self.num_batches,
            energy_tol=self.energy_tol,
            occupancies_tol=self.occupancies_tol,
            max_iterations=self.max_iterations,
            sci_solver=sci_solver,
            symmetrize_spin=self.symmetrize_spin,
            carryover_threshold=self.carryover_threshold,
            callback=callback,
            seed=12345
        )        

        self.result_history = result_history
        
    def __call__(self,postprocess=True,JobID=None):
        """
        Run the algorithm 
        
        parameters
        ----------
        postprocess=True
        JobID=None

        return
        ------
        self.result_history, self.result
        self.runtimejob
        
        """
        self.postprocess = postprocess
        self.JobID = JobID
        
        self.Initialize()
        self.Circuit()
        self.Transpile()
        self.RunDevice()
        
        if self.postprocess:
            self.Postprocess()
            return self.result_history, self.result
        else:
            return self.runtimejob



def GrabAmps(name,basisset):
    """
    Find the amplitudes to inject for a name/basis set pair

    parameters
    ----------
    name: str
        Name of molecule

    basisset: str
        Basis set

    returns
    -------
    ampdict: dict
        Dictionary containing pairs of (t1,t2) amplitudes
        Keys: MP2, CCSD, ML, ML_exact, zeroes, random
        
    """
    t1ML_exact = np.load(f'data/{basisset}/amplitudes/amps_{name}_{basisset}_t1_ML_exact.npz')['k']
    t1exact = np.load(f'data/{basisset}/amplitudes/amps_{name}_{basisset}_t1_exact.npz')['k']
    t1rand = np.load(f'data/{basisset}/amplitudes/amps_{name}_{basisset}_t1_rand.npz')['k']
    t1zeroes = np.load(f'data/{basisset}/amplitudes/amps_{name}_{basisset}_t1_zeroes.npz')['k']
    
    t2ML=np.load(f'data/{basisset}/amplitudes/amps_{name}_{basisset}_t2_ML.npz')['k']
    t2ML_exact=np.load(f'data/{basisset}/amplitudes/amps_{name}_{basisset}_t2_ML_exact.npz')['k']
    t2MP2=np.load(f'data/{basisset}/amplitudes/amps_{name}_{basisset}_t2_MP2.npz')['k']
    t2exact=np.load(f'data/{basisset}/amplitudes/amps_{name}_{basisset}_t2_exact.npz')['k']
    t2rand=np.load(f'data/{basisset}/amplitudes/amps_{name}_{basisset}_t2_rand.npz')['k']
    t2zeroes=np.load(f'data/{basisset}/amplitudes/amps_{name}_{basisset}_t2_zeroes.npz')['k']

    ampdict = {"MP2":(t1zeroes,t2MP2),"CCSD":(t1exact,t2exact),"ML":(t1zeroes,t2ML),"ML_exact":(t1ML_exact,t2ML_exact),"zeroes":(t1zeroes,t2zeroes),"random":(t1rand,t2rand)}
    
    return ampdict            