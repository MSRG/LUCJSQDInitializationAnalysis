import os
from functools import partial
import numpy as np

import pyscf
from pyscf import gto, scf, mcscf, cc

import ffsim
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.primitives import StatevectorSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler

from qiskit_addon_sqd.fermion import SCIResult, diagonalize_fermionic_hamiltonian
from qiskit_addon_sqd.counts import bit_array_to_arrays

from ansatzmap import get_zigzag_physical_layout


class DDLUCJ:
    def __init__(self, StructurePath, 
                 BasisSet, 
                 NElec,
                 NOrb,
                 NFroz=0,
                 Symmetry="C1",
                 Spin=0,
                 injected=False,
                 t1=None, 
                 t2=None,
                 n_reps=1,
                 channel=None,
                 instance=None,
                 backend=None,         
                 optimization_level=3,
                 shots=10_000,
                 energy_tol=1e-08,
                 occupancies_tol=1e-05,
                 max_iterations=100,
                 num_batches=1,
                 samples_per_batch=300,
                 symmetrize_spin=True,
                 carryover_threshold=1e-4,
                 max_cycle=200,
                 temp_dir="./",
                 clean_temp_dir=False,
                 n_jobs=None,
                 verbose=False
                ):

        # PySCF options
        self.StructurePath = StructurePath
        self.BasisSet = BasisSet
        self.Spin = Spin
        self.Symmetry = Symmetry
        self.NElec = NElec
        self.NOrb = NOrb
        self.NFroz = NFroz

        # Circuit setup
        self.injected = injected
        self.t1 = t1
        self.t2 = t2
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
        self.temp_dir = temp_dir
        self.clean_temp_dir = clean_temp_dir
        self.n_jobs = n_jobs

        self.verbose = verbose
        self.JobID = None
        self.BitArray = None
        
    def Initialize(self):
        """Initialize PySCF to return integrals, active space, etc."""
        mol = gto.Mole()
        mol.build(
            atom=self.StructurePath,
            basis=self.BasisSet,
            symmetry=self.Symmetry,
            spin=self.Spin
        )
        
        RHF = scf.RHF(mol).run(verbose=0)
        cas = mcscf.CASCI(RHF, self.NOrb, self.NElec, ncore=self.NFroz)
        self.num_elec_a = self.NElec // 2
        self.num_elec_b = self.NElec // 2

        active_space = list(range(cas.ncore, cas.ncore + cas.ncas))
        if self.verbose:
            print(f"Active Space Orbitals: {self.NOrb}, Electrons: {self.NElec}, Frozen: {self.NFroz}")
            print(f"Active indices: {active_space}")

        self.mo = cas.sort_mo(active_space, base=0)
        self.hcore, self.nuclear_repulsion_energy = cas.get_h1cas(self.mo)
        self.eri = pyscf.ao2mo.restore(1, cas.get_h2cas(self.mo), self.NOrb)   

    def Circuit(self):
        if not self.injected and self.t1 is None and self.t2 is None:
            mol = gto.Mole()
            mol.build(atom=self.StructurePath, basis=self.BasisSet, symmetry=self.Symmetry, spin=self.Spin)
            rhf = scf.RHF(mol).run(verbose=0)
            ccsd = cc.CCSD(rhf, frozen=range(self.NFroz)).run(verbose=0)
            self.t1 = ccsd.t1
            self.t2 = ccsd.t2
        
        Nocc, NVirt = self.t1.shape 
        Nact = self.NOrb - self.NFroz
        NVirtSlice = Nact - Nocc
        self.t1 = self.t1[self.NFroz:self.NOrb, :NVirtSlice]
        self.t2 = self.t2[self.NFroz:self.NOrb, self.NFroz:self.NOrb, :NVirtSlice, :NVirtSlice]
        
        alpha_alpha_indices = [(p, p + 1) for p in range(self.NOrb - 1)]
        alpha_beta_indices = [(p, p) for p in range(0, self.NOrb, 4)]
         
        ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
            t2=self.t2,
            t1=self.t1,
            n_reps=self.n_reps,
            interaction_pairs=(alpha_alpha_indices, alpha_beta_indices)
        )
         
        qubits = QuantumRegister(2 * self.NOrb, name="q")
        circuit = QuantumCircuit(qubits)
        
        circuit.append(ffsim.qiskit.PrepareHartreeFockJW(int(self.NOrb), (int(self.NElec // 2), int(self.NElec // 2))), qubits)
        circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), qubits)
        circuit.measure_all()            
        self.circuit = circuit

    def Transpile(self):
        # 1. Check for local statevector mode
        if self.backend == "statevector" or (self.backend is None and self.channel is None):
            pass_manager = generate_preset_pass_manager(
                optimization_level=self.optimization_level
            )
            pass_manager.pre_init = ffsim.qiskit.PRE_INIT
            self.isa_circuit = pass_manager.run(self.circuit)
            self.backend = "statevector" 
            
            if self.verbose:
                print("Transpiled circuit for local StatevectorSampler simulation.")
            return

        # 2. Hardware execution mode
        self.service = QiskitRuntimeService(channel=self.channel, instance=self.instance)

        if self.backend is None:
            self.backend = self.service.least_busy(operational=True, simulator=False)
        elif isinstance(self.backend, str):
            self.backend = self.service.backend(self.backend)
            
        if self.verbose:
            print(f"Using backend {self.backend.name}")
            
        initial_layout, _ = get_zigzag_physical_layout(self.NOrb, backend=self.backend)
         
        pass_manager = generate_preset_pass_manager(
            optimization_level=self.optimization_level, 
            backend=self.backend, 
            initial_layout=initial_layout
        )
        pass_manager.pre_init = ffsim.qiskit.PRE_INIT
        self.isa_circuit = pass_manager.run(self.circuit)

    def RunDevice(self):
        if self.JobID is None and self.BitArray is None:
            self.Circuit()
            self.Transpile()
            
            if self.backend == "statevector":
                if self.verbose:
                    print("Executing simulation using local StatevectorSampler...")
                sampler = StatevectorSampler()
                job = sampler.run([self.isa_circuit], shots=self.shots)
                primitive_result = job.result()
                pub_result = primitive_result[0]
                self.bit_array = pub_result.data.meas
                self.runtimejob = "local_statevector_sim"
            else:
                sampler = Sampler(mode=self.backend)
                job = sampler.run([self.isa_circuit], shots=self.shots)
                primitive_result = job.result()
                pub_result = primitive_result[0]
                self.bit_array = pub_result.data.meas
                if self.verbose:
                    print(f"Qiskit Runtime Job ID: {job.job_id()}")    
                self.runtimejob = job.job_id()
    
        elif self.BitArray is None:
            job = self.service.job(self.JobID)
            primitive_result = job.result()
            pub_result = primitive_result[0]
            self.bit_array = pub_result.data.meas
    
        elif self.JobID is None:
            self.bit_array = self.BitArray
                
    def PostprocessFulqrum(self):
        """
        Use Fulqrum postprocessing
        """
        from fulqum_sqd import diagonalize_fermionic_hamiltonian

        # Case A: Load from pre-saved npz counts file on disk
        if hasattr(self, 'bitarraypath') and self.bitarraypath is not None and os.path.exists(self.bitarraypath):
            counts = np.load(self.bitarraypath, allow_pickle=True)
            self.bitstring_matrix_full = counts['bitstrings']
            self.probs_arr_full = counts['probarr']

        # Case B: Convert BitArray generated by StatevectorSampler / Hardware
        elif hasattr(self, 'bit_array') and self.bit_array is not None:
            bitstring_matrix, probs_arr = bit_array_to_arrays(self.bit_array)
            self.bitstring_matrix_full = bitstring_matrix
            self.probs_arr_full = probs_arr
        else:
            raise ValueError("No bit_array found in memory and no valid bitarraypath provided!")

        occ_a = np.zeros(self.NOrb, dtype=int)
        occ_a[-self.num_elec_a:] = np.ones(self.num_elec_a, dtype=int)
        occ_b = np.zeros(self.NOrb, dtype=int)
        occ_b[-self.num_elec_b:] = np.ones(self.num_elec_b, dtype=int)     
        current_occupancies = [occ_a, occ_b]
        
        probs_arr_full = self.probs_arr_full
        bitstring_matrix_full = self.bitstring_matrix_full
        
        probs_arr_full = np.clip(probs_arr_full, 0, None)
        probs_arr_full = probs_arr_full / probs_arr_full.sum()
            
    
        if isinstance(bitstring_matrix_full, np.ndarray):
            if bitstring_matrix_full.ndim == 2:
                bitstring_matrix_full = [
                    ''.join(row.astype(int).astype(str)) for row in bitstring_matrix_full
                ]
            elif bitstring_matrix_full.ndim == 1:
                bitstring_matrix_full = bitstring_matrix_full.tolist()
                
        total_energy, subspace_dimension = diagonalize_fermionic_hamiltonian(
            self.nuclear_repulsion_energy,
            self.hcore,
            self.eri,
            bitstring_matrix_full, 
            probs_arr_full,
            current_occupancies=current_occupancies,
            samples_per_batch=self.samples_per_batch,
            norb=self.NOrb,
            nelec=(self.num_elec_a, self.num_elec_b),
            num_batches=self.num_batches,
            max_iterations=self.max_iterations,
            carryover_threshold=self.carryover_threshold
        )        

        self.total_energy, self.subspace_dimension = total_energy, subspace_dimension

    def Postprocess(self):
        if self.n_jobs == 1 or self.n_jobs is None:
            from qiskit_addon_sqd.fermion import solve_sci_batch
            sci_solver = partial(solve_sci_batch, spin_sq=self.Spin, max_cycle=self.max_cycle)
        else:
            from qiskit_addon_dice_solver import solve_sci_batch
            sci_solver = partial(
                solve_sci_batch, 
                spin_sq=self.Spin, 
                mpirun_options=["-quiet", "-n", f"{self.n_jobs}"],
                temp_dir=self.temp_dir,
                clean_temp_dir=self.clean_temp_dir
            )

        result_history = []
        
        def callback(results: list[SCIResult]):
            result_history.append(results)
            iteration = len(result_history)
            print(f"Iteration {iteration}")
            for i, result in enumerate(results):
                print(f"\tSubsample {i}")
                print(f"\t\tEnergy: {result.energy + self.nuclear_repulsion_energy}")
                print(f"\t\tSubspace dimension: {np.prod(result.sci_state.amplitudes.shape)}")

        try:
            self.result = diagonalize_fermionic_hamiltonian(
                self.hcore,
                self.eri,
                self.bit_array,
                samples_per_batch=self.samples_per_batch,
                norb=self.NOrb,
                nelec=(self.NElec // 2, self.NElec // 2),
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
        except ValueError:
            Vac = np.zeros(self.NOrb, dtype=int)
            Vac[-self.NElec // 2:] = np.ones(self.NElec // 2, dtype=int)
            
            self.result = diagonalize_fermionic_hamiltonian(
                self.hcore,
                self.eri,
                self.bit_array,
                samples_per_batch=self.samples_per_batch,
                norb=self.NOrb,
                nelec=(self.NElec // 2, self.NElec // 2),
                num_batches=self.num_batches,
                energy_tol=self.energy_tol,
                occupancies_tol=self.occupancies_tol,
                max_iterations=self.max_iterations,
                sci_solver=sci_solver,
                symmetrize_spin=self.symmetrize_spin,
                carryover_threshold=self.carryover_threshold,
                callback=callback,
                initial_occupancies=(Vac, Vac),
                seed=12345
            )              
        self.result_history = result_history
        
    def __call__(self, postprocess=True, JobID=None, BitArray=None, usefulqrum=False, bitarraypath=None):
        self.postprocess = postprocess
        self.JobID = JobID
        self.BitArray = BitArray
        self.usefulqrum = usefulqrum
        self.bitarraypath = bitarraypath
            
        self.Initialize()
        self.RunDevice()
        
        if self.postprocess:
            if self.usefulqrum:
                self.PostprocessFulqrum()
                return self.total_energy, self.subspace_dimension
            else:
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
