#!/usr/bin/env python
from collections import OrderedDict
from fulqrum.convert.integrals import integrals_to_fq_fermionic_op
from fulqrum.core.sqd import ( postselect_by_hamming_right_and_left, subsample, recover_configurations, get_carryover_full_strs)
import fulqrum as fq
import json
import numpy as np
import pyscf
import pyscf.mcscf
import scipy.sparse.linalg as spla
import time
import warnings
warnings.filterwarnings("ignore")

def unique_alpha_beta_combined(bitstrings):
    """A utility function for getting unique alpha and beta halves from full bitstrings.

    Args:
        bitstrings (list[str]): A list of full bitstrings consisting of both alpha and beta parts.

    Returns:
        A dictionary with combined unique ``alpha`` and ``beta`` half strings.
    """
    if not bitstrings:
        return OrderedDict()

    unique_ab = OrderedDict()
    num_spatial_orb = len(bitstrings[0]) // 2

    for bs in bitstrings:
        a = bs[num_spatial_orb:]
        b = bs[:num_spatial_orb]

        unique_ab[a] = 1
        unique_ab[b] = 1

    return unique_ab

def diagonalize_fermionic_hamiltonian(
    nuclear_repulsion_energy,
    hcore,
    eri,
    bit_array,
    prob_array,
    norb,
    nelec,
    samples_per_batch = 5000,
    num_batches=5,
    max_iterations=10,
    carryover_threshold=1e-4,
    current_occupancies=None,
    tol=1e-5,
    seed=12345):        

    fermionic_op = integrals_to_fq_fermionic_op( one_body_integrals=hcore, two_body_integrals=eri)
    fulqrum_operator = fermionic_op.extended_jw_transformation()


    num_elec_a, num_elec_b = nelec


    # options
    seed = 0
    
    carryover_full_strs = []
    S = None  # Subspace
    
    for ni in range(max_iterations):
        iter_start = time.perf_counter()
        print(f"ITERATION {ni}")
        if current_occupancies is None:
            # If we don't have average orbital occupancy information, simply postselect
            # bitstrings with the correct numbers of spin-up and spin-down electrons
            bitstrings, probs = postselect_by_hamming_right_and_left(
                bit_array, prob_array, num_elec_a, num_elec_b
            )
        else:
            # If we do have average orbital occupancy information, use it to refine the
            # full set of noisy configurations
            bitstrings, probs = recover_configurations(
                bit_array,
                prob_array,
                current_occupancies[0].astype(np.float64),  # occs_a
                current_occupancies[1].astype(np.float64),  # occs_b
                num_elec_a,
                num_elec_b,
                seed,
            )

        subsamples = [
            subsample(bitstrings, probs, min(len(bitstrings), len(bitstrings)), seed)
            for _ in range(num_batches)
        ]
    
        for batch in subsamples:
            hs_start = time.perf_counter()
    
            # In this tutorial, we include half strings with the following precedence
            # Hartree-Fock state (always include) > carryover (from previous round) > recovered samples.
            # We add bitstrings in a dictionary as it preserves order and also filters out duplicates.
            half_strs_dict = OrderedDict()
    
            # Add hartree-fock state
            hartree_fock_state = "0" * (norb - num_elec_a) + "1" * num_elec_a
            half_strs_dict[hartree_fock_state] = 1
    
            # Add carry-overs
            # First, we split carryover full strs into unique alpha and beta halves
            unique_ab_co = unique_alpha_beta_combined(carryover_full_strs)
            for bs in unique_ab_co.keys():
                half_strs_dict[bs] = 1
    
            # Add recovered bitstrings
            unique_ab_recovered = unique_alpha_beta_combined(batch)
            for bs in unique_ab_recovered.keys():
                half_strs_dict[bs] = 1
    
            half_strs = list(half_strs_dict.keys())
            print("  num total half strs: ", len(half_strs))
    
            # We will use a subset of accumulated half strings in the Subspace construction
            # defined by the `samples_per_batch` parameter. For this tutorial, the subspace
            # dimension will be samples_per_batch x samples_per_batch
            half_strs = half_strs[:samples_per_batch]
            print("  num selected half strs: ", len(half_strs))
            half_strs.sort()  # fq.Subspace() also sorts bitstrings, so can be skipped.
    
            hs_end = time.perf_counter()
            print(f"  Half strs construction took: {hs_end - hs_start:.4f} seconds")
    
            s_start = time.perf_counter()
            S = fq.Subspace([half_strs, half_strs])
            s_end = time.perf_counter()
            print(f"  Subspace construction took: {s_end - s_start:4f} seconds")
            subspace_dimension = len(half_strs) ** 2
            print(
                f"  Subspace dimension: {len(half_strs)} x {len(half_strs)} = {subspace_dimension:_}"
            )
            
            proj_start = time.perf_counter()
            Hsub = fq.SubspaceHamiltonian(fulqrum_operator, S)
            # Convert the SubspaceHamiltonian into a CSR format sparse matrix,
            # which will also be wrapped in a LinearOperator for faster eigensolving.
            Hsub_csr_linop = Hsub.to_csr_linearoperator_fast(verbose=False)
            proj_end = time.perf_counter()
            print(f"  Operator projection took: {proj_end - proj_start:4f} seconds")
    
            total_bytes = Hsub_csr_linop.memory_size
            total_mega_bytes = total_bytes / (1024 * 1024)
            print(f"  CSR matrix memory: {total_mega_bytes:.6f} MBs")
    
            # Constructing a simple initial guess vector
            v0_start = time.perf_counter()
            diag_vec = Hsub.diagonal_vector()
            min_idx = np.where(diag_vec == diag_vec.min())[0]
            v0 = np.zeros(len(S), dtype=Hsub.dtype)
            v0[min_idx] = 1
            v0_end = time.perf_counter()
            print(
                f"  Initial guess vector v0 construction took: {v0_end - v0_start:.4f} seconds"
            )
            print("  Starting eigensolving ...")
            start = time.perf_counter()
            
            # Check subspace dimension N (dimension of the Hamiltonian operator)
            N = Hsub_csr_linop.shape[0]
            
            if N == 1:
                # Subspace dimension is 1 (e.g. only HF state present)
                # Compute matrix element directly via LinearOperator matvec on [1.0]
                val = Hsub_csr_linop.matvec(np.array([1.0], dtype=Hsub.dtype))[0]
                eigvals = np.array([val])
                eigvecs = np.array([[1.0]])
            elif N <= 1: # k = 1 requested, so if N <= k
                # Convert LinearOperator to dense matrix for exact eigh
                dense_H = Hsub_csr_linop.matmat(np.eye(N, dtype=Hsub.dtype))
                eigvals, eigvecs = spla_dense.eigh(dense_H)
                eigvals, eigvecs = eigvals[:1], eigvecs[:, :1]
            else:
                # Standard ARPACK iterative eigensolver for N > 1
                eigvals, eigvecs = spla.eigsh(
                    Hsub_csr_linop,
                    k=1,
                    which="SA",
                    tol=tol,
                    v0=v0,
                )
                
            end = time.perf_counter()
            total_energy = eigvals + nuclear_repulsion_energy    
            # print("  Starting eigensolving ...")
            # start = time.perf_counter()
            # eigvals, eigvecs = spla.eigsh(
            #     Hsub_csr_linop,  # use `Hsub` for the matrix-free mode. Memory-efficient but slower.
            #     k=1,
            #     which="SA",
            #     tol=tol,
            #     v0=v0,
            # )
            # end = time.perf_counter()
            # total_energy = eigvals + nuclear_repulsion_energy
            print(f"  Eigensolving took: {end - start:.4f} seconds")
            print(f"  Electronic Energy: {eigvals}")
            print(f"  Total Energy: {total_energy}")
            
            eigvecs = eigvecs.ravel()
    
            # Subspace class has a method named `get_orbital_occupancies()` to compute
            # electron orbital occupancies from the eigenvector and subspace bitstrings.
            # The method accepts probabilities, and thus, we must do |amplitude| ^ 2.
            # Returns alpha and beta occupanices as a length-2 tuple in the order
            # ([a0 ... aN], [b0 ... bN]).
            current_occupancies = S.get_orbital_occupancies(
                probs=np.abs(eigvecs) ** 2, norb=norb
            )
    
            # Next, we get important bitstrings to include (carryover to) in the next round.
            # Important bitstrings are the ones with |amplitude| > threshold.
            # It is a must to provide |amplitude|, i.e., np.abs(eigenvector) to the function.
            # The function returns a list of length-2 tuples, where the first element of the tuple
            # is the full bitstrings, and the second element is its weight (|amplitude|).
            # The list is also sorted in the descending order of |amplitude|.
            carryover_full_strs_and_weights = get_carryover_full_strs(
                S, np.abs(eigvecs), carryover_threshold
            )
    
            carryover_full_strs = [item[0] for item in carryover_full_strs_and_weights]
    
            print(f"  num carryover full strs: {len(carryover_full_strs)}")
    
        iter_end = time.perf_counter()
        print(f"Iter {ni} took: {iter_end - iter_start:.4f} seconds\n")

    return total_energy, subspace_dimension
