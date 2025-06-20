#!/usr/bin/env python
# coding: utf-8

# In[6]:
import itertools
import gc
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import psi4
import numpy as np
import numpy as np
import pandas as pd
from psi4 import core

"""
A simple python script to compute RHF-CCSD energy. Equations (Spin orbitals) from reference 1
have been spin-factored. However, explicit building of Wabef intermediates are avoided here.

References:
1. J.F. Stanton, J. Gauss, J.D. Watts, and R.J. Bartlett,
   J. Chem. Phys., volume 94, pp. 4334-4345 (1991).
"""

__authors__ = "Ashutosh Kumar"
__credits__ = [
    "T. D. Crawford", "Daniel G. A. Smith", "Lori A. Burns", "Ashutosh Kumar"
]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2017-05-17"

import time
import numpy as np
import psi4

psi4.set_num_threads(12)


def distribute_triples(ijk,doubles):
    trips=np.zeros((doubles.shape))
    shape=ijk.shape
    for idx in itertools.product(*[range(s) for s in shape]):
        value = ijk[idx]
        denom=doubles[idx[0],idx[1]]+doubles[idx[1],idx[2]]+doubles[idx[0],idx[2]]
        trips[idx[0],idx[1]]+=ijk[idx]/denom*doubles[idx[0],idx[1]]
        trips[idx[1],idx[2]]+=ijk[idx]/denom*doubles[idx[1],idx[2]]
        trips[idx[0],idx[2]]+=ijk[idx]/denom*doubles[idx[0],idx[2]]
        #print(idx, value)
    return trips


def transform_occ(TransformMatrix, Umatrix):
    #determine U* and U
    conj=np.asarray(Umatrix).conj().T
    norm=np.asarray(Umatrix)
    #Determine indices of virtual MOs
    #Make a zeros of new matrix that we fill in
    transformed=np.einsum('ij,jklm,kn->inlm', conj,TransformMatrix,norm, optimize=True)
    return transformed

def transform_occ_back(TransformMatrix, Umatrix):
    #determine U* and U
    conj=np.asarray(Umatrix).conj().T
    norm=np.asarray(Umatrix)
    #Determine indices of virtual MOs
    #Make a zeros of new matrix that we fill in
    transformed=np.einsum('ij,jklm,kn->inlm', norm,TransformMatrix,conj, optimize=True)
    return transformed

def transform_vir(TransformMatrix, Umatrix):
    #determine U* and U
    conj=np.asarray(Umatrix).conj().T
    norm=np.asarray(Umatrix)
    #Determine indices of virtual MOs
    #Make a zeros of new matrix that we fill in
    transformed=np.einsum('ij,kljm,mn->klin', conj,TransformMatrix,norm,optimize=True)
    return transformed

def transform_vir_back(TransformMatrix, Umatrix):
    #determine U* and U
    conj=np.asarray(Umatrix).conj().T
    norm=np.asarray(Umatrix)
    #Determine indices of virtual MOs
    #Make a zeros of new matrix that we fill in
    transformed=np.einsum('ij,kljm,mn->klin', norm,TransformMatrix,conj,optimize=True)
    return transformed

#update for spin-factored 1/7/20
def compute_pairmatrix(t2matrix,integralmatrix):
    test = 2*t2matrix*integralmatrix
    test -= np.swapaxes(t2matrix,2,3)*t2matrix
    c=np.sum(test,axis=(2,3))
    return c

def go_canonical(WF):
    if (WF.Loc_occ != False):
        WF.t2 = transform_occ_back(WF.t2, WF.occ_U)
        #self.t1= np.einsum('ij,jk,kl->il',self.occ_U,self.t1,np.asarray(self.occ_U).conj().T)
    if (WF.Loc_vir != False):
        WF.t2 = transform_vir_back(WF.t2, WF.vir_U)
    if ((WF.Loc_occ != False) or (WF.Loc_vir != False)):
        print ('Going back to canonical orbitals')
        WF.MO=WF.MO_canonical.copy()
        #F=self.F.copy()
        WF.F=WF.F_canonical
        WF.t1 = np.zeros((WF.nocc, WF.nvirt))
        #I'm not sure how to put t1 back to canonical
        WF.compute_t1()
    return WF

# N dimensional dot
# Like a mini DPD library
def ndot(input_string, op1, op2, prefactor=None):
    """
    No checks, if you get weird errors its up to you to debug.

    ndot('abcd,cdef->abef', arr1, arr2)
    """
    inp, output_ind = input_string.split('->')
    input_left, input_right = inp.split(',')

    size_dict = {}
    for s, size in zip(input_left, op1.shape):
        size_dict[s] = size
    for s, size in zip(input_right, op2.shape):
        size_dict[s] = size

    set_left = set(input_left)
    set_right = set(input_right)
    set_out = set(output_ind)

    idx_removed = (set_left | set_right) - set_out
    keep_left = set_left - idx_removed
    keep_right = set_right - idx_removed

    # Tensordot axes
    left_pos, right_pos = (), ()
    for s in idx_removed:
        left_pos += (input_left.find(s), )
        right_pos += (input_right.find(s), )
    tdot_axes = (left_pos, right_pos)

    # Get result ordering
    tdot_result = input_left + input_right
    for s in idx_removed:
        tdot_result = tdot_result.replace(s, '')

    rs = len(idx_removed)
    dim_left, dim_right, dim_removed = 1, 1, 1
    for key, size in size_dict.items():
        if key in keep_left:
            dim_left *= size
        if key in keep_right:
            dim_right *= size
        if key in idx_removed:
            dim_removed *= size

    shape_result = tuple(size_dict[x] for x in tdot_result)
    used_einsum = False

    # Matrix multiply
    # No transpose needed
    if input_left[-rs:] == input_right[:rs]:
        new_view = np.dot(
            op1.reshape(dim_left, dim_removed),
            op2.reshape(dim_removed, dim_right))

    # Transpose both
    elif input_left[:rs] == input_right[-rs:]:
        new_view = np.dot(
            op1.reshape(dim_removed, dim_left).T,
            op2.reshape(dim_right, dim_removed).T)

    # Transpose right
    elif input_left[-rs:] == input_right[-rs:]:
        new_view = np.dot(
            op1.reshape(dim_left, dim_removed),
            op2.reshape(dim_right, dim_removed).T)

    # Tranpose left
    elif input_left[:rs] == input_right[:rs]:
        new_view = np.dot(
            op1.reshape(dim_removed, dim_left).T,
            op2.reshape(dim_removed, dim_right))

    # If we have to transpose vector-matrix, einsum is faster
    elif (len(keep_left) == 0) or (len(keep_right) == 0):
        new_view = np.einsum(input_string, op1, op2)
        used_einsum = True

    else:
        new_view = np.tensordot(op1, op2, axes=tdot_axes)

    # Make sure the resulting shape is correct
    if (new_view.shape != shape_result) and not used_einsum:
        if (len(shape_result) > 0):
            new_view = new_view.reshape(shape_result)
        else:
            new_view = np.squeeze(new_view)

    # In-place mult by prefactor if requested
    if prefactor is not None:
        new_view *= prefactor

    # Do final tranpose if needed
    if used_einsum:
        return new_view
    elif tdot_result == output_ind:
        return new_view
    else:
        return np.einsum(tdot_result + '->' + output_ind, new_view)


class helper_diis(object):
    def __init__(self, t1, t2, max_diis):

        self.oldt1 = t1.copy()
        self.oldt2 = t2.copy()
        self.diis_vals_t1 = [t1.copy()]
        self.diis_vals_t2 = [t2.copy()]
        self.diis_errors = []
        self.diis_size = 0
        self.max_diis = max_diis

    def add_error_vector(self, t1, t2):

        # Add DIIS vectors
        self.diis_vals_t1.append(t1.copy())
        self.diis_vals_t2.append(t2.copy())
        # Add new error vectors
        error_t1 = (self.diis_vals_t1[-1] - self.oldt1).ravel()
        error_t2 = (self.diis_vals_t2[-1] - self.oldt2).ravel()
        self.diis_errors.append(np.concatenate((error_t1, error_t2)))
        self.oldt1 = t1.copy()
        self.oldt2 = t2.copy()

    def extrapolate(self, t1, t2):

        # Limit size of DIIS vector
        if (len(self.diis_vals_t1) > self.max_diis):
            del self.diis_vals_t1[0]
            del self.diis_vals_t2[0]
            del self.diis_errors[0]

        self.diis_size = len(self.diis_vals_t1) - 1

        # Build error matrix B
        B = np.ones((self.diis_size + 1, self.diis_size + 1)) * -1
        B[-1, -1] = 0

        for n1, e1 in enumerate(self.diis_errors):
            B[n1, n1] = np.dot(e1, e1)
            for n2, e2 in enumerate(self.diis_errors):
                if n1 >= n2: continue
                B[n1, n2] = np.dot(e1, e2)
                B[n2, n1] = B[n1, n2]

        B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()

        # Build residual vector
        resid = np.zeros(self.diis_size + 1)
        resid[-1] = -1

        # Solve pulay equations
        ci = np.linalg.solve(B, resid)

        # Calculate new amplitudes
        t1 = np.zeros_like(self.oldt1)
        t2 = np.zeros_like(self.oldt2)
        for num in range(self.diis_size):
            t1 += ci[num] * self.diis_vals_t1[num + 1]
            t2 += ci[num] * self.diis_vals_t2[num + 1]

        # Save extrapolated amplitudes to old_t amplitudes
        self.oldt1 = t1.copy()
        self.oldt2 = t2.copy()

        return t1, t2

class HelperCCEnergy(object):
    def __init__(self, mol, memory=64, ML=False, Triples=False, freeze_core=True, Loc_occ=False, Loc_vir=False, iterate=True, triples_only=False):


        self.Loc_occ=Loc_occ
        self.Loc_vir=Loc_vir
        self.Triples=Triples
        self.iterate=iterate
        self.triples_only=triples_only
        print('Computing RHF reference.')
        psi4.core.set_active_molecule(mol)
        psi4.set_module_options('SCF', {'SCF_TYPE': 'PK'})
        psi4.set_module_options('SCF', {'E_CONVERGENCE': 10e-10})
        psi4.set_module_options('SCF', {'D_CONVERGENCE': 10e-10})
        psi4.set_options({'LOCAL_MAXITER':10000})
        if freeze_core== True:
            psi4.set_options({'FREEZE_CORE': 'TRUE'})
        else:
            psi4.set_options({'FREEZE_CORE': 'FALSE'})
        self.rhf_e, self.wfn = psi4.energy('SCF', return_wfn=True)

        self.nfzc=0

        if freeze_core == True:

            Zlist = np.array([mol.Z(x) for x in range(mol.natom())])
            self.nfzc = np.sum(Zlist > 2)
            self.nfzc += np.sum(Zlist > 10) * 4
            if np.any(Zlist > 18):
                raise Exception("Frozen core for Z > 18 not yet implemented")

            print("Cutting %d core orbitals." % self.nfzc)

            # Copy C
            #oldC = np.asarray(self.wfn.Ca())

            # Build new C matrix and view, set with numpy slicing
            #self.C = psi.Matrix(self.nmo, self.nmo - self.nfzc)
            #self.npC = np.asarray(self.C)
            #self.npC = oldC[:, self.nfzc:]

            self.C = self.wfn.Ca()
            C_occ = self.wfn.Ca_subset("AO", "ACTIVE_OCC")


            C_vir = self.wfn.Ca_subset("AO", "VIR")

            #self.C=core.Matrix.from_array(self.npC)
            # Update epsilon array
            #self.ndocc -= self.nfzc
        else:

            self.C = self.wfn.Ca()
            C_occ = self.wfn.Ca_subset("AO", "OCC")
            C_vir = self.wfn.Ca_subset("AO", "VIR")
        print("\nInitalizing CCSD object...\n")

        # Integral generation from Psi4's MintsHelper
        time_init = time.time()

        self.ccsd_corr_e = 0.0
        self.ccsd_e = 0.0

        self.ndocc = self.wfn.doccpi()[0]
        self.nmo = self.wfn.nmo()

        self.memory = memory


        self.npC = np.asarray(self.C)
        basis_ = self.wfn.basisset()
        new_C=np.asarray(self.C)

        #C_occ = self.wfn.Ca_subset("AO", "OCC") # canonical C_occ coefficients



        if ((self.Loc_occ != False) or (self.Loc_vir != False)):
            C_canonical=self.wfn.Ca()
            npC_canonical = np.asarray(C_canonical).copy()


        if self.Loc_occ == 'PM':
            #print ("using PM local orbitals")
            Localocc = psi4.core.Localizer.build("PIPEK_MEZEY", basis_, C_occ) # Pipek-Mezey Localization
            #Local = psi4.core.Localizer.build("PIPEK_MEZEY, basis_, C_occ) # Boys Localization
            Localocc.localize()
            new_C_occ = Localocc.L # local C_occ coefficientsi

            print("PM Occupied Localization Convergedness: {}".format(Localocc.converged))

            print (np.asarray(Localocc.L).shape)
            #This indexing is redundant
            #The C matrix is (rows x columns) where rows=AO coeff and columns= MOs
            new_C[:np.asarray(new_C_occ).shape[0],self.nfzc:self.ndocc]=np.asarray(Localocc.L)
            self.occ_U=Localocc.U

        if self.Loc_occ == 'BOYS':
            #print ("using PM local orbitals")
            Localocc = psi4.core.Localizer.build("BOYS", basis_, C_occ) # Pipek-Mezey Localization
            #Local = psi4.core.Localizer.build("PIPEK_MEZEY, basis_, C_occ) # Boys Localization
            Localocc.localize()
            new_C_occ = Localocc.L # local C_occ coefficients
            print("Boys Occupied Localization Convergedness: {}".format(Localocc.converged))
            #This indexing is redundant
            #The C matrix is (rows x columns) where rows=AO coeff and columns= MOs
            new_C[:np.asarray(new_C_occ).shape[0],self.nfzc:self.ndocc]=np.asarray(Localocc.L)
            self.occ_U=Localocc.U


        if self.Loc_vir == 'PM':
            Localvir = psi4.core.Localizer.build("PIPEK_MEZEY", basis_, C_vir)
            Localvir.localize()
            print("PM Virtual Localization Convergedness: {}".format(Localvir.converged))
            new_C_vir = Localvir.L
            new_C[:,-np.asarray(new_C_vir).shape[1]:]=np.asarray(Localvir.L)
            self.vir_U=Localvir.U

        if Loc_vir == 'BOYS':
            Localvir = psi4.core.Localizer.build("BOYS", basis_, C_vir)
            Localvir.localize()
            print("Boys Virtual Localization Convergedness: {}".format(Localvir.converged))

            new_C_vir = Localvir.L
            new_C[:,-np.asarray(new_C_vir).shape[1]:]=np.asarray(Localvir.L)
            self.vir_U=Localvir.U



        self.C=core.Matrix.from_array(new_C)
        self.npC = np.asarray(self.C)


        self.mints = psi4.core.MintsHelper(self.wfn.basisset())
        H = np.asarray(self.mints.ao_kinetic()) + np.asarray(
            self.mints.ao_potential())
        self.nmo = H.shape[0]

        # Update H, transform to MO basis
        H = np.einsum('uj,vi,uv', self.npC, self.npC, H, optimize=True)

        print('Starting AO ->  MO transformation...')

        ERI_Size = self.nmo * 128e-9
        memory_footprint = ERI_Size * 5
        #if memory_footprint > self.memory:
        #    psi.clean()
         #   raise Exception(
         #       "Estimated memory utilization (%4.2f GB) exceeds numpy_memory \
         #                   limit of %4.2f GB." % (memory_footprint,
         #                                          self.memory))

        # Integral generation from Psi4's MintsHelper
        self.MO = np.asarray(self.mints.mo_eri(self.C, self.C, self.C, self.C))
        # Physicist notation
        self.MO = self.MO.swapaxes(1, 2)
        print("Size of the ERI tensor is %4.2f GB, %d basis functions." %
              (ERI_Size, self.nmo))

        # Update nocc and nvirt

        self.nocc = self.ndocc - self.nfzc
        self.nvirt = self.nmo - self.nocc - self.nfzc

        # Make slices

        self.slice_o = slice(0, self.nocc )
        self.slice_v = slice(self.nocc, self.nmo-self.nfzc)
        self.slice_a = slice(0, self.nmo-self.nfzc)
        self.slice_dict = {
            'o': self.slice_o,
            'v': self.slice_v,
            'a': self.slice_a,

        }


        # Compute Fock matrix
        self.F = H + 2.0 * np.einsum('pmqm->pq',
                                     self.MO[:, 0:(self.nocc+self.nfzc), :, 0:(self.nocc+self.nfzc)],optimize=True)
        self.F -= np.einsum('pmmq->pq',
                            self.MO[:, 0:(self.nocc+self.nfzc), 0:(self.nocc+self.nfzc), :],optimize=True)
        self.H1 = H
        print (self.F.shape)

        self.F=self.F[self.nfzc:,self.nfzc:]
        self.MO=self.MO[self.nfzc:,self.nfzc:,self.nfzc:,self.nfzc:]

        print (self.F.shape)
        #print('this should be orbital energies')
        #self.intstuff = self.F - H
        ### Occupied and Virtual orbital energies
        self.J1 = np.einsum('pmqm->pq', self.MO[:, self.slice_o, :, self.slice_o],optimize=True)
        self.K1 = np.einsum('pmmq->pq', self.MO[:, self.slice_o, self.slice_o, :],optimize=True)
        #self.eint = ( 2 * self.J1 ) - self.K1
        self.doublecheck = self.MO[self.slice_o, self.slice_o, self.slice_v, self.slice_v]
        self.Jia = self.J1[0:self.nocc, self.nocc:self.nmo]
        #self.Jvir = np.diag(self.J1)[self.slice_v]
        #self.Kocc = np.diag(self.J1)[self.slice_o]
        self.Kia = self.K1[0:self.nocc, self.nocc:self.nmo]
        Focc = np.diag(self.F)[self.slice_o]
        Fvir = np.diag(self.F)[self.slice_v]
        Jocc = np.diag(self.J1)[self.slice_o]
        Jvir = np.diag(self.J1)[self.slice_v]
        Kocc = np.diag(self.K1)[self.slice_o]
        Kvir = np.diag(self.K1)[self.slice_v]
        Hocc = np.diag(self.H1)[self.slice_o]
        Hvir = np.diag(self.H1)[self.slice_v]

        self.orbocc=Focc
        self.orbvirt=Fvir

        self.Dia = Focc.reshape(-1, 1) - Fvir
        self.Dijab = Focc.reshape(-1, 1, 1, 1) + Focc.reshape(
            -1, 1, 1) - Fvir.reshape(-1, 1) - Fvir


        #H = np.asarray(self.mints.ao_kinetic()) + np.asarray(
   # self.mints.ao_potential())
       # self.nmo = H.shape[0]

        # Update H, transform to MO basis
       # H = np.einsum('uj,vi,uv', self.npC, self.npC, H)

        if ((self.Loc_occ != False) or (self.Loc_vir != False)):
            #The C object for the integrals has changed.  Taking the saved canonical npC and producing C object
            C_canonical=core.Matrix.from_array(npC_canonical)
            H_canonical = np.asarray(self.mints.ao_kinetic()) + np.asarray(self.mints.ao_potential())
            H_canonical = np.einsum('uj,vi,uv', npC_canonical, npC_canonical, H_canonical,optimize=True)
            #H_canonical = np.repeat(H_canonical, 2, axis=0)
            #H_canonical = np.repeat(H_canonical, 2, axis=1)
            #spin_ind = np.arange(H_canonical.shape[0], dtype=np.int) % 2
            #H_canonical *= (spin_ind.reshape(-1, 1) == spin_ind)
            #print ('Computing canonical MO integrals')
            self.MO_canonical = np.asarray(self.mints.mo_eri(C_canonical, C_canonical,C_canonical, C_canonical))
            self.MO_canonical = self.MO_canonical.swapaxes(1, 2)

            self.F_canonical = H_canonical + 2.0 * np.einsum('pmqm->pq',
                                     self.MO_canonical[:, :self.ndocc, :, :self.ndocc],optimize=True)
            self.F_canonical -= np.einsum('pmmq->pq',
                            self.MO_canonical[:, :self.ndocc, :self.ndocc, :],optimize=True)

            self.F_canonical=self.F_canonical[self.nfzc:,self.nfzc:]
            self.MO_canonical=self.MO_canonical[self.nfzc:,self.nfzc:,self.nfzc:,self.nfzc:]
            #F_canonical = H_canonical + np.einsum('pmqm->pq', MO_canonical[:, self.slice_o, :, self.slice_o])
            Focc_canonical = np.diag(self.F_canonical)[self.slice_o]
            Fvir_canonical = np.diag(self.F_canonical)[self.slice_v]
            Dia_canonical = Focc_canonical.reshape(-1, 1) - Fvir_canonical
            Dijab_canonical = Focc_canonical.reshape(-1, 1, 1, 1) + Focc_canonical.reshape(-1, 1, 1) - Fvir_canonical.reshape(-1, 1) - Fvir_canonical
            self.t2=self.MO_canonical[self.slice_o, self.slice_o, self.slice_v, self.slice_v] / Dijab_canonical
            if (self.Loc_occ != False):
                #print ('Transforming t2 matrix Occupado')
                self.t2 = transform_occ(self.t2, self.occ_U)
            if (self.Loc_vir != False):
                #print ('Transforming t2 matrix for virtual orbitals')
                self.t2 = transform_vir(self.t2, self.vir_U)
        else:
            self.t2 = self.MO[self.slice_o, self.slice_o, self.slice_v, self.slice_v] / self.Dijab
        ### Construct initial guess
        print('Building initial guess...')
        # t^a_i
        self.t1 = np.zeros((self.nocc, self.nvirt))
        # t^{ab}_{ij}
        #self.t2 = self.MO[self.slice_o, self.slice_o, self.slice_v,
         #                 self.slice_v] / self.Dijab


       # print ("here are the t2 amplitudes, saved as t2start")
        self.t2start=self.t2.copy()

        self.t2mag=np.log10(np.absolute(self.t2))
        infcheck=(self.t2mag == -np.inf)
        self.t2mag[infcheck]=20
        self.t2sign=(self.t2 > 0)
        self.orbdiff=(self.Dijab)
        #if ML==True:
        #    self.t2=MLt2
        #    print ('using ML amps')
        #print ("this should be integrals, saving as int1, int2, int3, int4")
        #print  (self.MO[self.slice_o, self.slice_o, self.slice_v,self.slice_v])
        #self.int1,self.int2=self.MO[self.slice_o, self.slice_o, self.slice_v,
         #                 self.slice_v]
        #print ("this is Dijab, which should be difference in orbital energies, saving as orbdiff")
       # print (self.Dijab)
        print('\n..initialized CCSD in %.3f seconds.\n' %
              (time.time() - time_init))
         #BUILD ALL THE Orbital energies for t2 amps.., four arrays Eocc1, Eocc2, Evir1, Evir2
        empty=np.zeros((self.nvirt,))
        occupado=np.zeros((self.nocc,))

            #This is the energy of both combined orbitals
        combined = (self.orbocc.reshape(-1,1,1,1)+self.orbocc.reshape(-1, 1, 1) - empty.reshape(-1, 1) - empty)
            #This is the first orbital
        self.Eocc1=( self.orbocc.reshape(-1,1,1,1) + occupado.reshape(-1, 1, 1) - empty.reshape(-1, 1) - empty )
        self.Eocc2 = (combined-self.Eocc1)
        self.doublyocc = ( self.Eocc1 == self.Eocc2).astype(int)

           #Let's work on virtual orbitals
        combinedvirtual=(occupado.reshape(-1,1,1,1) + occupado.reshape(-1, 1 , 1) + self.orbvirt.reshape(-1 , 1) + self.orbvirt)
        firstvirtual= (occupado.reshape(-1,1,1,1) + occupado.reshape(-1, 1 , 1) - empty.reshape(-1 , 1) + self.orbvirt)
        self.Evir1= (occupado.reshape(-1,1,1,1) + occupado.reshape(-1, 1 , 1) - empty.reshape(-1 , 1) + self.orbvirt)
        self.Evir2= (combinedvirtual-firstvirtual)
        #self.doublyvir = ( self.Evir1 == self.Evir2).astype(int)
        #self.doubledouble = (self.doublyvir == self.doublyocc).astype(int)

        test=np.zeros(self.t2.shape)
       #This is for 2 electron matrices
        self.Jia1=(test + self.Jia[:,np.newaxis,:,np.newaxis])
        self.Jia2=(test + self.Jia[np.newaxis,:,np.newaxis,:])
        self.Kia1=(test + self.Kia[:,np.newaxis,:,np.newaxis])
        self.Kia2=(test + self.Kia[:,np.newaxis,:,np.newaxis])
        self.Jia1mag=np.log10(np.absolute(self.Jia1))
        self.Jia2mag=np.log10(np.absolute(self.Jia2))
        self.Kia1mag=np.log10(np.absolute(self.Kia1))
        self.Kia2mag=np.log10(np.absolute(self.Kia2))
        self.diag=test
        for i in range (0,self.nocc):
            for j in range (0,self.nocc):
                np.fill_diagonal(self.diag[i,j,:,:],1)
       #I think those features are not great, try <ii,aa>
        temp=np.zeros((self.nocc,self.nvirt))
        for i in range (0,self.nocc):
            for j in range (0,self.nvirt):
                temp[i,j]=self.doublecheck[i,i,j,j]
        test1=np.zeros((self.t2.shape))
        self.screen1=test1+temp[:,np.newaxis,:,np.newaxis]
        self.screen2=test1+temp[np.newaxis,:,np.newaxis,:]


        val=self.nmo-self.nfzc
        temp=np.zeros((val,val))
        for i in range (0,val):
            for j in range (0,val):
                temp[i,j]=self.MO[i,i,j,j]

        temp =temp[self.slice_v,self.slice_v]
        self.screenvirt=test1+temp[np.newaxis,np.newaxis,:,:]
       #print (test + matrix[:,np.newaxis,np.newaxis,:])
#print (test + matrix [np.newaxis,:,:,np.newaxis])



   #Creating the same equations for J, K, H
        #occupied orbs
        combinedJocc = (Jocc.reshape(-1,1,1,1) + Jocc.reshape(-1, 1, 1) - empty.reshape(-1, 1) - empty)
        combinedKocc = ( Kocc.reshape(-1,1,1,1)+ Kocc.reshape(-1, 1, 1) - empty.reshape(-1, 1) - empty)
        combinedHocc = ( Hocc.reshape(-1,1,1,1)+ Hocc.reshape(-1, 1, 1) - empty.reshape(-1, 1) - empty)
        self.Jocc1=(Jocc.reshape(-1,1,1,1) + occupado.reshape(-1, 1, 1) - empty.reshape(-1, 1) - empty )
        self.Kocc1=(Kocc.reshape(-1,1,1,1) + occupado.reshape(-1, 1, 1) - empty.reshape(-1, 1) - empty )
        self.Hocc1=(Hocc.reshape(-1,1,1,1) + occupado.reshape(-1, 1, 1) - empty.reshape(-1, 1) - empty )
        self.Jocc2 = combinedJocc - self.Jocc1
        self.Kocc2 = combinedKocc - self.Kocc1
        self.Hocc2 = combinedHocc - self.Hocc1

        #virtual orbs
        combinedJvir=(occupado.reshape(-1,1,1,1) + occupado.reshape(-1, 1 , 1) + Jvir.reshape(-1 , 1) + Jvir )
        combinedKvir=(occupado.reshape(-1,1,1,1) + occupado.reshape(-1, 1 , 1) + Kvir.reshape(-1 , 1) + Kvir )
        combinedHvir=(occupado.reshape(-1,1,1,1) + occupado.reshape(-1, 1 , 1) + Hvir.reshape(-1 , 1) + Hvir )
        self.Jvir1 = (occupado.reshape(-1,1,1,1) + occupado.reshape(-1, 1 , 1) - empty.reshape(-1 , 1) + Jvir )
        self.Kvir1 = (occupado.reshape(-1,1,1,1) + occupado.reshape(-1, 1 , 1) - empty.reshape(-1 , 1) + Kvir )
        self.Hvir1 = (occupado.reshape(-1,1,1,1) + occupado.reshape(-1, 1 , 1) - empty.reshape(-1 , 1) + Hvir )
        self.Jvir2= (combinedJvir - self.Jvir1)
        self.Kvir2= (combinedKvir - self.Kvir1)
        self.Hvir2= (combinedHvir - self.Hvir1)
        self.pairenergy=(np.zeros(self.doublecheck.shape)+compute_pairmatrix(self.t2start,self.doublecheck)[:,:,np.newaxis,np.newaxis])
        self.triplecheck=2*self.t2start*self.get_MO('oovv')
        self.triplecheck -=  np.swapaxes(self.get_MO('oovv'),2,3)*self.t2start
        #t1 equations
        self.t1occ=self.orbocc.reshape(-1, 1) - empty.reshape(-1)
        #have to truncate the dimensions of this one
        int_t2vir= (empty.reshape(-1,1) + self.orbvirt)
        self.t1vir= int_t2vir[0:self.nocc, :]
        #doublyocc, doublyvir, doubledouble
#This double checks that they are equivalent
        tmp_tau = self.build_tau()
        self.pairs=2*tmp_tau*self.get_MO('oovv')
        self.pairs-= np.swapaxes(self.get_MO('oovv'),2,3)*tmp_tau
        self.pairs = np.sum(self.pairs,axis=(2,3))
    # occ orbitals  : i, j, k, l, m, n
    # virt orbitals : a, b, c, d, e, f
    # all oribitals : p, q, r, s, t, u, v







    def get_MO(self, string):
        if len(string) != 4:
            psi4.core.clean()
            raise Exception('get_MO: string %s must have 4 elements.' % string)
        return self.MO[self.slice_dict[string[0]], self.slice_dict[string[1]],
                       self.slice_dict[string[2]], self.slice_dict[string[3]]]

    def get_F(self, string):
        if len(string) != 2:
            psi4.core.clean()
            raise Exception('get_F: string %s must have 4 elements.' % string)
        return self.F[self.slice_dict[string[0]], self.slice_dict[string[1]]]

    #Equations from Reference 1 (Stanton's paper)

    #Bulid Eqn 9:
    def build_tilde_tau(self):
        ttau = self.t2.copy()
        tmp = 0.5 * np.einsum('ia,jb->ijab', self.t1, self.t1,optimize=True)
        ttau += tmp
        return ttau

    #Build Eqn 10:
    def build_tau(self):
        ttau = self.t2.copy()
        tmp = np.einsum('ia,jb->ijab', self.t1, self.t1,optimize=True)
        ttau += tmp
        return ttau

    #Build Eqn 3:
    def build_Fae(self):
        Fae = self.get_F('vv').copy()
        Fae -= ndot('me,ma->ae', self.get_F('ov'), self.t1, prefactor=0.5)
        Fae += ndot('mf,mafe->ae', self.t1, self.get_MO('ovvv'), prefactor=2.0)
        Fae += ndot(
            'mf,maef->ae', self.t1, self.get_MO('ovvv'), prefactor=-1.0)
        Fae -= ndot(
            'mnaf,mnef->ae',
            self.build_tilde_tau(),
            self.get_MO('oovv'),
            prefactor=2.0)
        Fae -= ndot(
            'mnaf,mnfe->ae',
            self.build_tilde_tau(),
            self.get_MO('oovv'),
            prefactor=-1.0)
        return Fae

    #Build Eqn 4:
    def build_Fmi(self):
        Fmi = self.get_F('oo').copy()
        Fmi += ndot('ie,me->mi', self.t1, self.get_F('ov'), prefactor=0.5)
        Fmi += ndot('ne,mnie->mi', self.t1, self.get_MO('ooov'), prefactor=2.0)
        Fmi += ndot(
            'ne,mnei->mi', self.t1, self.get_MO('oovo'), prefactor=-1.0)
        Fmi += ndot(
            'inef,mnef->mi',
            self.build_tilde_tau(),
            self.get_MO('oovv'),
            prefactor=2.0)
        Fmi += ndot(
            'inef,mnfe->mi',
            self.build_tilde_tau(),
            self.get_MO('oovv'),
            prefactor=-1.0)
        return Fmi

    #Build Eqn 5:
    def build_Fme(self):
        Fme = self.get_F('ov').copy()
        Fme += ndot('nf,mnef->me', self.t1, self.get_MO('oovv'), prefactor=2.0)
        Fme += ndot(
            'nf,mnfe->me', self.t1, self.get_MO('oovv'), prefactor=-1.0)
        return Fme

    #Build Eqn 6:
    def build_Wmnij(self):
        Wmnij = self.get_MO('oooo').copy()
        Wmnij += ndot('je,mnie->mnij', self.t1, self.get_MO('ooov'))
        Wmnij += ndot('ie,mnej->mnij', self.t1, self.get_MO('oovo'))
        # prefactor of 1 instead of 0.5 below to fold the last term of
        # 0.5 * tau_ijef Wabef in Wmnij contraction: 0.5 * tau_mnab Wmnij_mnij
        Wmnij += ndot(
            'ijef,mnef->mnij',
            self.build_tau(),
            self.get_MO('oovv'),
            prefactor=1.0)
        return Wmnij

    #Build Eqn 8:
    def build_Wmbej(self):
        Wmbej = self.get_MO('ovvo').copy()
        Wmbej += ndot('jf,mbef->mbej', self.t1, self.get_MO('ovvv'))
        Wmbej -= ndot('nb,mnej->mbej', self.t1, self.get_MO('oovo'))
        tmp = (0.5 * self.t2)
        tmp += np.einsum('jf,nb->jnfb', self.t1, self.t1,optimize=True)
        Wmbej -= ndot('jnfb,mnef->mbej', tmp, self.get_MO('oovv'))
        Wmbej += ndot(
            'njfb,mnef->mbej', self.t2, self.get_MO('oovv'), prefactor=1.0)
        Wmbej += ndot(
            'njfb,mnfe->mbej', self.t2, self.get_MO('oovv'), prefactor=-0.5)
        return Wmbej

    # This intermediate appaears in the spin factorization of Wmbej terms.
    def build_Wmbje(self):
        Wmbje = -1.0 * (self.get_MO('ovov').copy())
        Wmbje -= ndot('jf,mbfe->mbje', self.t1, self.get_MO('ovvv'))
        Wmbje += ndot('nb,mnje->mbje', self.t1, self.get_MO('ooov'))
        tmp = (0.5 * self.t2)
        tmp += np.einsum('jf,nb->jnfb', self.t1, self.t1,optimize=True)
        Wmbje += ndot('jnfb,mnfe->mbje', tmp, self.get_MO('oovv'))
        return Wmbje

    # This intermediate is required to build second term of 0.5 * tau_ijef * Wabef,
    # as explicit construction of Wabef is avoided here.
    def build_Zmbij(self):
        Zmbij = 0
        Zmbij += ndot('mbef,ijef->mbij', self.get_MO('ovvv'), self.build_tau())
        return Zmbij

    def update(self):

        ### Build OEI intermediates
        Fae = self.build_Fae()
        Fmi = self.build_Fmi()
        Fme = self.build_Fme()

        #### Build residual of T1 equations by spin adaption of  Eqn 1:
        r_T1 = self.get_F('ov').copy()
        r_T1 += ndot('ie,ae->ia', self.t1, Fae)
        r_T1 -= ndot('ma,mi->ia', self.t1, Fmi)
        r_T1 += ndot('imae,me->ia', self.t2, Fme, prefactor=2.0)
        r_T1 += ndot('imea,me->ia', self.t2, Fme, prefactor=-1.0)
        r_T1 += ndot(
            'nf,nafi->ia', self.t1, self.get_MO('ovvo'), prefactor=2.0)
        r_T1 += ndot(
            'nf,naif->ia', self.t1, self.get_MO('ovov'), prefactor=-1.0)
        r_T1 += ndot(
            'mief,maef->ia', self.t2, self.get_MO('ovvv'), prefactor=2.0)
        r_T1 += ndot(
            'mife,maef->ia', self.t2, self.get_MO('ovvv'), prefactor=-1.0)
        r_T1 -= ndot(
            'mnae,nmei->ia', self.t2, self.get_MO('oovo'), prefactor=2.0)
        r_T1 -= ndot(
            'mnae,nmie->ia', self.t2, self.get_MO('ooov'), prefactor=-1.0)

        ### Build residual of T2 equations by spin adaptation of Eqn 2:
        # <ij||ab> ->  <ij|ab>
        #   spin   ->  spin-adapted (<alpha beta| alpha beta>)
        r_T2 = self.get_MO('oovv').copy()

        # Conventions used:
        #   P(ab) f(a,b) = f(a,b) - f(b,a)
        #   P(ij) f(i,j) = f(i,j) - f(j,i)
        #   P^(ab)_(ij) f(a,b,i,j) = f(a,b,i,j) + f(b,a,j,i)

        # P(ab) {t_ijae Fae_be}  ->  P^(ab)_(ij) {t_ijae Fae_be}
        tmp = ndot('ijae,be->ijab', self.t2, Fae)
        r_T2 += tmp
        r_T2 += tmp.swapaxes(0, 1).swapaxes(2, 3)

        # P(ab) {-0.5 * t_ijae t_mb Fme_me} -> P^(ab)_(ij) {-0.5 * t_ijae t_mb Fme_me}
        tmp = ndot('mb,me->be', self.t1, Fme)
        first = ndot('ijae,be->ijab', self.t2, tmp, prefactor=0.5)
        r_T2 -= first
        r_T2 -= first.swapaxes(0, 1).swapaxes(2, 3)

        # P(ij) {-t_imab Fmi_mj}  ->  P^(ab)_(ij) {-t_imab Fmi_mj}
        tmp = ndot('imab,mj->ijab', self.t2, Fmi, prefactor=1.0)
        r_T2 -= tmp
        r_T2 -= tmp.swapaxes(0, 1).swapaxes(2, 3)

        # P(ij) {-0.5 * t_imab t_je Fme_me}  -> P^(ab)_(ij) {-0.5 * t_imab t_je Fme_me}
        tmp = ndot('je,me->jm', self.t1, Fme)
        first = ndot('imab,jm->ijab', self.t2, tmp, prefactor=0.5)
        r_T2 -= first
        r_T2 -= first.swapaxes(0, 1).swapaxes(2, 3)

        # Build TEI Intermediates
        tmp_tau = self.build_tau()
        Wmnij = self.build_Wmnij()
        Wmbej = self.build_Wmbej()
        Wmbje = self.build_Wmbje()
        Zmbij = self.build_Zmbij()

        # 0.5 * tau_mnab Wmnij_mnij  -> tau_mnab Wmnij_mnij
        # This also includes the last term in 0.5 * tau_ijef Wabef
        # as Wmnij is modified to include this contribution.
        r_T2 += ndot('mnab,mnij->ijab', tmp_tau, Wmnij, prefactor=1.0)

        # Wabef used in eqn 2 of reference 1 is very expensive to build and store, so we have
        # broken down the term , 0.5 * tau_ijef * Wabef (eqn. 7) into different components
        # The last term in the contraction 0.5 * tau_ijef * Wabef is already accounted
        # for in the contraction just above.

        # First term: 0.5 * tau_ijef <ab||ef> -> tau_ijef <ab|ef>
        r_T2 += ndot(
            'ijef,abef->ijab', tmp_tau, self.get_MO('vvvv'), prefactor=1.0)

        # Second term: 0.5 * tau_ijef (-P(ab) t_mb <am||ef>)  -> -P^(ab)_(ij) {t_ma * Zmbij_mbij}
        # where Zmbij_mbij = <mb|ef> * tau_ijef
        tmp = ndot('ma,mbij->ijab', self.t1, Zmbij)
        r_T2 -= tmp
        r_T2 -= tmp.swapaxes(0, 1).swapaxes(2, 3)

        # P(ij)P(ab) t_imae Wmbej -> Broken down into three terms below
        # First term: P^(ab)_(ij) {(t_imae - t_imea)* Wmbej_mbej}
        tmp = ndot('imae,mbej->ijab', self.t2, Wmbej, prefactor=1.0)
        tmp += ndot('imea,mbej->ijab', self.t2, Wmbej, prefactor=-1.0)
        r_T2 += tmp
        r_T2 += tmp.swapaxes(0, 1).swapaxes(2, 3)

        # Second term: P^(ab)_(ij) t_imae * (Wmbej_mbej + Wmbje_mbje)
        tmp = ndot('imae,mbej->ijab', self.t2, Wmbej, prefactor=1.0)
        tmp += ndot('imae,mbje->ijab', self.t2, Wmbje, prefactor=1.0)
        r_T2 += tmp
        r_T2 += tmp.swapaxes(0, 1).swapaxes(2, 3)

        # Third term: P^(ab)_(ij) t_mjae * Wmbje_mbie
        tmp = ndot('mjae,mbie->ijab', self.t2, Wmbje, prefactor=1.0)
        r_T2 += tmp
        r_T2 += tmp.swapaxes(0, 1).swapaxes(2, 3)

        # -P(ij)P(ab) {-t_ie * t_ma * <mb||ej>} -> P^(ab)_(ij) {-t_ie * t_ma * <mb|ej>
        #                                                      + t_ie * t_mb * <ma|je>}
        tmp = ndot('ie,ma->imea', self.t1, self.t1)
        tmp1 = ndot('imea,mbej->ijab', tmp, self.get_MO('ovvo'))
        r_T2 -= tmp1
        r_T2 -= tmp1.swapaxes(0, 1).swapaxes(2, 3)
        tmp = ndot('ie,mb->imeb', self.t1, self.t1)
        tmp1 = ndot('imeb,maje->ijab', tmp, self.get_MO('ovov'))
        r_T2 -= tmp1
        r_T2 -= tmp1.swapaxes(0, 1).swapaxes(2, 3)

        # P(ij) {t_ie <ab||ej>} -> P^(ab)_(ij) {t_ie <ab|ej>}
        tmp = ndot(
            'ie,abej->ijab', self.t1, self.get_MO('vvvo'), prefactor=1.0)
        r_T2 += tmp
        r_T2 += tmp.swapaxes(0, 1).swapaxes(2, 3)

        # P(ab) {-t_ma <mb||ij>} -> P^(ab)_(ij) {-t_ma <mb|ij>}
        tmp = ndot(
            'ma,mbij->ijab', self.t1, self.get_MO('ovoo'), prefactor=1.0)
        r_T2 -= tmp
        r_T2 -= tmp.swapaxes(0, 1).swapaxes(2, 3)

        ### Update T1 and T2 amplitudes
        self.t1 += r_T1 / self.Dia
      #  print ("t1 residual & max")
    #    print ( np.absolute(r_T1 / self.Dia).mean() )
     #   print ( np.absolute(r_T1 / self.Dia).max() )
       # print ("t2 residual")
        self.t2 += r_T2 / self.Dijab
    #    print ( np.absolute(r_T2 / self.Dijab).mean() )
   #     print ( np.absolute(r_T2 / self.Dijab).max() )

        rms = np.einsum('ia,ia->', r_T1 / self.Dia, r_T1 / self.Dia,optimize=True)
        rms += np.einsum('ijab,ijab->', r_T2 / self.Dijab, r_T2 / self.Dijab,optimize=True)

        return np.sqrt(rms)

    def update2(self):

        ### Build OEI intermediates
        Fae = self.build_Fae()
        Fmi = self.build_Fmi()
        Fme = self.build_Fme()

        #### Build residual of T1 equations by spin adaption of  Eqn 1:
        r_T1 = self.get_F('ov').copy()
        r_T1 += ndot('ie,ae->ia', self.t1, Fae)
        r_T1 -= ndot('ma,mi->ia', self.t1, Fmi)
        r_T1 += ndot('imae,me->ia', self.t2, Fme, prefactor=2.0)
        r_T1 += ndot('imea,me->ia', self.t2, Fme, prefactor=-1.0)
        r_T1 += ndot(
            'nf,nafi->ia', self.t1, self.get_MO('ovvo'), prefactor=2.0)
        r_T1 += ndot(
            'nf,naif->ia', self.t1, self.get_MO('ovov'), prefactor=-1.0)
        r_T1 += ndot(
            'mief,maef->ia', self.t2, self.get_MO('ovvv'), prefactor=2.0)
        r_T1 += ndot(
            'mife,maef->ia', self.t2, self.get_MO('ovvv'), prefactor=-1.0)
        r_T1 -= ndot(
            'mnae,nmei->ia', self.t2, self.get_MO('oovo'), prefactor=2.0)
        r_T1 -= ndot(
            'mnae,nmie->ia', self.t2, self.get_MO('ooov'), prefactor=-1.0)

        ### Build residual of T2 equations by spin adaptation of Eqn 2:
        # <ij||ab> ->  <ij|ab>
        #   spin   ->  spin-adapted (<alpha beta| alpha beta>)
        r_T2 = self.get_MO('oovv').copy()

        # Conventions used:
        #   P(ab) f(a,b) = f(a,b) - f(b,a)
        #   P(ij) f(i,j) = f(i,j) - f(j,i)
        #   P^(ab)_(ij) f(a,b,i,j) = f(a,b,i,j) + f(b,a,j,i)

        # P(ab) {t_ijae Fae_be}  ->  P^(ab)_(ij) {t_ijae Fae_be}
        tmp = ndot('ijae,be->ijab', self.t2, Fae)
        r_T2 += tmp
        r_T2 += tmp.swapaxes(0, 1).swapaxes(2, 3)

        # P(ab) {-0.5 * t_ijae t_mb Fme_me} -> P^(ab)_(ij) {-0.5 * t_ijae t_mb Fme_me}
        tmp = ndot('mb,me->be', self.t1, Fme)
        first = ndot('ijae,be->ijab', self.t2, tmp, prefactor=0.5)
        r_T2 -= first
        r_T2 -= first.swapaxes(0, 1).swapaxes(2, 3)

        # P(ij) {-t_imab Fmi_mj}  ->  P^(ab)_(ij) {-t_imab Fmi_mj}
        tmp = ndot('imab,mj->ijab', self.t2, Fmi, prefactor=1.0)
        r_T2 -= tmp
        r_T2 -= tmp.swapaxes(0, 1).swapaxes(2, 3)

        # P(ij) {-0.5 * t_imab t_je Fme_me}  -> P^(ab)_(ij) {-0.5 * t_imab t_je Fme_me}
        tmp = ndot('je,me->jm', self.t1, Fme)
        first = ndot('imab,jm->ijab', self.t2, tmp, prefactor=0.5)
        r_T2 -= first
        r_T2 -= first.swapaxes(0, 1).swapaxes(2, 3)

        # Build TEI Intermediates
        tmp_tau = self.build_tau()
        Wmnij = self.build_Wmnij()
        Wmbej = self.build_Wmbej()
        Wmbje = self.build_Wmbje()
        Zmbij = self.build_Zmbij()

        # 0.5 * tau_mnab Wmnij_mnij  -> tau_mnab Wmnij_mnij
        # This also includes the last term in 0.5 * tau_ijef Wabef
        # as Wmnij is modified to include this contribution.
        r_T2 += ndot('mnab,mnij->ijab', tmp_tau, Wmnij, prefactor=1.0)

        # Wabef used in eqn 2 of reference 1 is very expensive to build and store, so we have
        # broken down the term , 0.5 * tau_ijef * Wabef (eqn. 7) into different components
        # The last term in the contraction 0.5 * tau_ijef * Wabef is already accounted
        # for in the contraction just above.

        # First term: 0.5 * tau_ijef <ab||ef> -> tau_ijef <ab|ef>
        r_T2 += ndot(
            'ijef,abef->ijab', tmp_tau, self.get_MO('vvvv'), prefactor=1.0)

        # Second term: 0.5 * tau_ijef (-P(ab) t_mb <am||ef>)  -> -P^(ab)_(ij) {t_ma * Zmbij_mbij}
        # where Zmbij_mbij = <mb|ef> * tau_ijef
        tmp = ndot('ma,mbij->ijab', self.t1, Zmbij)
        r_T2 -= tmp
        r_T2 -= tmp.swapaxes(0, 1).swapaxes(2, 3)

        # P(ij)P(ab) t_imae Wmbej -> Broken down into three terms below
        # First term: P^(ab)_(ij) {(t_imae - t_imea)* Wmbej_mbej}
        tmp = ndot('imae,mbej->ijab', self.t2, Wmbej, prefactor=1.0)
        tmp += ndot('imea,mbej->ijab', self.t2, Wmbej, prefactor=-1.0)
        r_T2 += tmp
        r_T2 += tmp.swapaxes(0, 1).swapaxes(2, 3)

        # Second term: P^(ab)_(ij) t_imae * (Wmbej_mbej + Wmbje_mbje)
        tmp = ndot('imae,mbej->ijab', self.t2, Wmbej, prefactor=1.0)
        tmp += ndot('imae,mbje->ijab', self.t2, Wmbje, prefactor=1.0)
        r_T2 += tmp
        r_T2 += tmp.swapaxes(0, 1).swapaxes(2, 3)

        # Third term: P^(ab)_(ij) t_mjae * Wmbje_mbie
        tmp = ndot('mjae,mbie->ijab', self.t2, Wmbje, prefactor=1.0)
        r_T2 += tmp
        r_T2 += tmp.swapaxes(0, 1).swapaxes(2, 3)

        # -P(ij)P(ab) {-t_ie * t_ma * <mb||ej>} -> P^(ab)_(ij) {-t_ie * t_ma * <mb|ej>
        #                                                      + t_ie * t_mb * <ma|je>}
        tmp = ndot('ie,ma->imea', self.t1, self.t1)
        tmp1 = ndot('imea,mbej->ijab', tmp, self.get_MO('ovvo'))
        r_T2 -= tmp1
        r_T2 -= tmp1.swapaxes(0, 1).swapaxes(2, 3)
        tmp = ndot('ie,mb->imeb', self.t1, self.t1)
        tmp1 = ndot('imeb,maje->ijab', tmp, self.get_MO('ovov'))
        r_T2 -= tmp1
        r_T2 -= tmp1.swapaxes(0, 1).swapaxes(2, 3)

        # P(ij) {t_ie <ab||ej>} -> P^(ab)_(ij) {t_ie <ab|ej>}
        tmp = ndot(
            'ie,abej->ijab', self.t1, self.get_MO('vvvo'), prefactor=1.0)
        r_T2 += tmp
        r_T2 += tmp.swapaxes(0, 1).swapaxes(2, 3)

        # P(ab) {-t_ma <mb||ij>} -> P^(ab)_(ij) {-t_ma <mb|ij>}
        tmp = ndot(
            'ma,mbij->ijab', self.t1, self.get_MO('ovoo'), prefactor=1.0)
        r_T2 -= tmp
        r_T2 -= tmp.swapaxes(0, 1).swapaxes(2, 3)

        ### Update T1 and T2 amplitudes
        self.t1 += (r_T1 / self.Dia / 100)
        self.t2 += (r_T2 / self.Dijab / 100)

        rms = np.einsum('ia,ia->', r_T1 / self.Dia, r_T1 / self.Dia,optimize=True)
        rms += np.einsum('ijab,ijab->', r_T2 / self.Dijab, r_T2 / self.Dijab,optimize=True)

        return np.sqrt(rms)

    def compute_corr_energy(self):
        CCSDcorr_E = 2.0 * np.einsum('ia,ia->', self.get_F('ov'), self.t1,optimize=True)
        tmp_tau = self.build_tau()
        CCSDcorr_E += 2.0 * np.einsum('ijab,ijab->', tmp_tau,
                                      self.get_MO('oovv'),optimize=True)
        CCSDcorr_E -= 1.0 * np.einsum('ijab,ijba->', tmp_tau,
                                      self.get_MO('oovv'),optimize=True)

        self.ccsd_corr_e = CCSDcorr_E
        self.ccsd_e = self.rhf_e + self.ccsd_corr_e
        return CCSDcorr_E

    def compute_energy(self,
                       e_conv=1e-7,
                       r_conv=1e-7,
                       maxiter=100,
                       max_diis=8,
                       start_diis=1,
                       iterate=True):

        ### Start Iterations
        ccsd_tstart = time.time()

        # Compute MP2 energy
        CCSDcorr_E_old = self.compute_corr_energy()
        self.StartEnergy=CCSDcorr_E_old
        print(
            "CCSD Iteration %3d: CCSD correlation = %.15f   dE = % .5E   MP2" %
            (0, CCSDcorr_E_old, -CCSDcorr_E_old))

        # Set up DIIS before iterations begin
        diis_object = helper_diis(self.t1, self.t2, max_diis)

        # Iterate!
        for CCSD_iter in range(1, maxiter + 1):
            if iterate == True:
                rms = self.update()
            else:
                rms=0

            # Compute CCSD correlation energy
            CCSDcorr_E = self.compute_corr_energy()

            # Print CCSD iteration information
            print(
                'CCSD Iteration %3d: CCSD correlation = %.15f   dE = % .5E   DIIS = %d'
                % (CCSD_iter, CCSDcorr_E, (CCSDcorr_E - CCSDcorr_E_old),
                   diis_object.diis_size))

            # Check convergence
            if (abs(CCSDcorr_E - CCSDcorr_E_old) < e_conv and rms < r_conv):
                print('\nCCSD has converged in %.3f seconds!' %
                      (time.time() - ccsd_tstart))
                self.steps = CCSD_iter
                self.FinalEnergy=CCSDcorr_E
                tmp_tau = self.build_tau()


                self.pairs=2*tmp_tau*self.get_MO('oovv')
                self.pairs-= np.swapaxes(self.get_MO('oovv'),2,3)*tmp_tau
                self.pairs = np.sum(self.pairs,axis=(2,3))




                if self.Triples:
                    print ('Starting (T) calculation ~')
                    #Put all the amplitudes and orbitals back in canonical for the (T)
                    ccsdt_start=time.time()
                    F=self.F.copy()
                    if (self.Loc_occ != False):
                        self.t2 = transform_occ_back(self.t2, self.occ_U)
                        #self.t1= np.einsum('ij,jk,kl->il',self.occ_U,self.t1,np.asarray(self.occ_U).conj().T)
                    if (self.Loc_vir != False):
                        self.t2 = transform_vir_back(self.t2, self.vir_U)
                    if ((self.Loc_occ != False) or (self.Loc_vir != False)):
                        self.MO=self.MO_canonical.copy()
                        F=self.F.copy()
                        self.F=self.F_canonical
                        self.t1 = np.zeros((self.nocc, self.nvirt))
                        #I'm not sure how to put t1 back to canonical
                        self.compute_t1()
                    tmp = ndot('kjcd,badi->ijkabc', self.t2, self.get_MO('vvvo')).copy().astype('float32')
                    tmp -= ndot('ilab,cjkl->ijkabc', self.t2, self.get_MO('vooo'))
                    W=tmp.copy()
                    W += np.einsum('ijkabc->ikjacb', tmp,optimize=True)
                    W += np.einsum('ijkabc->kijcab', tmp,optimize=True)
                    W += np.einsum('ijkabc->kjicba', tmp,optimize=True)
                    W += np.einsum('ijkabc->jkibca', tmp,optimize=True)
                    W += np.einsum('ijkabc->jikbac', tmp,optimize=True)
                    del tmp
                    gc.collect()
                    V=W.copy()
                    V+= np.einsum('ia,bcjk->ijkabc', self.t1, self.get_MO('vvoo'),optimize=True)
                    V+= np.einsum('jb,acik->ijkabc', self.t1, self.get_MO('vvoo'),optimize=True)
                    V+= np.einsum('kc,abij->ijkabc', self.t1, self.get_MO('vvoo'),optimize=True)

                    #First= (4*W + np.einsum('ijkabc->ijkbca',W,optimize=True) + np.einsum('ijkabc->ijkcab',W,optimize=True))
                    #Second= V-np.einsum('ijkabc->ijkcba',V,optimize=True) )



                    CCSD_pairs=self.pairs.copy()
                    #self.pairs=(First*Second/Dijkabc)/3
                    self.pairs=(4*W + np.einsum('ijkabc->ijkbca',W,optimize=True) + np.einsum('ijkabc->ijkcab',W,optimize=True))*(V-np.einsum('ijkabc->ijkcba',V,optimize=True) )  / 3

                    del W
                    del V
                    gc.collect()
                    Focc = np.diag(self.get_F('oo'))
                    Fvir = np.diag(self.get_F('vv'))
                    Dijkabc = Focc.reshape(-1, 1, 1, 1, 1, 1) + Focc.reshape(-1, 1, 1, 1, 1) + Focc.reshape(-1, 1, 1, 1)
                    Dijkabc = Dijkabc - Fvir.reshape(-1, 1, 1) - Fvir.reshape(-1, 1) - Fvir
                    self.pairs=self.pairs/Dijkabc
                    Pert_T=np.sum(self.pairs)


                    #self.triple_pairs=np.einsum('ijkabc->ik', self.pairs, optimize=True)
                    self.triple_pairs=distribute_triples(np.einsum('ijkabc->ijk', self.pairs, optimize=True),CCSD_pairs)

                    self.pairs=self.triple_pairs + CCSD_pairs
                    if self.triples_only == True:
                        self.pairs=self.triple_pairs
                    #Pert_T=np.sum(self.pairs)
                    CCSD_T_E = CCSDcorr_E + Pert_T + self.rhf_e
                    self.F=F

                    print ('(T) has converged in '+str(time.time()-ccsdt_start) +' seconds')
                    print('CCSD Correlation Energy:           '+str(CCSDcorr_E))
                    print('Perturbative (T) correlation energy:     ' + str(Pert_T))
                    print ('CCSD(T) correlation energy: {}'.format(np.sum(self.pairs)))
                    print('Total CCSD(T) energy:                   ' + str(CCSD_T_E))
                    self.FinalEnergy= CCSDcorr_E + Pert_T


                return CCSDcorr_E

            # Update old energy
            CCSDcorr_E_old = CCSDcorr_E

            #  Add the new error vector
            diis_object.add_error_vector(self.t1, self.t2)

            if CCSD_iter >= start_diis:
                self.t1, self.t2 = diis_object.extrapolate(self.t1, self.t2)

    def updatet1(self):

        ### Build OEI intermediates
        Fae = self.build_Fae()
        Fmi = self.build_Fmi()
        Fme = self.build_Fme()

        #### Build residual of T1 equations by spin adaption of  Eqn 1:
        r_T1 = self.get_F('ov').copy()
        r_T1 += ndot('ie,ae->ia', self.t1, Fae)
        r_T1 -= ndot('ma,mi->ia', self.t1, Fmi)
        r_T1 += ndot('imae,me->ia', self.t2, Fme, prefactor=2.0)
        r_T1 += ndot('imea,me->ia', self.t2, Fme, prefactor=-1.0)
        r_T1 += ndot(
            'nf,nafi->ia', self.t1, self.get_MO('ovvo'), prefactor=2.0)
        r_T1 += ndot(
            'nf,naif->ia', self.t1, self.get_MO('ovov'), prefactor=-1.0)
        r_T1 += ndot(
            'mief,maef->ia', self.t2, self.get_MO('ovvv'), prefactor=2.0)
        r_T1 += ndot(
            'mife,maef->ia', self.t2, self.get_MO('ovvv'), prefactor=-1.0)
        r_T1 -= ndot(
            'mnae,nmei->ia', self.t2, self.get_MO('oovo'), prefactor=2.0)
        r_T1 -= ndot(
            'mnae,nmie->ia', self.t2, self.get_MO('ooov'), prefactor=-1.0)

        self.t1 += r_T1 / self.Dia

        rms = np.einsum('ia,ia->', r_T1 / self.Dia, r_T1 / self.Dia)

        return np.sqrt(rms)


    def compute_t1(self,
                       e_conv=1e-7,
                       r_conv=1e-3,
                       maxiter=100,
                       max_diis=8,
                       start_diis=1):

        ### Start Iterations
        ccsd_tstart = time.time()

        # Compute MP2 energy
        CCSDcorr_E_old = self.compute_corr_energy()
        print(
            "CCSD Iteration %3d: CCSD correlation = %.15f   dE = % .5E   MP2" %
            (0, CCSDcorr_E_old, -CCSDcorr_E_old))

        # Set up DIIS before iterations begin
        diis_object = helper_diis(self.t1, self.t2, max_diis)

        # Iterate!
        for CCSD_iter in range(1, maxiter + 1):

            rms = self.updatet1()

            # Compute CCSD correlation energy
            CCSDcorr_E = self.compute_corr_energy()

            # Print CCSD iteration information
            print(
                'CCSD Iteration %3d: CCSD correlation = %.15f   dE = % .5E   DIIS = %d'
                % (CCSD_iter, CCSDcorr_E, (CCSDcorr_E - CCSDcorr_E_old),
                   diis_object.diis_size))

            # Check convergence
            if (abs(CCSDcorr_E - CCSDcorr_E_old) < e_conv and rms < r_conv):
                print('\nCCSD has converged in %.3f seconds!' %
                      (time.time() - ccsd_tstart))
                return CCSDcorr_E

            # Update old energy
            CCSDcorr_E_old = CCSDcorr_E

            #  Add the new error vector
            diis_object.add_error_vector(self.t1, self.t2)

            if CCSD_iter >= start_diis:
                self.t1, self.t2 = diis_object.extrapolate(self.t1, self.t2)




    def PlotMP2Approx(self, Percents,c=list([1e-4,1e-5,1e-6,1e-7,1e-8,1e-9]), ylab='Percent CCSD Correlation',savefig=False):
            fig=sns.lineplot(x=c, y=Percents)
            plt.xscale('log')
            plt.gca().invert_xaxis()
            matplotlib.rcParams['font.family'] = "sans-serif"
            matplotlib.rcParams['font.sans-serif'] = "Times New Roman"
            fig.set(xlabel='MP2 cutoff value', ylabel=ylab)
            if savefig == True:
                plt.savefig(ylab+'.png', dpi=300)
            plt.show()
            #fig.xlabel='MP2 cutoff value'
            #fig.ylabel='Percent CCSD correlation'
            return fig


    def TestMP2Approx(self, savefig=False,c=list([1e-4,1e-5,1e-6,1e-7,1e-8,1e-9])):
        Included=list()
        Excluded=list()
        Error=list()
        PercentCorrelation=list()


        #psi4.core.clean()
        self.compute_energy()
        t2=self.t2.copy()
        MP2=np.abs((self.StartEnergy)-(self.FinalEnergy)).copy()
        FinalEnergy=self.FinalEnergy.copy()
        for val in c:

            print ("Cutoff is "+str(val))
            b=( np.abs(self.triplecheck) < val)
            g=( np.abs(self.triplecheck) > val)
            self.t2[b]=self.t2start[b]
            self.t2[g]=t2[g]
            self.compute_t1()
            self.compute_energy(iterate=False)
            Included.append(np.sum(b == False))
            Excluded.append(np.sum(b == True))

            print (MP2)
            print (self.StartEnergy)
            print (FinalEnergy)
            Error.append(np.abs(self.StartEnergy-FinalEnergy)*627.509)
            PercentCorrelation.append((1-((np.abs(self.StartEnergy-FinalEnergy))/np.abs(MP2))))
            print ('We managed to recover this percentage of CCSD correlation'+ str((1-((np.abs(self.StartEnergy-FinalEnergy))/np.abs(MP2)))))
            print ('Included amplitudes:'+ str(np.sum(g== True)))
            print ('Excluded amplitudes:'+ str(np.sum(b == True)))
        fig1=self.PlotMP2Approx(PercentCorrelation,c=c, savefig=savefig)
        plt.show()
        Configs=np.array(Included)/(np.array(Included)+np.array(Excluded))
        fig2=self.PlotMP2Approx(Configs,ylab='Percent Configurations Used', savefig=savefig, c=c)
        plt.show()
        print (Configs)
        print (PercentCorrelation)
        return Configs, PercentCorrelation, Error






