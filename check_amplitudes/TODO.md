# TO-DO List
- [ ] Create a conda environment for this project
- [ ] Install the following:
```python
conda install eigen pybind11 boost psi4 python=3.12 -y -c conda-forge && \
python -m pip install -r requirements.txt
```
- [ ] Compare the Psi4 and PySCF $t_{1}$- and $t_{2}$-amplitudes, along with the correlation energy, across systems and basis set sizes
- [ ] Look into how to make the parameters the same, if possible, e.g. these are the Psi4 parameters I would try to make pyscf match these if possible:
        - 'scf_type': 'pk'
        - 'reference': 'rhf'
        - 'mp2_type': 'conv'
        - 'e_convergence': 1e-8
        - 'd_convergence': 1e-8
- [ ] Make sure the spin is correct (PySCF uses $2S$, while Psi4 uses the spin multipliticy $2S+1$, which is what I have in the xyz files with the charge=0). Read the documentation for both to see how to do this, for Psi4 just load in the structures for now since I have those in the files. For the "open-shell" systems, that have a  doublet spin-multiplicity ($2S+1=2$) use restricted open-shell Hartree-Fock (ROHF) in both PySCF and Psi4, and for "closed-shell" systems that have a singlet spin-multiplicity ($2S+1=1$) use restricted Hartree-Fock (RHF), which should be the default in both.
- [ ] We also need to check with and with out frozen orbitals (see if there is an automated way to do that in PySCF, like in Psi4, if not, I can send the parameters for these diatomics)