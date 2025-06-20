#!/usr/bin/env python
# coding: utf-8
import shutil
import numpy as np
import psi4
from helper_CC_ML_spacial import *
import os
from sklearn.model_selection import train_test_split

features = ['Evir1', 'Hvir1', 'Jvir1', 'Kvir1', 'Evir2', 'Hvir2', 'Jvir2', 'Kvir2', 'Eocc1', 'Jocc1', 'Kocc1', 'Hocc1',
            'Eocc2', 'Jocc2', 'Kocc2', 'Hocc2', 'Jia1', 'Jia2', 'Kia1', 'Kia2', 'diag', 'orbdiff', 'doublecheck', 't2start', 't2mag', 't2sign', 'Jia1mag', 'Jia2mag', 'Kia1mag', 'Kia2mag','pairenergy','triplecheck']
'''
Key:
Letters:
E-Energy of the orbital
H-1e contribution to the orbital energy
J-Coulombic contribution to orbital energy
K-Exchange contribution to orbital energy
Placement:
occ or virt, you get this..
Number:
is it electron one or two from the two electron excitation


Jia1- coulomb integral between orbital occ1 and vir1
Jia2 " but 2
Kia1 - exchange integral between orbital 
Kia2 Same but exchange integral
diag - is it on the diagonal, aka, are the two excited electrons going to the same orbital **this is important fyi
orbdiff - (Evir2 + Evir1 - Eocc1 - Eocc2)
doublecheck - full 2electron integral
t2start - INITIAL MP2 amplitude **this is the inital guess
t2mag - np.log10(np.absolute(t2start)) ~ this is going to be a common trend, since it is more straightforward for ML algorithms to understand
t2sign - (t2start > 1)? 
Jia1mag - np.log10(np.absolute(feature))
Jia2mag np.log10(np.absolute(feature))
Kia1mag  np.log10(np.absolute(feature))
Kia2mag np.log10(np.absolute(feature))

'''
def prune_amps(A, Bigmatrix, Bigamp):
    fill=np.zeros(A.t2.shape)
    fill[:]=True
    for i in range (0,A.t2.shape[0]):
        for j in  range (0,A.t2.shape[0]):
            for a in range (0,A.t2.shape[2]):
                for b in range (0,A.t2.shape[2]):

                    if i==j and a > b:
                        fill[i,j,b,a]=False

                    if i> j and a!= b:
                        fill[j,i,b,a]=False
                    

    fill=fill.reshape(A.nocc*A.nocc*A.nvirt*A.nvirt)
    Bigmatrix=np.delete(Bigmatrix, np.where(fill == False), axis=0)
    Bigamp=np.delete(Bigamp, np.where(fill == False), axis=0)
    return Bigmatrix,Bigamp

def GetAmps(Foldername, occ=False, vir=False, cutoff=False, xyz=True, basis='sto-3g'):
    i=1
    for filename in os.listdir(str(Foldername)):
        if filename.endswith('.xyz'):
            psi4.core.clean()
            print (filename)            
            stuff=str(str(Foldername)+filename)
            text = open(stuff, 'r').read()
            
            #I added this because some of the older files were not xyz format
            #This just uses .xyz files standard, if it was old psi4 format, it just takes the file as is
            if xyz==True: 
                qmol = psi4.qcdb.Molecule.from_string(text, dtype='xyz')
                mol = psi4.geometry(qmol.create_psi4_string_from_molecule()+ 'symmetry c1')                
            else:                                
                mol = psi4.geometry(text)               
            psi4.core.clean()
            psi4.set_options({'basis':        basis,#'6-31g',
                              'scf_type':     'pk',
                              'reference':    'rhf',
                              'mp2_type':     'conv',
                              'e_convergence': 1e-8,

                              'd_convergence': 1e-8})
        #    rhf_e, scf_wfn = psi4.energy('scf', return_wfn=True)
         #   scf_e, scf_wfn = psi4.energy('scf', return_wfn=True)
            A=HelperCCEnergy(mol, Loc_occ=occ, Loc_vir=vir)

            A.compute_energy()
            matrixsize=A.nocc*A.nocc*A.nvirt*A.nvirt
            Bigmatrix=np.zeros([matrixsize, len(features)])
            for x in range(0,len(features)):
                Bigmatrix[:, x]=getattr(A, features[x]).reshape(matrixsize)
            Bigamp=A.t2.reshape(matrixsize,1)
            Bigmatrix,Bigamp=prune_amps(A,Bigmatrix,Bigamp)
            if (cutoff != False):
                a=Bigmatrix[:,31]
                Bigmatrix = np.delete(Bigmatrix, np.where(np.abs(a) < cutoff), axis=0)            
                Bigamp=np.delete(Bigamp, np.where(np.abs(a) < cutoff), axis=0)
            
            if i==1:
                Bigfeatures=Bigmatrix
                Bigamps=Bigamp
                i=2
            else:
                Bigfeatures=np.vstack((Bigfeatures,Bigmatrix))
                Bigamps=np.vstack((Bigamps,Bigamp))


       # shape=np.size(A.t2start.flatten(),0)
       # structurenumber=40
        #total=(shape * structurenumber)
        #array=np.reshape(Bigfeatures, (total,len(features)))
    array=Bigfeatures
    finalamps=Bigamps
    return array,finalamps

def test_prune(A):
    fill=np.zeros(A.t2.shape)
    fill[:]=True
    for i in range (0,A.t2.shape[0]):
        for j in  range (0,A.t2.shape[0]):
            for a in range (0,A.t2.shape[2]):
                for b in range (0,A.t2.shape[2]):

                    if i==j and a > b:
                        for x in range (0,len(features)):
                            getattr(A, features[x])[i,j,b,a]=getattr(A, features[x])[i,j,a,b]
                        fill[i,j,b,a]=False

                    if i> j and a!= b:
                        for x in range (0,len(features)):
                            getattr(A, features[x])[j,i,b,a]=getattr(A, features[x])[i,j,a,b]
                        fill[j,i,b,a]=False



def Test(Foldername, occ=False, vir=False, cutoff=False, xyz=True, basis='sto-3g', gocanon=False, scale=False):
    steps=list()
    difference=list()
    supalist=list()
    startenergy=list()
    finalenergy=list()
    filenames=list()
    rhfenergy=list()
    for filename in os.listdir(Foldername):
        if filename.endswith('.xyz'):
                psi4.core.clean()
                filenames.append(filename)
                print ("filename is "+filename)
                stuff=str(Foldername+filename)
                text = open(stuff, 'r').read()
                if xyz==True:             
                    qmol = psi4.qcdb.Molecule.from_string(text, dtype='xyz')
                    mol = psi4.geometry(qmol.create_psi4_string_from_molecule()+ 'symmetry c1')                
                else:                                
                    mol = psi4.geometry(text)  


                psi4.set_options({'basis':        basis,
                                  'scf_type':     'pk',
                                  'reference':    'rhf',
                                  'mp2_type':     'conv',
                                  'e_convergence': 1e-8,
                                  'd_convergence': 1e-8})
            
                A=HelperCCEnergy(mol, Loc_occ=occ, Loc_vir=vir)
                test_prune(A)
                matrixsize=A.nocc*A.nocc*A.nvirt*A.nvirt
                Xnew=np.zeros([1,matrixsize,len(features)])
                for x in range (0,len(features)):
                    Xnew[0,:,x]=getattr(A, features[x]).reshape(matrixsize)
                Xnew=np.reshape(Xnew, (matrixsize,len(features)))

                if (cutoff != False):
                    a=Xnew[:,31]
                    #i=(np.abs(a) < cutoff)           
                    j=(np.abs(a) > cutoff)
                    
                    #MP2set = np.delete(X_new, np.where(np.abs(a) < cutoff), axis=0)            
                    MLset = np.delete(Xnew, np.where(np.abs(a) < cutoff), axis=0)
                    X_new_scaled= scaler.transform(MLset)
                    a[j]=np.squeeze(model.predict(X_new_scaled))
                    a=a.reshape(A.nocc,A.nocc,A.nvirt,A.nvirt)
                else:
                    X_new_scaled= scaler.transform(X_new)
                    a=model.predict(X_new_scaled)
                    a=a.reshape(A.nocc,A.nocc,A.nvirt,A.nvirt)
                              
                if (scale != False):
                    a=a/scale   

                A.t2=a
                #print ('Changing back to canonical for solution')
                #if (A.Loc_occ != False):
                #    A.t2 = transform_occ_back(A.t2, A.occ_U)
    #self.t1= np.einsum('ij,jk,kl->il',self.occ_U,self.t1,np.asarray(self.occ_U).conj().T)
                #if (A.Loc_vir != False):
                #    A.t2 = transform_vir_back(A.t2, A.vir_U)

                #if ((A.Loc_occ != False) or (A.Loc_vir != False)):
                #    A.MO=A.MO_canonical.copy()
                #    A.F=A.F_canonical.copy()
                if gocanon == True:
                    A=go_canonical(A)
                A.compute_t1()
                A.compute_energy(r_conv=1e-5, e_conv=1e-8)
                rhfenergy.append(A.rhf_e)
                startenergy.append(A.StartEnergy)
                finalenergy.append(A.FinalEnergy)
    
    difference.append(sum( np.abs(np.asarray(startenergy) - np.asarray(finalenergy))) /len(startenergy))
    print ('Filenames')
    print (filenames)
    print ('Start Energy')
    print (np.add(np.array(startenergy),np.array(rhfenergy)))
    print ('Individual Differences')
    
    print (np.abs(np.asarray(startenergy) - np.asarray(finalenergy)))
    print ('Average Differences')
    print (difference)
    return filenames,np.abs(np.asarray(startenergy) - np.asarray(finalenergy)), np.asarray(startenergy), np.asarray(finalenergy)


def checkMP2Approx(filename, basis='6-31g',occ=False, vir=False,xyz=True, savefig=False, c=list((1e-2,5e-3,1e-3,5e-4, 1e-4, 5e-5, 1e-5, 1e-6))):
    stuff=str(filename)
    text = open(stuff, 'r').read()

    #I added this because some of the older files were not xyz format
    #This just uses .xyz files standard, if it was old psi4 format, it just takes the file as is
    if xyz==True:
        qmol = psi4.qcdb.Molecule.from_string(text, dtype='xyz')
        mol = psi4.geometry(qmol.create_psi4_string_from_molecule()+ 'symmetry c1')
    else:
        mol = psi4.geometry(text)

    psi4.core.clean()
    psi4.set_options({'basis':        basis,#'6-31g',
                      'scf_type':     'pk',
                      'reference':    'rhf',
                      'mp2_type':     'conv',
                      'e_convergence': 1e-8,
                      'd_convergence': 1e-8})

    wf_object=HelperCCEnergy(mol, Loc_occ=occ, Loc_vir=vir)
    a,b,c=wf_object.TestMP2Approx(savefig=savefig, c=c)
    return a,b,c

def SplitFolder(Original, testsize=0.10):
    if os.path.isdir(Original+'train'):
        shutil.rmtree(Original+'train')

    if os.path.isdir(Original+'test'):
        shutil.rmtree(Original+'test')

    X=y=os.listdir(Original)

    os.mkdir(Original+'train')
    os.mkdir(Original+'test')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, random_state=0)
    #print (X_train)

    for x in X_train:
        shutil.copyfile(Original+x, Original+'train/'+x)
    for x in X_test:
        shutil.copyfile(Original+x, Original+'test/'+x)



def CompareLocalization(filename, xyz=True, savefig=False,basis='sto-3g',c=list((1e-2,5e-3,1e-3,5e-4, 1e-4, 5e-5, 1e-5, 1e-6))):
    can_a,can_b,can_c=checkMP2Approx(filename,basis=basis, xyz=xyz, c=c)
    boys_a,boys_b,boys_c=checkMP2Approx(filename,basis=basis, occ='BOYS', vir='BOYS', xyz=xyz, c=c)
    pm_a,pm_b,pm_c=checkMP2Approx(filename,basis=basis, occ='PM', vir='PM', xyz=xyz, c=c)
    df=pd.DataFrame()
    df['Cutoff']=np.array(c)
    df['Percent_Configs']=can_a
    df['Percent_Corr']=can_b
    df['Absolute_err']=can_c
    df['Orbitals'] = df.apply(lambda x: 'Canonical', axis=1)

    df2=pd.DataFrame()
    df2['Cutoff']=np.array(c)
    df2['Percent_Configs']=boys_a
    df2['Percent_Corr']=boys_b
    df2['Absolute_err']=boys_c
    df2['Orbitals'] = df.apply(lambda x: 'Boys', axis=1)

    df3=pd.DataFrame()
    df3['Cutoff']=np.array(c)
    df3['Percent_Configs']=pm_a
    df3['Percent_Corr']=pm_b
    df3['Absolute_err']=pm_c
    df3['Orbitals'] = df.apply(lambda x: 'PM', axis=1)

    df=df.append(df2)
    df=df.append(df3)

    print (df)

    fig=sns.lineplot(x=df['Cutoff'], y=df['Percent_Configs'], hue=df['Orbitals'])
    plt.xscale('log')
    plt.gca().invert_xaxis()
    matplotlib.rcParams['font.family'] = "sans-serif"
    matplotlib.rcParams['font.sans-serif'] = "Times New Roman"

    fig.set(xlabel='MP2 cutoff value', ylabel='Percent Configurations')
    plt.tight_layout()
    if savefig == True:
        plt.savefig(filename+'_Config.png', dpi=350)
        print ('File Saved to: '+str(filename+'_cutoff.png'))
    plt.show()
            #fig.xlabel='MP2 cutoff value'
            #fig.ylabel='Percent CCSD correlation'


    fig=sns.lineplot(x=df['Cutoff'], y=df['Percent_Corr'], hue=df['Orbitals'])
    plt.xscale('log')
    plt.gca().invert_xaxis()
    plt.tight_layout()
    matplotlib.rcParams['font.family'] = "sans-serif"
    matplotlib.rcParams['font.sans-serif'] = "Times New Roman"
    fig.set(xlabel='MP2 cutoff value', ylabel='Percent CCSD Correlation Captured')
    if savefig == True:
        plt.savefig(filename+'_Correlation.png', dpi=350)
        print ('File Saved to: '+str(filename+'_cutoff.png'))
    plt.show()
            #fig.xlabel='MP2 cutoff value'
            #fig.ylabel='Percent CCSD correlation'

    plt.show()

    fig=sns.lineplot(x=df['Cutoff'], y=df['Absolute_err'], hue=df['Orbitals'])
    plt.xscale('log')
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.ylim(0,1)
    matplotlib.rcParams['font.family'] = "sans-serif"
    matplotlib.rcParams['font.sans-serif'] = "Times New Roman"
    fig.set(xlabel='MP2 cutoff value', ylabel='Absolute Error (kcal/mol)')
    if savefig == True:
        plt.savefig(filename+'_Error.png', dpi=350)
        print ('File Saved to: '+str(filename+'_Error.png'))
    plt.show()
    return df



