



import psi4
import numpy as np
import numpy as np
import pandas as pd
#import tensorflow as tf
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import cm
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from psi4 import core
from helper_CC_ML_spacial import *
import helper_ML_tools
#import matplotlib as mpl
from sklearn.preprocessing import MinMaxScaler
import os




#this this has been updated for for e_ij + e_ji = 2e_ij, so only e_ij is fed through to input
#this this has been updated for for e_ij + e_ji = 2e_ij, so only e_ij is fed through to input
featurelist=list()


def GenerateFeatures(wf_object, Miller=False, values=4):
    b =   wf_object.triplecheck

    #c = np.log10(np.absolute(wf_object.get_MO('oovv')*wf_object.t2start))
    ##infcheck=(c == -np.inf)
    #c[infcheck]=20

    #d = wf_object.pairs

    if Miller == False:

        for i in range(b.shape[0]):
            for j in range (b.shape[1]):
                featurelist.clear()

                ind=np.argsort(b[i,j].flatten(),axis=0)
                new=np.sum(b[i,j])#0
                featurelist.append('Pair_Energy')
                new=np.hstack((new,wf_object.MO[i,i,j,j]))
                featurelist.append('coulomb_ij')
                new=np.hstack((new,np.take_along_axis(wf_object.screen1[i,j].flatten(), ind, axis=0)[:values]))#1,2,3,4
                featurelist.append('i_screen1')
                new=np.hstack((new,np.take_along_axis(wf_object.screen2[i,j].flatten(), ind, axis=0)[:values]))#9,10,11,12
                featurelist.append('j_screen1')
                new=np.hstack((new,np.take_along_axis(wf_object.screenvirt[i,j].flatten(), ind, axis=0)[:values]))#9,10,11,12
                new=np.hstack((new,np.take_along_axis(b[i,j].flatten(), ind, axis=0)[:values]))#17,18,19,20

                featurelist.append('triplecheck1')



                if i==j:
                    ind=np.argsort(b[i,j].flatten(),axis=0)
                    new=np.sum(b[i,j])#0
                    featurelist.append('Pair_Energy')
                    #new=np.hstack((new,wf_object.MO[i,i,j,j]))
                    featurelist.append('coulomb_ij')
                    new=np.hstack((new,np.take_along_axis(wf_object.screen1[i,j].flatten(), ind, axis=0)[:values]))#1,2,3,4
                    featurelist.append('i_screen1')
                    new=np.hstack((new,np.take_along_axis(wf_object.screen2[i,j].flatten(), ind, axis=0)[:values]))#9,10,11,12
                    featurelist.append('j_screen1')




                    #new=np.hstack((new,np.take_along_axis(wf_object.doublecheck[i,j].flatten(), ind, axis=0)[:values]))
                    #new=np.hstack((new,np.take_along_axis(wf_object.diag[i,j].flatten(), ind, axis=0)[:values]))

                    #new=np.hstack((new,np.take_along_axis(wf_object.screenvirt[i,j].flatten(), ind, axis=0)[:values]))


                    #new=np.hstack((new,np.take_along_axis(wf_object.doublecheck[i,j].flatten(), ind, axis=0)[-values:]))
                    #new=np.hstack((new,np.take_along_axis(wf_object.diag[i,j].flatten(), ind, axis=0)[-values:]))

                    #new=np.hstack((new,np.take_along_axis(wf_object.screenvirt[i,j].flatten(), ind, axis=0)[-values:]))


#9,10,11,12
                    new=np.hstack((new,np.take_along_axis(wf_object.screen1[i,j].flatten(), ind, axis=0)[-values:]))#1,2,3,4
                    featurelist.append('i_screen1')
                    new=np.hstack((new,np.take_along_axis(wf_object.screen2[i,j].flatten(), ind, axis=0)[-values:]))#9,10,11,12
                    featurelist.append('j_screen1')
                    #new=np.hstack((new,np.take_along_axis(wf_object.doublecheck[i,j].flatten(), ind, axis=0)[:values]))
                    #new=np.hstack((new,np.take_along_axis(wf_object.diag[i,j].flatten(), ind, axis=0)[:values]))

                    #new=np.hstack((new,np.take_along_axis(wf_object.screenvirt[i,j].flatten(), ind, axis=0)[:values]))#9,10,11,12
                    new=np.hstack((new,np.take_along_axis(b[i,j].flatten(), ind, axis=0)[:values]))#17,18,19,20
                    new=np.hstack((new,np.take_along_axis(b[i,j].flatten(), ind, axis=0)[-values:]))#17,18,19,20

                    

                  
 
                    one=np.sum(np.take_along_axis(b[i,j].flatten(), ind, axis=0)[-values:])
                    two=np.sum(np.take_along_axis(b[i,j].flatten(), ind, axis=0)[:values])
                    new=np.hstack((new, np.sum(b[i,j])-one-two))
                    featurelist.append('triplecheck1')
                else:
                    ind=np.argsort(b[i,j].flatten(),axis=0)
                    new=np.sum(b[i,j])#0
                    featurelist.append('Pair_Energy')
                    new=np.hstack((new,wf_object.MO[i,i,j,j]))
                    featurelist.append('coulomb_ij')
                    new=np.hstack((new,np.take_along_axis(wf_object.screen1[i,j].flatten(), ind, axis=0)[:values]))#1,2,3,4
                    featurelist.append('i_screen1')
                    new=np.hstack((new,np.take_along_axis(wf_object.screen2[i,j].flatten(), ind, axis=0)[:values]))#9,10,11,12
                    featurelist.append('j_screen1')
                    #new=np.hstack((new,np.take_along_axis(wf_object.screenvirt[i,j].flatten(), ind, axis=0)[:values]))#9,10,11,12
                    #new=np.hstack((new,np.take_along_axis(b[i,j].flatten(), ind, axis=0)[:values]))#17,18,19,20
                    #new=np.hstack((new,np.take_along_axis(b[i,j].flatten(), ind, axis=0)[-values:]))
                    featurelist.append('triplecheck1')
                    #new=np.hstack((new,np.take_along_axis(wf_object.doublecheck[i,j].flatten(), ind, axis=0)[:values]))
                    #new=np.hstack((new,np.take_along_axis(wf_object.diag[i,j].flatten(), ind, axis=0)[:values]))


                    new=np.hstack((new,np.take_along_axis(wf_object.screen1[i,j].flatten(), ind, axis=0)[-values:]))#1,2,3,4
                    featurelist.append('i_screen1')
                    new=np.hstack((new,np.take_along_axis(wf_object.screen2[i,j].flatten(), ind, axis=0)[-values:]))#9,10,11,12
                    featurelist.append('j_screen1')
                    #new=np.hstack((new,np.take_along_axis(wf_object.doublecheck[i,j].flatten(), ind, axis=0)[:values]))
                    #new=np.hstack((new,np.take_along_axis(wf_object.diag[i,j].flatten(), ind, axis=0)[:values]))

                    #new=np.hstack((new,np.take_along_axis(wf_object.screenvirt[i,j].flatten(), ind, axis=0)[:values]))#9,10,11,12
                    #new=np.hstack((new,np.take_along_axis(b[i,j].flatten(), ind, axis=0)[-values:]))#17,18,19,20
                    

                    new=np.hstack((new,np.take_along_axis(b[i,j].flatten(), ind, axis=0)[:values]))#17,18,19,20
                    new=np.hstack((new,np.take_along_axis(b[i,j].flatten(), ind, axis=0)[-values:]))#17,18,19,20

                    one=np.sum(np.take_along_axis(b[i,j].flatten(), ind, axis=0)[-values:])
                    two=np.sum(np.take_along_axis(b[i,j].flatten(), ind, axis=0)[:values])
                    new=np.hstack((new, np.sum(b[i,j])-one-two))
                    featurelist.append('triplecheck1')


                if ((i==0) and (j==0)):
                     a=new.copy()
                     diag=wf_object.pairs[i,j]
                elif ((i==0) and (j==1)):
                     g=new.copy()
                     offdiag=wf_object.pairs[i,j]
                elif (i==j):
                     a=np.vstack((a,new))#41
                     diag=np.vstack((diag,wf_object.pairs[i,j]))
                elif (j > i):
                     g=np.vstack((g,new))#41
                     offdiag=np.vstack((offdiag,wf_object.pairs[i,j]))


        return a,diag,g,offdiag
    else:


        #c = np.log10(np.absolute(wf_object.get_MO('oovv')*wf_object.t2start))
        ##infcheck=(c == -np.inf)
        #c[infcheck]=20
        tmp_tau = wf_object.build_tau()
        d =   2*tmp_tau*wf_object.get_MO('oovv')
        d -=  np.swapaxes(wf_object.get_MO('oovv'),2,3)*tmp_tau
        d = np.array(np.sum(d,axis=(2,3)))



        for i in range(b.shape[0]):
            for j in range(b.shape[1]):


                if i==j:
                    new=wf_object.F[i,i]
                    new=np.hstack((new, wf_object.J1[i,i]))
                    #new=np.hstack((new, wf_object.K1[i,i]))
                    ind=np.argsort(np.diag(wf_object.F[wf_object.nocc:(wf_object.nmo-wf_object.nfzc),wf_object.nocc:(wf_object.nmo-wf_object.nfzc)]),axis=0)
                    new=np.hstack((new, np.sort(np.take_along_axis(wf_object.J1[i,wf_object.nocc:].flatten(), ind, axis=0)[0:4])))
                    new=np.hstack((new, np.sort(np.take_along_axis(wf_object.K1[i,wf_object.nocc:].flatten(), ind, axis=0)[0:4])))

                else:
                    new=np.sort((wf_object.F[i,i],wf_object.F[i,j], wf_object.F[j,j]))
                    new=np.hstack((new, np.sort((wf_object.J1[i,i],wf_object.J1[i,j], wf_object.J1[j,j]))))
                    new=np.hstack((new, wf_object.K1[i,j]))

                    ind=np.argsort(np.diag(wf_object.F[wf_object.nocc:(wf_object.nmo-wf_object.nfzc),wf_object.nocc:(wf_object.nmo-wf_object.nfzc)]),axis=0)
                    new=np.hstack((new, np.sort(np.take_along_axis(wf_object.J1[i,wf_object.nocc:].flatten(), ind, axis=0)[0:4])))
                    new=np.hstack((new, np.sort(np.take_along_axis(wf_object.J1[j,wf_object.nocc:].flatten(), ind, axis=0)[0:4])))
                    new=np.hstack((new, np.sort(np.take_along_axis(wf_object.K1[i,wf_object.nocc:].flatten(), ind, axis=0)[0:4])))
                    new=np.hstack((new, np.sort(np.take_along_axis(wf_object.K1[j,wf_object.nocc:].flatten(), ind, axis=0)[0:4])))



                if ((i==0) and (j==0)):
                     a=new.copy()
                     diag=wf_object.pairs[i,j]
                elif ((i==0) and (j==1)):
                     g=new.copy()
                     offdiag=wf_object.pairs[i,j]
                elif (i==j):
                     a=np.vstack((a,new))#41
                     diag=np.vstack((diag,wf_object.pairs[i,j]))
                elif (j > i):
                     g=np.vstack((g,new))#41
                     offdiag=np.vstack((offdiag,wf_object.pairs[i,j]))


        return a,diag,g,offdiag

def checkcutoff(wf_object, c=list((1e-2,5e-3,1e-3,5e-4, 1e-4, 5e-5, 1e-5, 1e-6))):
    wf_object.compute_energy()
    b =   2*wf_object.get_MO('oovv')*wf_object.t2start
    b -=  np.swapaxes(wf_object.get_MO('oovv'),2,3)*wf_object.t2start
    MP2=np.sum(b,axis=(2,3))
    
    tmp_tau = wf_object.build_tau()
    d =   2*tmp_tau*wf_object.get_MO('oovv')
    d -=  np.swapaxes(wf_object.get_MO('oovv'),2,3)*tmp_tau
    CCSD=np.sum(d,axis=(2,3))
    print ('real CCSD'+str(np.sum(CCSD)))
    
    for i in c:
        test=CCSD.copy()
        c= np.abs(test) < i
        print (np.sum(c))
        test[c]=MP2[c]
        print ('Error for '+str(i)+':'+str(np.sum(test)-np.sum(CCSD)))
        print ('Error for '+str(i)+' in kcal/mol :'+str((np.sum(test)-np.sum(CCSD))*627.509))
        print (1-((np.abs(np.sum(test)-np.sum(CCSD)))/np.abs(np.sum(MP2)-np.sum(CCSD))))
    
    

def GetPairEnergies(Foldername, occ=False, vir=False, cutoff=False, xyz=True, basis='sto-3g', values=4, Triples=False, Miller=False, triples_only=False):
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
            A=HelperCCEnergy(mol, Loc_occ=occ, Loc_vir=vir, Triples=Triples, triples_only=triples_only)
            #This generates MP2 featureset
            Bigmatrix,nada,Bigmatrix2,nada2=GenerateFeatures(A, values=values, Miller=Miller)
            A.compute_energy(r_conv=1e-5, e_conv=1e-8)
            #This generates the CCSD y values
            nada3,Bigamp,nada4,Bigamp2=GenerateFeatures(A, values=values, Miller=Miller)





            if (cutoff != False):
                a=Bigmatrix[:,0]
                #i=(np.abs(a) < cutoff)           
                Bigmatrix = np.delete(Bigmatrix, np.where(np.abs(a) < cutoff), axis=0)
                Bigamp=np.delete(Bigamp, np.where(np.abs(a) < cutoff), axis=0)
                a=Bigmatrix2[:,0]
                Bigmatrix2 = np.delete(Bigmatrix2, np.where(np.abs(a) < cutoff), axis=0)
                Bigamp2=np.delete(Bigamp2, np.where(np.abs(a) < cutoff), axis=0)

            if i==1:
                Bigfeatures=Bigmatrix
                Bigamps=Bigamp
                Bigfeatures2=Bigmatrix2
                Bigamps2=Bigamp2
                i=2
            else:
                Bigfeatures=np.vstack((Bigfeatures,Bigmatrix))
                Bigamps=np.vstack((Bigamps,Bigamp))
                Bigfeatures2=np.vstack((Bigfeatures2,Bigmatrix2))
                Bigamps2=np.vstack((Bigamps2,Bigamp2))


    return Bigfeatures, Bigamps, Bigfeatures2, Bigamps2


def Test(Foldername, occ=False, vir=False, cutoff=False, xyz=True, basis='sto-3g', graph=False, values=4, Triples=False):
    steps=list()
    difference=list()
    supalist=list()
    startenergy=list()
    finalenergy=list()
    filenames=list()
    rhfenergy=list()
    for filename in os.listdir(Foldername):
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
                                  'd_convergence': 1e-8,
                                  'FREEZE_CORE': 'TRUE'
                                  })
            
                A=HelperCCEnergy(mol, Loc_occ=occ, Loc_vir=vir, Triples=Triples)
                X_new_diag,blank,X_new_off,blank2=GenerateFeatures(A, values=values, Miller=self.Miller)
                
                print (np.sum(blank)+ (2*np.sum(blank2)))
                X_new_diag_scaled= scaler_diag.transform(X_new_diag)
                X_new_off_scaled= scaler_off.transform(X_new_off)
                
                
                
                if (cutoff != False):
                    a=X_new_diag[:,0]
                    MP2_diag=a.copy()
                    #i=(np.abs(a) < cutoff)           
                    j=(np.abs(a) > cutoff)

                    #MP2set = np.delete(X_new, np.where(np.abs(a) < cutoff), axis=0)            
                    MLset = np.delete(X_new_diag, np.where(np.abs(a) < cutoff), axis=0)
                    X_new_diag_scaled= scaler_diag.transform(MLset)
                    a[j]=np.squeeze(model_diag.predict(X_new_diag_scaled))
                    
                    b=X_new_off[:,0]
                    MP2_off=b.copy()
                    #i=(np.abs(a) < cutoff)           
                    j=(np.abs(b) > cutoff)

                    #MP2set = np.delete(X_new, np.where(np.abs(a) < cutoff), axis=0)            
                    MLset = np.delete(X_new_off, np.where(np.abs(b) < cutoff), axis=0)
                    X_new_off_scaled= scaler_off.transform(MLset)
                    b[j]=np.squeeze(model_off.predict(X_new_off_scaled))
                    
                    
                    
                    
                    
                else:
                    a=model_diag.predict(X_new_diag_scaled)
                    b=model_off.predict(X_new_off_scaled)
                    
                predict=np.sum(a) +(np.sum(b)*2)                
                
                if (graph == True):
                    #To get pair energies
                    A.compute_energy(r_conv=1e-5, e_conv=1e-8)
                else:
                    A.FinalEnergy=psi4.energy('CCSD(T)')-A.rhf_e
                
                
                

                
                if (graph == True):
                    X_new_diag,blank_diag,X_newoff,blank_off=GenerateFeatures(A, values=values)
                    
                    if (cutoff != False):
                        b=np.delete(a, np.where(np.abs(MP2) < cutoff), axis=0)
                        blank=np.delete(blank, np.where(np.abs(MP2) < cutoff), axis=0)
                        
                        
                        
                    sns.scatterplot(x=np.ravel(b.reshape(-1,1)), y=np.ravel(blank))
                    plt.show()
                    #This shows our ML error based on true CCSD pair energy
                    plt.figure()
                    plt.ylim(-.001,.001)
                    sns.scatterplot(x=np.ravel(blank), y=np.ravel(blank-b.reshape(-1,1)))
                    plt.show()


                    #Let's see MP2 error, so we can show some difference

                    plt.figure()
                    MP2=np.delete(MP2, np.where(np.abs(MP2) < cutoff), axis=0)

                    sns.scatterplot(x=np.ravel(blank), y=np.ravel(blank-MP2.reshape(-1,1)))
                    plt.ylim(-.001,.001)
                    plt.show()
                print ('The ML prediction is' +str(predict))
                rhfenergy.append(A.rhf_e)
                startenergy.append(predict)
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
    return filenames,np.abs(np.asarray(startenergy) - np.asarray(finalenergy))





from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process import obj_func
from sklearn.gaussian_process.kernels import Matern
from helper_ML_pairtools import GetPairEnergies, Test
from sklearn.preprocessing import MinMaxScaler
import os
import time


from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process import obj_func
from sklearn.gaussian_process.kernels import Matern
from helper_ML_pairtools import GetPairEnergies, Test
from sklearn.preprocessing import MinMaxScaler
import os
import time
class pair_energy(object):
    def __init__(self, train_folder, test_folder=False, loc_occ=False, loc_vir=False, Triples=False, cutoff=False, basis='sto-3g', values=6):


        
        self.training_set=train_folder
        self.test_set=test_folder
        self.occ=loc_occ
        self.vir=loc_vir
        self.Triples=Triples
        self.basis=basis
        self.cutoff=cutoff
        self.values=values
        #Default learner
        self.Miller=False
        kernel=Matern(nu=5/2)
        self.learner_diag=GaussianProcessRegressor(kernel=kernel,  random_state=43,
                                              alpha=1e-8, normalize_y=True)
        self.learner_off=GaussianProcessRegressor(kernel=kernel,  random_state=43,
                                              alpha=1e-8, normalize_y=True)


        a=np.array(((0,1e-5)))
        self.y_scaler_diag=MinMaxScaler().fit(a.reshape(-1, 1))
        self.y_scaler_off=MinMaxScaler().fit(a.reshape(-1, 1))


        print ('Pair Energy ML Settings \n')

        print ('Training Set Initialized to path {} \n'.format(self.training_set))

        if (self.test_set != False):
            print ('Test Set Initialized to path {}'.format(self.test_set))


        print ('Basis Set: {}'.format(self.basis))
        print ('Compute CCSD(T): {}'.format(self.Triples))

        if (self.occ == False):
            print ('Occupied Orbital Scheme: Canonical')
        elif (self.occ == 'BOYS'):
            print ('Occupied Orbital Scheme: Boys')
        elif (self.occ == 'PM'):
            print ('Occupied Orbital Scheme: PM')

        if (self.vir == False):
            print ('Virtual Orbital Scheme: Canonical')
        elif (self.vir == 'BOYS'):
            print ('Virtual Orbital Scheme: Boys')
        elif (self.vir == 'PM'):
            print ('Virtual Orbital Scheme: PM')

        print ('\nThe number of excitations for each pair energy considered is {}'.format(self.values))
        print ('Pair energies below {} will be treated at MP2 level'.format(self.cutoff))




    def summary(self):
        print ('Pair Energy ML Settings \n')

        print ('Training Set Initialized to path {} \n'.format(self.training_set))

        if (self.test_set != False):
            print ('Test Set Initialized to path {}'.format(self.test_set))


        print ('Basis Set: {}'.format(self.basis))
        print ('Compute CCSD(T): {}'.format(self.Triples))

        if (self.occ == False):
            print ('Occupied Orbital Scheme: Canonical')
        elif (self.occ == 'BOYS'):
            print ('Occupied Orbital Scheme: Boys')
        elif (self.occ == 'PM'):
            print ('Occupied Orbital Scheme: PM')

        if (self.vir == False):
            print ('Virtual Orbital Scheme: Canonical')
        elif (self.vir == 'BOYS'):
            print ('Virtual Orbital Scheme: Boys')
        elif (self.vir == 'PM'):
            print ('Virtual Orbital Scheme: PM')

        print ('\n The number of excitations for each pair energy considered is {}'.format(self.values))
        print ('Pair energies below {} will be treated at MP2 level'.format(self.cutoff))
        return





    def calculate_train(self, xyz=True):
        if self.Triples == True and (self.occ != False):
            self.X_train_diag,self.y_train_diag,self.X_train_off,self.y_train_off=GetPairEnergies(self.training_set,
                                                                               occ=self.occ,
                                                                               vir=self.vir,
                                                                               basis=self.basis,
                                                                               cutoff=self.cutoff,
                                                                               Triples=False,
                                                                               values=self.values,
                                                                               xyz=xyz,
                                                                               Miller=self.Miller)
            self.X_train_diag_trip,self.y_train_diag_trip,self.X_train_off_trip,self.y_train_off_trip=GetPairEnergies(self.training_set,
                                                                               occ=False,
                                                                               vir=False,
                                                                               basis=self.basis,
                                                                               #cutoff=self.cutoff,
                                                                               Triples=self.Triples,
                                                                               values=self.values,
                                                                               xyz=xyz,
                                                                               Miller=self.Miller,
                                                                               triples_only=True)

        else:
            self.X_train_diag,self.y_train_diag,self.X_train_off,self.y_train_off=GetPairEnergies(self.training_set,
                                                                                   occ=self.occ,
                                                                                   vir=self.vir,
                                                                                   basis=self.basis,
                                                                                   cutoff=self.cutoff,
                                                                                   Triples=self.Triples,
                                                                                   values=self.values,
                                                                                   xyz=xyz,
                                                                                   Miller=self.Miller)
   
        return




    def train_model(self, scale_y=False):
        #Scale diagonal terms
        self.scale_y=scale_y
        self.scaler_diag = MinMaxScaler().fit(self.X_train_diag)
        self.X_train_diag_scaled = self.scaler_diag.transform(self.X_train_diag)
        #Scale off diagonal
        self.scaler_off = MinMaxScaler().fit(self.X_train_off)
        self.X_train_off_scaled = self.scaler_off.transform(self.X_train_off)


        self.model_diag = self.learner_diag
        self.model_off = self.learner_off
        if scale_y == True:
            print ('Scaling y values')
        #    a=np.array(((0,1e-5)))
         #   self.y_scaler_diag=MinMaxScaler().fit(a.reshape(-1, 1))
          #  self.y_scaler_off=MinMaxScaler().fit(a.reshape(-1, 1))
           # self.y_scaler_diag = MinMaxScaler().fit(self.y_train_diag)
            self.y_train_diag_scaled = self.y_scaler_diag.transform(self.y_train_diag.reshape(-1, 1))

           # self.y_scaler_off = MinMaxScaler().fit(self.y_train_off)
            self.y_train_off_scaled = self.y_scaler_off.transform(self.y_train_off.reshape(-1, 1))
           #
            print ('Training diagonal pair energy model. \n')
            start_time=time.time()
            self.model_diag.fit (self.X_train_diag_scaled,
                                self.y_train_diag_scaled)
   
            print ('Training completed in {} seconds'.format(time.time()-start_time))
   
            print ('\n Training off-diagonal pair energy model. \n')
            start_time=time.time()
            self.model_off.fit (self.X_train_off_scaled,
                                self.y_train_off_scaled)
   
            print ('Training completed in {} seconds'.format(time.time()-start_time))
            

#if self.Triples == True and (self.occ != False):
            #     self.scaler_diag=(self.X_train_diag_trip)
            #     self.scaler_off=(self.X_train_diag_trip)
    


        else:
            print ('Training diagonal pair energy model. \n')
            start_time=time.time()
            self.model_diag.fit (self.X_train_diag_scaled,
                                self.y_train_diag)
    
            print ('Training completed in {} seconds'.format(time.time()-start_time))
    
            print ('\n Training off-diagonal pair energy model. \n')
            start_time=time.time()
            self.model_off.fit (self.X_train_off_scaled,
                                self.y_train_off)
    
            print ('Training completed in {} seconds'.format(time.time()-start_time))
            if self.Triples == True and (self.occ != False):
                 self.scaler_diag_trip = MinMaxScaler().fit(self.X_train_diag_trip)
                 self.X_train_diag_scaled_trip = self.scaler_diag_trip.transform(self.X_train_diag_trip)
                 #Scale off diagonal
                 self.scaler_off_trip = MinMaxScaler().fit(self.X_train_off_trip)
                 self.X_train_off_scaled_trip = self.scaler_off_trip.transform(self.X_train_off_trip)


                 from sklearn.base import clone
                 self.model_diag_trip = clone(self.learner_diag)
                 self.model_off_trip = clone(self.learner_off)
                 print ('Training diagonal pair energy model. \n')
                 start_time=time.time()
                 self.model_diag_trip.fit (self.X_train_diag_scaled_trip,
                                     self.y_train_diag_trip)
        
                 print ('Triples Training completed in {} seconds'.format(time.time()-start_time))
        
                 print ('\n Training off-diagonal pair energy model. \n')
                 start_time=time.time()
                 self.model_off_trip.fit (self.X_train_off_scaled_trip,
                                     self.y_train_off_trip)
   
                 print ('Triples Training completed in {} seconds'.format(time.time()-start_time))


            return
    


    def test(self,xyz=True, graph=False):# Miller=self.Miller):#, Miller=self.Miller):
        
        steps=list()
        difference=list()
        supalist=list()
        startenergy=list()
        finalenergy=list()
        filenames=list()
        rhfenergy=list()
        for filename in os.listdir(self.test_set):
            if filename.endswith('.xyz'):
                psi4.core.clean()
                filenames.append(filename)
                print ("filename is "+filename)
                stuff=str(self.test_set+filename)
                text = open(stuff, 'r').read()
                if xyz==True:
                    qmol = psi4.qcdb.Molecule.from_string(text, dtype='xyz')
                    mol = psi4.geometry(qmol.create_psi4_string_from_molecule()+ 'symmetry c1')
                else:
                    mol = psi4.geometry(text)


                psi4.set_options({'basis':        self.basis,
                                  'scf_type':     'pk',
                                  'reference':    'rhf',
                                  'mp2_type':     'conv',
                                  'e_convergence': 1e-8,
                                  'd_convergence': 1e-8,
                                  'FREEZE_CORE': 'TRUE'})

                A=HelperCCEnergy(mol, Loc_occ=self.occ, Loc_vir=self.vir, Triples=self.Triples)
                X_new_diag,blank,X_new_off,blank2=GenerateFeatures(A, values=self.values, Miller=self.Miller)
                
                X_new_diag_scaled= self.scaler_diag.transform(X_new_diag)
                X_new_off_scaled= self.scaler_off.transform(X_new_off)



                if (self.cutoff != False):
                    a=X_new_diag[:,0]
                    MP2_diag=a.copy()
                    #i=(np.abs(a) < cutoff)
                    j=(np.abs(a) > self.cutoff)

                    #MP2set = np.delete(X_new, np.where(np.abs(a) < cutoff), axis=0)
                    MLset = np.delete(X_new_diag, np.where(np.abs(a) < self.cutoff), axis=0)
                    X_new_diag_scaled= self.scaler_diag.transform(MLset)
                    
                    if self.scale_y == True:
                        a[j]=np.squeeze(self.y_scaler_diag.inverse_transform(self.model_diag.predict(X_new_diag_scaled)))

                    else:    
                        a[j]=np.squeeze(self.model_diag.predict(X_new_diag_scaled))
                    
                    b=X_new_off[:,0]
                    MP2_off=b.copy()
                    #i=(np.abs(a) < cutoff)
                    j=(np.abs(b) > self.cutoff)

                    #MP2set = np.delete(X_new, np.where(np.abs(a) < cutoff), axis=0)
                    MLset = np.delete(X_new_off, np.where(np.abs(b) < self.cutoff), axis=0)
                    X_new_off_scaled= self.scaler_off.transform(MLset)
                    if self.scale_y == True:
                        b[j]=np.squeeze(self.y_scaler_off.inverse_transform(self.model_off.predict(X_new_off_scaled)))
                    else:
                        b[j]=np.squeeze(self.model_off.predict(X_new_off_scaled))





                else:
                    if self.scale_y== True:
                        a=(self.y_scaler_diag.inverse_transform(self.model_diag.predict(X_new_diag_scaled).reshape(-1, 1)))
                        b=self.y_scaler_off.inverse_transform(self.model_off.predict(X_new_off_scaled).reshape(-1, 1))


                        #a=(self.y_scaler_diag.inverse_transform(self.model_diag.predict(X_new_diag_scaled)))
                        #b=self.y_scaler_off.inverse_transform(self.model_off.predict(X_new_off_scaled))
                    else:
                        a=self.model_diag.predict(X_new_diag_scaled)
                        b=self.model_off.predict(X_new_off_scaled)

                predict=np.sum(a) +(np.sum(b)*2)


                if self.Triples == True and (self.occ != False):
                        #print ("We here boys: {}".format(self.occ))
                        A=HelperCCEnergy(mol, Triples=True, triples_only=True)
                        X_new_diag_trip,blank,X_new_off_trip,blank2=GenerateFeatures(A, values=self.values, Miller=self.Miller)#, triples_only=True)
                
                        X_new_diag_trip_scaled= self.scaler_diag_trip.transform(X_new_diag_trip)
                        X_new_off_trip_scaled= self.scaler_off_trip.transform(X_new_off_trip)
                        a=self.model_diag_trip.predict(X_new_diag_trip_scaled)
                        b=self.model_off_trip.predict(X_new_off_trip_scaled)
                        predict+=np.sum(a) +(np.sum(b)*2)
                        print ('Triples prediction: {}'.format(np.sum(a) +(np.sum(b)*2)))



                if (graph == True):
                    #To get pair energies
                    A.compute_energy(r_conv=1e-5, e_conv=1e-8)
                else:
                    if self.Triples == True:
                        A.FinalEnergy=psi4.energy('CCSD(T)')-A.rhf_e
                    else:
                        A.FinalEnergy=psi4.energy('CCSD')-A.rhf_e

                    print ('True energy: {}'.format(A.FinalEnergy))



                if (graph == True):
                    X_new_diag,blank_diag,X_newoff,blank_off=GenerateFeatures(A, values=self.values)

                    if (self.cutoff != False):
                        b=np.delete(a, np.where(np.abs(MP2) < self.cutoff), axis=0)
                        blank=np.delete(blank, np.where(np.abs(MP2) < self.cutoff), axis=0)



                    sns.scatterplot(x=np.ravel(b.reshape(-1,1)), y=np.ravel(blank))
                    plt.show()
                    #This shows our ML error based on true CCSD pair energy
                    plt.figure()
                    plt.ylim(-.001,.001)
                    sns.scatterplot(x=np.ravel(blank), y=np.ravel(blank-b.reshape(-1,1)))
                    plt.show()


                    #Let's see MP2 error, so we can show some difference

                    plt.figure()
                    MP2=np.delete(MP2, np.where(np.abs(MP2) < self.cutoff), axis=0)

                    sns.scatterplot(x=np.ravel(blank), y=np.ravel(blank-MP2.reshape(-1,1)))
                    plt.ylim(-.001,.001)
                    plt.show()
                print ('The ML prediction is' +str(predict))
                rhfenergy.append(A.rhf_e)
                startenergy.append(predict)
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
        return filenames,np.abs(np.asarray(startenergy) - np.asarray(finalenergy)), np.asarray(finalenergy), np.asarray(startenergy)



    def bigtest(self,xyz=True, graph=False):# Miller=self.Miller):#, Miller=self.Miller):
        
        steps=list()
        difference=list()
        supalist=list()
        startenergy=list()
        finalenergy=list()
        mp2=list()
        predict1=list()
        predict10=list()
        predict100=list()
        predict1000=list()



        filenames=list()
        rhfenergy=list()
        for filename in os.listdir(self.test_set):
            if filename.endswith('.xyz'):
                psi4.core.clean()
                filenames.append(filename)
                print ("filename is "+filename)
                stuff=str(self.test_set+filename)
                text = open(stuff, 'r').read()
                if xyz==True:
                    qmol = psi4.qcdb.Molecule.from_string(text, dtype='xyz')
                    mol = psi4.geometry(qmol.create_psi4_string_from_molecule()+ 'symmetry c1')
                else:
                    mol = psi4.geometry(text)


                psi4.set_options({'basis':        self.basis,
                                  'scf_type':     'pk',
                                  'reference':    'rhf',
                                  'mp2_type':     'conv',
                                  'e_convergence': 1e-8,
                                  'd_convergence': 1e-8,
                                  'FREEZE_CORE': 'TRUE'})

                A=HelperCCEnergy(mol, Loc_occ=self.occ, Loc_vir=self.vir, Triples=self.Triples)
                X_new_diag,blank,X_new_off,blank2=GenerateFeatures(A, values=self.values, Miller=self.Miller)
                
                #X_new_diag_scaled= self.scaler_diag.transform(X_new_diag)
                #X_new_off_scaled= self.scaler_off.transform(X_new_off)
                mp2.append(np.sum(blank)+ (2*np.sum(blank2)))


                if (self.cutoff != False):
                    a=X_new_diag[:,0]
                    MP2_diag=a.copy()
                    #i=(np.abs(a) < cutoff)
                    j=(np.abs(a) > self.cutoff)
                    b=X_new_off[:,0]
                    k=(np.abs(b) > self.cutoff)
                    #MP2set = np.delete(X_new, np.where(np.abs(a) < cutoff), axis=0)
                    MLset = np.delete(X_new_diag, np.where(np.abs(a) < self.cutoff), axis=0)
                    #X_new_diag_scaled= self.scaler_diag.transform(MLset)
                    
                    if self.scale_y == True:
                        #a[j]=np.squeeze(self.y_scaler_diag.inverse_transform(self.model_diag.predict(X_new_diag_scaled)))

                        X_new_diag = np.delete(X_new_diag, np.where(np.abs(a) < self.cutoff), axis=0)
                        X_new_off = np.delete(X_new_off, np.where(np.abs(b) < self.cutoff), axis=0)
                        X_new_diag_scaled= self.scaler_diag1.transform(X_new_diag)
                        X_new_off_scaled= self.scaler_off1.transform(X_new_off)
                        a[j]=np.squeeze(self.y_scaler_diag.inverse_transform(self.model_diag1.predict(X_new_diag_scaled).reshape(-1, 1)))
                        b[k]=np.squeeze(self.y_scaler_off.inverse_transform(self.model_off1.predict(X_new_off_scaled).reshape(-1, 1)))
                        predict1.append(np.sum(a) +(np.sum(b)*2))

                        X_new_diag_scaled= self.scaler_diag10.transform(X_new_diag)
                        X_new_off_scaled= self.scaler_off10.transform(X_new_off)
                        a[j]=np.squeeze(self.y_scaler_diag.inverse_transform(self.model_diag10.predict(X_new_diag_scaled).reshape(-1, 1)))
                        b[k]=np.squeeze(self.y_scaler_off.inverse_transform(self.model_off10.predict(X_new_off_scaled).reshape(-1, 1)))
                        predict10.append(np.sum(a) +(np.sum(b)*2))

                        X_new_diag_scaled= self.scaler_diag100.transform(X_new_diag)
                        X_new_off_scaled= self.scaler_off100.transform(X_new_off)
                        a[j]=np.squeeze(self.y_scaler_diag.inverse_transform(self.model_diag100.predict(X_new_diag_scaled).reshape(-1, 1)))
                        b[k]=np.squeeze(self.y_scaler_off.inverse_transform(self.model_off100.predict(X_new_off_scaled).reshape(-1, 1)))
                        predict100.append(np.sum(a) +(np.sum(b)*2))

                        X_new_diag_scaled= self.scaler_diag1000.transform(X_new_diag)
                        X_new_off_scaled= self.scaler_off1000.transform(X_new_off)
                        a[j]=np.squeeze(self.y_scaler_diag.inverse_transform(self.model_diag1000.predict(X_new_diag_scaled).reshape(-1, 1)))
                        b[k]=np.squeeze(self.y_scaler_off.inverse_transform(self.model_off1000.predict(X_new_off_scaled).reshape(-1, 1)))
                        predict1000.append(np.sum(a) +(np.sum(b)*2))

                        #a=(self.y_scaler_diag.inverse_transform(self.model_diag.predict(X_new_diag_scaled)))






                    else:    
                        a[j]=np.squeeze(self.model_diag.predict(X_new_diag_scaled))
                    
                   # b=X_new_off[:,0]
                   # MP2_off=b.copy()
                   # #i=(np.abs(a) < cutoff)
                   # j=(np.abs(b) > self.cutoff)

                   # #MP2set = np.delete(X_new, np.where(np.abs(a) < cutoff), axis=0)
                   # MLset = np.delete(X_new_off, np.where(np.abs(b) < self.cutoff), axis=0)
                   # X_new_off_scaled= self.scaler_off.transform(MLset)
                   # if self.scale_y == True:
                   #     b[j]=np.squeeze(self.y_scaler_off.inverse_transform(self.model_off.predict(X_new_off_scaled)))
                   # else:
                   #     b[j]=np.squeeze(self.model_off.predict(X_new_off_scaled))





                else:
                    if self.scale_y== True:
                        X_new_diag_scaled= self.scaler_diag1.transform(X_new_diag)
                        X_new_off_scaled= self.scaler_off1.transform(X_new_off)
                        a=(self.y_scaler_diag.inverse_transform(self.model_diag1.predict(X_new_diag_scaled).reshape(-1, 1)))
                        b=self.y_scaler_off.inverse_transform(self.model_off1.predict(X_new_off_scaled).reshape(-1, 1))
                        predict1.append(np.sum(a) +(np.sum(b)*2))

                        X_new_diag_scaled= self.scaler_diag10.transform(X_new_diag)
                        X_new_off_scaled= self.scaler_off10.transform(X_new_off) 
                        a=(self.y_scaler_diag.inverse_transform(self.model_diag10.predict(X_new_diag_scaled).reshape(-1, 1)))
                        b=self.y_scaler_off.inverse_transform(self.model_off10.predict(X_new_off_scaled).reshape(-1, 1))
                        predict10.append(np.sum(a) +(np.sum(b)*2))

                        X_new_diag_scaled= self.scaler_diag100.transform(X_new_diag)
                        X_new_off_scaled= self.scaler_off100.transform(X_new_off)
                        a=(self.y_scaler_diag.inverse_transform(self.model_diag100.predict(X_new_diag_scaled).reshape(-1, 1)))
                        b=self.y_scaler_off.inverse_transform(self.model_off100.predict(X_new_off_scaled).reshape(-1, 1))
                        predict100.append(np.sum(a) +(np.sum(b)*2))

                        X_new_diag_scaled= self.scaler_diag1000.transform(X_new_diag)
                        X_new_off_scaled= self.scaler_off1000.transform(X_new_off)
                        a=(self.y_scaler_diag.inverse_transform(self.model_diag1000.predict(X_new_diag_scaled).reshape(-1, 1)))
                        b=self.y_scaler_off.inverse_transform(self.model_off1000.predict(X_new_off_scaled).reshape(-1, 1))
                        predict1000.append(np.sum(a) +(np.sum(b)*2))

                        #a=(self.y_scaler_diag.inverse_transform(self.model_diag.predict(X_new_diag_scaled)))
                        #b=self.y_scaler_off.inverse_transform(self.model_off.predict(X_new_off_scaled))
                    else:
                        a=self.model_diag.predict(X_new_diag_scaled)
                        b=self.model_off.predict(X_new_off_scaled)

                #predict=np.sum(a) +(np.sum(b)*2)


                if self.Triples == True and (self.occ != False):
                        #print ("We here boys: {}".format(self.occ))
                        A=HelperCCEnergy(mol, Triples=True, triples_only=True)
                        X_new_diag_trip,blank,X_new_off_trip,blank2=GenerateFeatures(A, values=self.values, Miller=self.Miller)#, triples_only=True)
                
                        X_new_diag_trip_scaled= self.scaler_diag_trip.transform(X_new_diag_trip)
                        X_new_off_trip_scaled= self.scaler_off_trip.transform(X_new_off_trip)
                        a=self.model_diag_trip.predict(X_new_diag_trip_scaled)
                        b=self.model_off_trip.predict(X_new_off_trip_scaled)
                        predict+=np.sum(a) +(np.sum(b)*2)
                        print ('Triples prediction: {}'.format(np.sum(a) +(np.sum(b)*2)))



                if (graph == True):
                    #To get pair energies
                    A.compute_energy(r_conv=1e-5, e_conv=1e-8)
                else:
                    if self.Triples == True:
                        A.FinalEnergy=psi4.energy('CCSD(T)')-A.rhf_e
                    else:
                        A.FinalEnergy=psi4.energy('CCSD')-A.rhf_e

                    print ('True energy: {}'.format(A.FinalEnergy))



                if (graph == True):
                    X_new_diag,blank_diag,X_newoff,blank_off=GenerateFeatures(A, values=self.values)

                    if (self.cutoff != False):
                        b=np.delete(a, np.where(np.abs(MP2) < self.cutoff), axis=0)
                        blank=np.delete(blank, np.where(np.abs(MP2) < self.cutoff), axis=0)



                    sns.scatterplot(x=np.ravel(b.reshape(-1,1)), y=np.ravel(blank))
                    plt.show()
                    #This shows our ML error based on true CCSD pair energy
                    plt.figure()
                    plt.ylim(-.001,.001)
                    sns.scatterplot(x=np.ravel(blank), y=np.ravel(blank-b.reshape(-1,1)))
                    plt.show()


                    #Let's see MP2 error, so we can show some difference

                    plt.figure()
                    MP2=np.delete(MP2, np.where(np.abs(MP2) < self.cutoff), axis=0)

                    sns.scatterplot(x=np.ravel(blank), y=np.ravel(blank-MP2.reshape(-1,1)))
                    plt.ylim(-.001,.001)
                    plt.show()
                print ('ML Predictions:  1:  {}    10:  {}  100:   {}    1000:   {}'.format(str(predict1), str(predict10), str(predict100), str(predict1000)))
                rhfenergy.append(A.rhf_e)
                #startenergy.append(predict)
                finalenergy.append(A.FinalEnergy)

        #difference.append(sum( np.abs(np.asarray(startenergy) - np.asarray(finalenergy))) /len(startenergy))
        #print ('Filenames')
      #  print (filenames)
        #print ('Start Energy')
       # print (np.add(np.array(startenergy),np.array(rhfenergy)))
        #print ('Individual Differences')

        #print (np.abs(np.asarray(startenergy) - np.asarray(finalenergy)))
        #print ('Average Differences')
        #print (difference)
        return filenames,np.asarray(mp2), np.asarray(finalenergy), np.asarray(predict1), np.asarray(predict10),np.asarray(predict100),np.asarray(predict1000)





def checkcutoff(filename, c=list((1e-2,5e-3,1e-3,5e-4, 1e-4, 5e-5, 1e-5, 1e-6)),
                occ=False, vir=False, Triples=False, cutoff=1e-4, basis='sto-3g', values=4, xyz=True):

    psi4.core.clean()
    
    
    stuff=str(filename)
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


    wf_object=HelperCCEnergy(mol, Loc_occ=occ, Loc_vir=vir, Triples=Triples)
    wf_object.compute_energy()
    b =   2*wf_object.get_MO('oovv')*wf_object.t2start
    b -=  np.swapaxes(wf_object.get_MO('oovv'),2,3)*wf_object.t2start
    MP2=np.sum(b,axis=(2,3))
    
    tmp_tau = wf_object.build_tau()
    d =   2*tmp_tau*wf_object.get_MO('oovv')
    d -=  np.swapaxes(wf_object.get_MO('oovv'),2,3)*tmp_tau
    CCSD=np.sum(d,axis=(2,3))
    print ('real CCSD'+str(np.sum(CCSD)))
    print (CCSD.shape[0]**2)
    percent_pairs=list()
    absolute_error=list()
    percent_error=list()
    for i in c:
        test=CCSD.copy()
        
        c= np.abs(test) < i
        test[c]=MP2[c]
        percent_pairs.append((CCSD.shape[0]**2-np.sum(c))/CCSD.shape[0]**2)
        absolute_error.append((np.abs(np.sum(test)-np.sum(CCSD))*627.509))
        percent_error.append(1-((np.abs(np.sum(test)-np.sum(CCSD)))/np.abs(np.sum(MP2)-np.sum(CCSD))))
        print ('Error for '+str(i)+':'+str((np.sum(test)-np.sum(CCSD))*627.509))
        print ('Percent of Pair Energies Used: {}'.format((CCSD.shape[0]**2-np.sum(c))/CCSD.shape[0]**2))
        
        
        print (1-((np.abs(np.sum(test)-np.sum(CCSD)))/np.abs(np.sum(MP2)-np.sum(CCSD))))
    return percent_pairs, percent_error, absolute_error
    

    



def CompareLocalization(filename, xyz=True, savefig=False,basis='sto-3g',c=list((1e-2,5e-3,1e-3,5e-4, 1e-4, 5e-5, 1e-5, 1e-6))):
    can_a,can_b,can_c=checkcutoff(filename,basis=basis, xyz=xyz, c=c)
    boys_a,boys_b,boys_c=checkcutoff(filename,basis=basis, occ='BOYS', vir='BOYS', xyz=xyz, c=c)
    pm_a,pm_b,pm_c=checkcutoff(filename,basis=basis, occ='PM', vir='PM', xyz=xyz, c=c)
    x=c
    df=pd.DataFrame()
    df['Cutoff']=np.array(x)
    df['Percent_Configs']=can_a
    df['Percent_Corr']=can_b
    df['Absolute_err']=can_c
    df['Orbitals'] = df.apply(lambda x: 'Canonical', axis=1)

    df2=pd.DataFrame()
    df2['Cutoff']=np.array(x)
    df2['Percent_Configs']=boys_a
    df2['Percent_Corr']=boys_b
    df2['Absolute_err']=boys_c
    df2['Orbitals'] = df.apply(lambda x: 'Boys', axis=1)

    df3=pd.DataFrame()
    df3['Cutoff']=np.array(x)
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

    fig.set(xlabel='MP2 cutoff value', ylabel='Percent Pair Orbitals')
    plt.tight_layout()
    if savefig == True:
        plt.savefig(filename+'_Config.png', dpi=350)
        print ('File Saved to: '+str(filename+'_cutoff.png'))
    plt.show()
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



