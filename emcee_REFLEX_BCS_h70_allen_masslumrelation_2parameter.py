"""
Created on Fri Mar 27 16:28:39 2020

@author: arindam
"""
'''
emcee code to calculate the probability distribution of $\Omega_m$ and $\sigma_8$ 
We try to reproduce the results of Allen et al. (2003) (https://ui.adsabs.harvard.edu/abs/2003MNRAS.342..287A/abstract)

For detailed documentation on MCMC and emcee refer to:
    https://prappleizer.github.io/Tutorials/MCMC/MCMC_Tutorial.html
    https://emcee.readthedocs.io/en/stable/
'''

import sys
#change the directory to the folder where cosmology.py is located
sys.path.append("/Users/arindam/Documents/IUCAA_project/code/aum/pwd/install/lib/python3.8/site-packages/")
import cosmology as cc
import numpy as np
import matplotlib.pyplot as plt
import emcee
from scipy.interpolate import interp1d
import time
import corner
            

'''REFLEX'''
#Create logarithmic bins to calculate the observed luminosity function as defined in Allen et al.
 
REFLEX_bins = np.array([11.25+2.19, 11.25-1.19, 16.27+4.61, 16.27-2.83, 29.95+76.2, 29.95-9.07])
REFLEX_bins = np.unique(REFLEX_bins) *(7./5.)**(-2) #in units of h70^-2 1e44 ergs/s 
# The 7/5 factor is to convert from units of h50 to h70
logREFLEX_bins = np.log10(REFLEX_bins)

#Define approximate midpoints of the logarithmic bins as given in Allen et al.
x_REFLEX = np.array([11.25, 16.27, 29.95]) *(7./5.)**(-2) #in units of h70^-2 1e44 ergs/s
n_REFLEX_bins = REFLEX_bins.size-1

REFLEX_data = np.loadtxt('REFLEX_Catalogue_h70_lambda_cosm.txt', delimiter=';')
z_REFLEX= REFLEX_data[:,0]
L_REFLEX= REFLEX_data[:,6] # in units of h_70^-2 1e44 erg/s
 
#select the REFLEX clusters with redshift<=0.3 and luminosities> 5 h_70^-2 1e44 erg/s as done by Allen et al.
idx_REFLEX = (z_REFLEX<=0.3) & (L_REFLEX>5)
z_REFLEX = REFLEX_data[idx_REFLEX,0]
L_REFLEX = REFLEX_data[idx_REFLEX,6]

def get_REFLEX_lumfunc(theta, bins, x, L, z):
    '''Function to calculate the observed luminosity function given the parameteres of MCMC, bins
    and luminosity and redshift data from the REFLEX catalgoue
    '''
    #L should be an array containing all the luminosities of all clusters for which we want to calculate lumfunc
    #z should be the corresponding redshifts
    omegam, sigma8 = theta
    h=0.7 #Hubble parameter/100
    a = cc.cosmology(omegam,0.0,-1.0,0.0,0.0476,h,2.726,sigma8,0.96,np.log10(8.0),1.0) #load the cosmology
    
    def get_lumlimit(z):
        '''
        Parameters
        ----------
        z : redshift.

        Returns
        -------
        lum : The minimum luminosity a cluster must have at redshift z to be detected by the telescope

        '''
        lum = np.zeros(np.size(z))
        for i in range(np.size(z)):
            Flim = 3*1e-15 #in units of W/m^2
            lum[i]= Flim * 4*np.pi * (a.Dlofz(z[i])*3.087e22/h)**2*1e7/1e44 #h_70^-2 1e44 ergs/s
        return lum
    
    #Now calculate the Vmax for each cluster, using the luminosity limit
    z_arr=np.linspace(0.0005,2, n_REFLEX_bins**2)
    lum_limit= get_lumlimit(z_arr)
    spl= interp1d(lum_limit,z_arr)
    
    
    Vmax = L*0.0
    for i in range(L.size):
        zz = np.float64(spl(L[i]))
        Vmax[i] = 4.24/3*(a.Dcofz(zz)/h)**3 #Vmax has units of MPc^3

            
    dellogL = (np.diff(logREFLEX_bins))
    n = np.histogram(L, bins=bins, weights=1/Vmax)[0]
    obs_lumfunc =  n/(x*np.log(10)*dellogL) # Mpc-3 (1e44 ergs/s)^-1
            
    return obs_lumfunc



'''BCS data'''
a_allen = cc.cosmology(0.3,0.0,-1.0,0.0,0.0476,0.7,2.726,0.8,0.96,np.log10(8.0),1.0) #Allen
a_ebeling = cc.cosmology(1.,0.0,-1.0,0.0,0.0476,0.5,2.726,0.8,0.96,np.log10(8.0),1.0) #Ebeling

#Create logarithmic bins to calculate the observed luminosity function as defined in Allen et al.
BCS_bins = np.array([11.73-1.73, 11.73+1.62, 15.65+2.2, 15.65-2.3, 23.91+28.8, 23.91-6.06])
BCS_bins = np.unique(BCS_bins)*(7./5.)**(-2)  #in units of h70^-2 1e44 ergs/s 
# The 7/5 factor is to convert from units of h50 to h70

BCS_data = np.loadtxt('BCS_data.txt', delimiter=';')
z_BCS = BCS_data[:,5]
L_BCS = BCS_data[:,7]*(7./5.)**(-2) #in units of h_70^-2 1e44 ergs/s

Lmin_BCS = 5.1
idx_BCS = (z_BCS<=0.3) #& (L_BCS>Lmin_BCS)

L_BCS = BCS_data[idx_BCS,7]
z_BCS = BCS_data[idx_BCS, 5]
for i in range(L_BCS.size):
    L_BCS[i] = L_BCS[i] * (a_allen.Dlofz(z_BCS[i])/a_ebeling.Dlofz(z_BCS[i])*(5./7.))**2

'''extended BCS or eBCS'''
eBCS_data = np.loadtxt('eBCS_data.txt', delimiter=';')
#RA; DEC; nH20; T; CRV; RAD; kT; z; Fx; Lx 
z_eBCS = eBCS_data[:,7]
L_eBCS = eBCS_data[:,9]*(7./5.)**(-2) #in units of h_70^-2 1e44 ergs/s
idx_eBCS = (z_eBCS<=0.3) #& (L_eBCS>Lmin_BCS)

L_eBCS = eBCS_data[idx_eBCS,9]
z_eBCS = eBCS_data[idx_eBCS, 7]
for i in range(L_eBCS.size):
    #To convert the luminosites calculated in Ebeling et al's cosmology to Allen et al.'s cosmology
    L_eBCS[i] = L_eBCS[i] * (a_allen.Dlofz(z_eBCS[i])/a_ebeling.Dlofz(z_eBCS[i])*(5./7.))**2 

#Combine the BCS and eBCS data
L_bothBCS = np.concatenate([L_eBCS, L_BCS])
z_bothBCS = np.concatenate([z_eBCS, z_BCS])


#Now lets get the mean values of each lum. bin
logBCS_bins = np.log10(BCS_bins)
x_BCS = np.array([11.73, 15.65, 23.91]) *(7./5.)**(-2) #in units of h70^-2 1e44 ergs/s
x = np.concatenate([x_BCS, x_REFLEX])

def get_BCS_lumfunc(theta, bins, x, L, z):
    '''Function to calculate the observed luminosity function given the parameteres of MCMC, bins
    and luminosity and redshift data from the REFLEX catalgoue
    '''
    #L should be an array containing all the luminosities of all clusters for which we want to calculate lumfunc
    #z should be the corresponding redshifts
    omegam, sigma8 = theta
    h=0.7
    a = cc.cosmology(omegam,0.0,-1.0,0.0,0.0476,h,2.726,sigma8,0.96,np.log10(8.0),1.0)
 
    def get_lumlimit(z):
        '''
        Parameters
        ----------
        z : redshift.

        Returns
        -------
        lum : The minimum luminosity a cluster must have at redshift z to be detected by the telescope

        '''
        lum=np.zeros(np.size(z))
        for i in range(np.size(z)):
            Flim = 2.8*1e-15 #in units of W/m^2
            lum[i]= Flim*4*np.pi*(a.Dlofz(z[i])*3.087e22/h)**2*1e7/1e44
        return lum

        
    z_arr=np.linspace(0.0005,0.7, 40*40)
    lum_limit= get_lumlimit(z_arr)
    spl= interp1d(lum_limit,z_arr)
   
    Vmax = L*0.0
    for i in range(L.size):
        zmax = 0.3
        maxVmax = (4.14)/3*(a.Dcofz(zmax)/h)**3
        
        zz = np.float64(spl(L[i]))
        Vmax[i] = (4.14)/3*(a.Dcofz(zz)/h)**3 #Vmax has units of MPc^3
        
        if Vmax[i]>maxVmax:
            Vmax[i] = maxVmax

      
    dellogL = np.diff(logBCS_bins)
    n = np.histogram(L, bins=bins, weights=1/Vmax)[0]
    obs_lumfunc =  n/(x*np.log(10)*dellogL) # Mpc-3 (1e44 ergs/s)^-1

    return obs_lumfunc

def get_model_without_scatter(theta, x, z):
    '''
    Get the model of the luminosity function assuming zero scatter over the mass-luminosity relationship
    
    Parameters
    ----------
    theta : cosmological parameters
    x : approx midpoint of the luminosity bins
    z : mean redshift of the clusters as given in Allen et al.
    p : slope of the mass-luminosity relation
    logM0 : intercept of the mass-luminosity relation
    
    Returns
    -------
    phi_L : Model of the luminosity function
    '''
    omegam, sigma8 = theta
    p = 0.76 
    logM0 = 14.29 -np.log10(2)
    h=0.7
    a = cc.cosmology(omegam,0.0,-1.0,0.0,0.0476,h,2.726,sigma8,0.96,np.log10(8.0),1.0) #load cosmology
 
    E = a.Eofz(z)

    M = 10**(logM0 + p*(np.log10(x/E)) + 2*p*np.log10(7./5.) - np.log10(E))
    
    phi_L = np.zeros(x.size)
    for i in range(x.size):
        #We obtain the luminosity function from the Jenkins mass function
        phi_L[i] = a.MF_Jenkins(M[i],z)* M[i] * p /x[i] * h**3 
    
    return phi_L


#–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––       


def ln_likelihood(theta, x, obs_lumfunc, yerr):
    '''
    Parameters
    ----------
    theta : cosmological parameters
    x : approx midpoint of the luminosity bins
    obs_lumfunc : The observed luminosity function combined from both REFLEX and BCS data
    yerr : Error in the observed luminosity function. We use the numbers given in Allen et al.
    
    Returns
    -------
    The log likelihood

    '''

    REFLEX_lumfunc = get_REFLEX_lumfunc(theta, REFLEX_bins, x_REFLEX, L=L_REFLEX, z=z_REFLEX)
    bothBCS_lumfunc = get_BCS_lumfunc(theta, BCS_bins, x_BCS, L_bothBCS, z_bothBCS)
    obs_lumfunc = np.concatenate([bothBCS_lumfunc, REFLEX_lumfunc])
    
    model = get_model_without_scatter(theta, x, 0.21)

    idx = (yerr>0.0)
    num = obs_lumfunc[idx]-model[idx]
    lchisq = np.sum((num/yerr[idx])**2.)
    return -0.5*lchisq

            
       
def ln_prior(theta):
    #Returns the prior to be multiplied with the likelihood given the cosmological parameters
    omegam, sigma8 = theta
    if (0.05<omegam<1. and 0.2<sigma8<1.5):
        return 0. 
    else:
        return -np.inf

def ln_prob(theta, x, obs_lumfunc, yerr):
    '''
    Calculate the posterior probability distribution of the cosmological parameters at each MCMC iteration.
    Prob ~ likelihood * prior

    Parameters
    ----------
    theta : cosmological parameters
     x : approx midpoint of the luminosity bins
    obs_lumfunc : The observed luminosity function combined from both REFLEX and BCS data
    yerr : Error in the observed luminosity function. We use the numbers given in Allen et al.
   
    Returns
    -------
    Posterior probability distribution of the cosmological parameters at each MCMC iteration
    '''
    lp = ln_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    
    totlike = lp + ln_likelihood(theta, x, obs_lumfunc, yerr)
    return totlike

if __name__ == "__main__":

    # Initialise walkers for emcee
    nwalkers = 20 #No of walkers to be used for emcee
    np.random.seed(1)
    
    #pos0 is the starting position of the random walkers
    initial = np.array([0.4,0.8])
    ndim = len(initial)
    pos0 = initial + 1e-2*np.random.randn(nwalkers, ndim)
    
    
    truth = np.array([0.3,0.7]) 
    
    #data
    REFLEX_lumfunc = get_REFLEX_lumfunc(initial, REFLEX_bins, x_REFLEX, L_REFLEX, z_REFLEX)
    bothBCS_lumfunc = get_BCS_lumfunc(initial, BCS_bins, x_BCS, L_bothBCS, z_bothBCS)
    obs_lumfunc = np.concatenate([bothBCS_lumfunc, REFLEX_lumfunc])
    
    # model, the (7/5) factor is to convert from units of h50 to h70
    Lx_arr_h50 = np.logspace(np.log10(11),np.log10(30),100)
    Lx_arr_h70 = Lx_arr_h50 * (7./5.)**(-2)
    
    BCS_model_without_scatter = get_model_without_scatter(truth, x_BCS, 0.21)
    REFLEX_model_without_scatter = get_model_without_scatter(truth, x_REFLEX, 0.085)
    model_without_scatter = np.concatenate([BCS_model_without_scatter, REFLEX_model_without_scatter])
    
    #Allen et al. data which we are trying to reproduce
    
    #Allen et al. approx midpoint of the luminosity bins
    L_allen_h50 = np.array([11.73, 15.65, 23.91, 11.25, 16.27, 29.95]) #in units of h50^-2 1e44 ergs/s
    L_allen_h70 = L_allen_h50 * (7./5.)**(-2) #in units of h70^-2 1e44 ergs/s
    # the (7/5) factor is to convert from units of h50 to h70
    
    #Allen et al. luminosity function
    phi_allen_h50 = np.array([1.32, 0.745, 0.081, 1.56, 0.444, 0.0177]) * 1e-9 #h50^5 MPc-3 1e-44 ergs^-1 s
    phi_allen_h70 = phi_allen_h50 * (7./5.)**5 #h70^2 Mpc-3 1e-44 ergs^-1 s
    
    err_allen_h50 = np.array([0.32, 0.181, 0.0197, 0.34, 0.098, 0.0039]) * 1e-9 #h50^5 MPc-3 1e-44 ergs^-1 s
    err_allen_h70 = err_allen_h50 * (7./5.)**5 #h70^2 Mpc-3 1e-44 ergs^-1 s
    
    #Plotting
    fig1,ax1 = plt.subplots()
    ax1.errorbar(L_allen_h70, phi_allen_h70, yerr=err_allen_h70, fmt='.', label='Allen et al. (2003)')
    ax1.plot(Lx_arr_h70, get_model_without_scatter(truth, Lx_arr_h70, 0.21), label='model without scatter', c='green')
    # ax1.scatter(x_REFLEX, REFLEX_lumfunc, label='REFLEX data (Bohringer et al. (2002))', s=20, c='red')
    # ax1.scatter(x_BCS, bothBCS_lumfunc, label ='extended BCS data (Ebeling et. al (2000))', s=20, c='orange')
    ax1.scatter(x, np.concatenate([bothBCS_lumfunc, REFLEX_lumfunc]), s=20, label = 'REFLEX+BCS lumfunc', c='orange')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('L ($h_{70}^{-2}\ 10^{44}$ erg/s)')
    ax1.set_ylabel('$\phi (L)\ (h_{70}^5$ Mpc$^{-3} (10^{44}$ ergs/s)$^{-1}$)')
    ax1.legend()
    plt.savefig('plot_lumfunc_final.png')
    plt.show()
    
    #------------------------------------------------------------------------------------------------------------------
    '''Run emcee'''
    
    yerr = err_allen_h70 #Error bars on the luminosity function as given in Allen et al. to be used for emcee
    
    from multiprocessing import Pool 
    with Pool() as pool:    
        t0=time.time()
        sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_prob, args=(x, obs_lumfunc, yerr), pool=pool)
        pos, prob, state = sampler.run_mcmc(pos0, 10**4, progress=True) #The middle argument decides the no. of MCMC iterations
        
    t1=time.time()
    total_time = t1-t0
    print('Runtime in min =', total_time/60.) 
    
    #------------------------------------------------------------------------------------------------------------------

    '''Displaying results'''
    
    samples = sampler.get_chain()
    
    #Chains
    fig, axes = plt.subplots(2, figsize=(15, 7), sharex=True)
    labels = ["$\Omega_m$","$\sigma_8$"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    
    axes[-1].set_xlabel("step number");
    
    
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    fig1= corner.corner(flat_samples, labels=labels, levels= (0.68,0.95))
    plt.savefig('corner_emcee.png')
    np.save('emcee_sample_chains_2par', samples)
    np.savetxt('emcee_flatsamples_2par.txt', flat_samples)
    
    #mean values of cosmological parameters with uncertainities
    from IPython.display import display, Math
    labels2 = ["\Omega_m","\sigma_8"]
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], labels2[i])
        display(Math(txt))
