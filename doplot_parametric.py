import numpy as np
import matplotlib.pyplot as pl
import h5py
from astropy.io import fits

f = fits.open('solo_L2_phi-hrt-stokes_20230407T032009_VnoPSF_0344070601.fits')
weak_obs = f[0].data[3, 3, 420:780, 550:910]
noise_obs = f[0].data[3, 3, 50:300, 300:550]


f = h5py.File('parametric.h5', 'r')

lambda_wavelets = [0.002**2, 0.003**2, 0.004**2, 0.005**2]    
wavelets = ['haar', 'db3', 'db5', 'db9', 'bior1.3', 'bior3.5']

########################
# Noise signals
########################
fig, ax = pl.subplots(nrows=5, ncols=6, figsize=(15, 10), sharex=True, sharey=True)
loop = 0
for i in range(6):
    ax.flat[loop].imshow(noise_obs, cmap='seismic', vmin=-0.05, vmax=0.05)
    loop += 1
    
for lambda_wavelet in lambda_wavelets:
    for wavelet in wavelets:
        ax.flat[loop].imshow(f[f'noise_{wavelet}_{lambda_wavelet}'][:], cmap='seismic', vmin=-0.01, vmax=0.01)        
        loop += 1

for i in range(6):
    ax[1, i].set_title(wavelets[i])

ax[0, 0].set_ylabel('Stokes V')
for i in range(4):
    ax[i+1, 0].set_ylabel(rf'$\lambda_w$={lambda_wavelets[i]:.6f}')
pl.tight_layout()
pl.savefig('noise_signals.png')

########################
# Weak signals
########################
fig, ax = pl.subplots(nrows=5, ncols=6, figsize=(15, 10), sharex=True, sharey=True)
loop = 0
for i in range(6):
    ax.flat[loop].imshow(weak_obs, cmap='seismic', vmin=-0.05, vmax=0.05)
    loop += 1

for lambda_wavelet in lambda_wavelets:
    for wavelet in wavelets:
        ax.flat[loop].imshow(f[f'weak_{wavelet}_{lambda_wavelet}'][:], cmap='seismic', vmin=-0.05, vmax=0.05)
        loop += 1

for i in range(6):
    ax[1, i].set_title(wavelets[i])

ax[0, 0].set_ylabel('Stokes V')
for i in range(4):
    ax[i, 0].set_ylabel(rf'$\lambda_w$={lambda_wavelets[i]:.6f}')
pl.tight_layout()
pl.savefig('weak_signals.png')

########################
# Weak signals (zoom)
########################
fig, ax = pl.subplots(nrows=5, ncols=6, figsize=(15, 10), sharex=True, sharey=True)
loop = 0
for i in range(6):
    ax.flat[loop].imshow(weak_obs[200:240, 140:180], cmap='seismic', vmin=-0.05, vmax=0.05)
    loop += 1

for lambda_wavelet in lambda_wavelets:
    for wavelet in wavelets:
        ax.flat[loop].imshow(f[f'weak_{wavelet}_{lambda_wavelet}'][:][200:240, 140:180], cmap='seismic', vmin=-0.05, vmax=0.05)
        loop += 1

for i in range(6):
    ax[1, i].set_title(wavelets[i])

ax[0, 0].set_ylabel('Stokes V')
for i in range(4):
    ax[i+1, 0].set_ylabel(rf'$\lambda_w$={lambda_wavelets[i]:.6f}')
pl.tight_layout()
pl.savefig('weak_signals_zoom.png')

########################
# Histograms
########################
fig, ax = pl.subplots(nrows=5, ncols=6, figsize=(15, 10), sharex=True, sharey=True)

loop = 0
for i in range(6):
    ax.flat[loop].hist(noise_obs.flatten(), bins=150)
    ax.flat[loop].text(0.1, 0.9, rf'$\mu$={np.mean(noise_obs):.6f}', transform=ax.flat[loop].transAxes)
    ax.flat[loop].text(0.1, 0.8, rf'$\sigma$={np.std(noise_obs):.6f}', transform=ax.flat[loop].transAxes)
    loop += 1

for lambda_wavelet in lambda_wavelets:
    for wavelet in wavelets:
        signal = f[f'noise_{wavelet}_{lambda_wavelet}'][:]
        ax.flat[loop].hist(signal.flatten(), bins=150)
        ax.flat[loop].text(0.1, 0.9, rf'$\mu$={np.mean(signal):.6f}', transform=ax.flat[loop].transAxes)
        ax.flat[loop].text(0.1, 0.8, rf'$\sigma$={np.std(signal):.6f}', transform=ax.flat[loop].transAxes)
        loop += 1
for i in range(6):
    ax[1, i].set_title(wavelets[i])

ax[0, 0].set_ylabel('Stokes V')
for i in range(4):
    ax[i+1, 0].set_ylabel(rf'$\lambda_w$={lambda_wavelets[i]:.6f}')

fig.supxlabel('Stokes V')
pl.tight_layout()
pl.savefig('noise_histograms.png')
