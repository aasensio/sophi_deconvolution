import numpy as np
from astropy.io import fits
from deconvolution_mod import Deconvolution
import matplotlib.pyplot as pl

if (__name__ == '__main__'):

    # Zernike coefficients (in radians), starting from Z1 (offset)
    coefs= [0.0,0.0,0.0,
        2.5384410288410995,
        0.10884970414084948,
        0.5778766523678903,
        0.17499516023395728,
        -0.22788919399655982,
        -0.10533576475415085,
        1.7010925728045585,
        1.3308455704245563,
        0.023446694681437074,
        -0.0667308907394535,
        -0.05345526313091036,
        0.03671330897504666,
        -0.05111485547951494,
        0.02619538247956514,
        0.12905269063257063,
        -0.015503522073734656,
        0.16477602560109075,
        -0.07481142465157851]
    
    coefs = np.array(coefs)

    # Demodulation matrix
    dmod50 = np.array([ [ 0.27801745, 0.20265843, 0.23917961, 0.28021289],
    [ 0.40710264, 0.10566463, -0.71883365, 0.20604219],
    [ 0.18965334, 0.53960556, -0.07998936, -0.65140151],
    [-0.57469798, 0.58770792, -0.25784864, 0.24585137] ])
    mod50 = np.linalg.inv(dmod50)

    # Read the data
    f = fits.open('solo_L2_phi-hrt-stokes_20230407T032009_VnoPSF_0344070601.fits')
    stokes = f[0].data[:, :, 512:1024, 512:1024]

    nl, ns, nx, ny = stokes.shape

    # Pad width
    pad_width = 24
    
    # General configuration
    config = {
        'gpu': 0,
        'npix_apodization': 24,
        'n_pixel': 512,
        'n_iter' : 50,
        'n_iter_regularization': 50,
        'wavelength': 6173.341,
        'diameter': 14.0,
        'pix_size': 0.5,
        'central_obs' : 0.0,        
        'psf_modes': coefs,
        'pad_width': pad_width,
        'precision': 'float32',
        'checkpointing': False,
        'modulation_matrix': mod50
    }           

    # Regularization parameters    
    lambda_grad = [0.0, 0.0, 0.0, 0.0]
    lambda_obj = [0.0, 0.0, 0.0, 0.05]
    lambda_spectral = [0.0, 0.0, 0.0, 0.0]
    lambda_continuum = [0.0, 0.0, 0.0, 0.0]
    lambda_wavelet = [0.0, 1.0e-5, 1.0e-5, 1.0e-5]
    wavelet = ['db3', 'db3', 'db3', 'db3']

    # Labels
    which = 3

    if (which == 0):
        ranges = [0.0, 1.0]  
        wavel = 0 
        cmap = 'viridis'        
    if (which == 1):
        ranges = [-0.01, 0.01]
        wavel = 3
        cmap = 'seismic'        
    if (which == 2):
        ranges = [-0.01, 0.01]
        wavel = 3
        cmap = 'seismic'        
    if (which == 3):
        ranges = [-0.05, 0.05]
        wavel = 3
        cmap = 'seismic'        
    
    config['n_pixel'] = nx + pad_width

    # Instantiate the model
    deconvolver = Deconvolution(config)
        
    # Deconvolve the data
    # It returns the deconvolved image, the Fourier filtered image and the loss
    rec, rec_H, loss = deconvolver.deconvolve(stokes,
                                                regularize_fourier='mask', 
                                                diffraction_limit=0.95,
                                                lambda_grad=lambda_grad, 
                                                lambda_obj=lambda_obj,
                                                lambda_spectral=lambda_spectral,
                                                lambda_continuum=lambda_continuum,
                                                lambda_wavelet=lambda_wavelet,
                                                wavelet=wavelet)
    
    # restored = fits.open('restored_reg_0.1_solo_L2_phi-hrt-stokes_20230407T032009_VnoPSF_0344070601.fits')

    # if (which == 0):
    #     label1 = f'Wiener: {100*np.std(restored[0].data[60:180, 1200:, which, wavel])/np.mean(restored[0].data[60:180, 1200:, which, wavel]):.2f}'
    #     label2 = f'Wavelet: {100*np.std(rec[wavel, 0, 70:180, 1200:])/np.mean(rec[wavel, 0, 70:180, 1200:]):.2f}'
    #     label3 = f'Wavelet+mask: {100*np.std(rec_H[wavel, 0, 70:180, 1200:])/np.mean(rec_H[wavel, 0, 70:180, 1200:]):.2f}'
    #     label4 = f'Original: {100*np.std(stokes[wavel, which, 70:180, 1200:])/np.mean(stokes[wavel, which, 70:180, 1200:]):.2f}'
    # else:
    #     label1 = f'{np.std(restored[0].data[60:180, 1200:, which, wavel]):.4f}'
    #     label2 = f'{np.std(rec[3, 0, 70:180, 1200:]):.4f}'
    #     label3 = f'{np.std(rec_H[3, 0, 70:180, 1200:]):.4f}'
    #     label4 = f'{np.std(stokes[3, which, 70:180, 1200:]):.4f}'
    
    

    # # fig, ax = pl.subplots(nrows=3, ncols=4, figsize=(20, 15))
    
    # # im = ax[0, 0].imshow(restored[0].data[70:180, 1200:, which, wavel], cmap=cmap)
    # # pl.colorbar(im, ax=ax[0, 0])
    # # ax[0, 0].set_title(label1)

    # # im = ax[0, 1].imshow(rec[3, 0, 70:180, 1200:], cmap=cmap)
    # # pl.colorbar(im, ax=ax[0, 1])
    # # ax[0, 1].set_title(label2)

    # # im = ax[0, 2].imshow(rec_H[3, 0, 70:180, 1200:], cmap=cmap)
    # # pl.colorbar(im, ax=ax[0, 2])
    # # ax[0, 2].set_title(label3)

    # # im = ax[0, 3].imshow(stokes[3, which, 70:180, 1200:], cmap=cmap)
    # # pl.colorbar(im, ax=ax[0, 3])
    # # ax[0, 3].set_title(label4)
    
    # # im = ax[1, 0].imshow(restored[0].data[300:800, 700:1200, which, wavel], cmap=cmap)
    # # pl.colorbar(im, ax=ax[1, 0])
    
    # # im = ax[1, 1].imshow(rec[wavel, 0, 300:800, 700:1200], cmap=cmap)
    # # pl.colorbar(im, ax=ax[1, 1])
    
    # # im = ax[1, 2].imshow(rec_H[wavel, 0, 300:800, 700:1200], cmap=cmap)
    # # pl.colorbar(im, ax=ax[1, 2])

    # # im = ax[1, 3].imshow(stokes[wavel, which, 300:800, 700:1200], cmap=cmap)
    # # pl.colorbar(im, ax=ax[1, 3])

    # # im = ax[2, 0].imshow(restored[0].data[:, :, which, wavel], cmap=cmap)
    # # pl.colorbar(im, ax=ax[2, 0])
    
    # # im = ax[2, 1].imshow(rec[wavel, 0, :, :], cmap=cmap)
    # # pl.colorbar(im, ax=ax[2, 1])
    
    # # im = ax[2, 2].imshow(rec_H[wavel, 0, :, :], cmap=cmap)
    # # pl.colorbar(im, ax=ax[2, 2])

    # # im = ax[2, 3].imshow(stokes[wavel, which, :, :], cmap=cmap)
    # # pl.colorbar(im, ax=ax[2, 3])

    # import az_average
    # k, power = az_average.power_spectrum(rec[3, 0, :, :])
    # k, power_H = az_average.power_spectrum(rec_H[3, 0, :, :])
    # pl.semilogy(k, power)
    # pl.semilogy(k, power_H)