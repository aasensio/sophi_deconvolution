import numpy as np
from astropy.io import fits
from deconvolution import Deconvolution

if __name__ == '__main__':

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

    # Read the data
    f = fits.open('solo_L2_phi-hrt-stokes_20230407T032009_VnoPSF_0344070601.fits')
    stokes = f[0].data

    nl, ns, nx, ny = stokes.shape

    # Pad width
    pad_width = 24 #int(nx*10/(100-10*2))
    
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
    }           

    # Regularization parameters    
    lambda_grad = 0.00
    lambda_obj = 0.000
    lambda_spectral = 0.0
    lambda_continuum = 0.0

    rec_all = []    

    for which in range(4):
        
        if (which == 0):
            lambda_wavelet = 0.00**2
            ranges = [0.0, 1.0]  
            wavel = 0 
            cmap = 'viridis'
            wavelet = 'db3'
        if (which == 1):
            lambda_wavelet = 0.004**2
            ranges = [-0.01, 0.01]
            wavel = 3
            cmap = 'seismic'
            wavelet = 'db3'
        if (which == 2):
            lambda_wavelet = 0.004**2
            ranges = [-0.01, 0.01]
            wavel = 3
            cmap = 'seismic'
            wavelet = 'db3'
        if (which == 3):
            lambda_wavelet = 0.004**2
            ranges = [-0.05, 0.05]
            wavel = 3
            cmap = 'seismic'
            wavelet = 'db3'

        labels = ['stokesI', 'stokesQ', 'stokesU', 'stokesV']    

        print('#######################')
        print(f'Processing {labels[which]}')
        print('#######################')
        # Pad the image            
        frames = stokes[:, which:which+1, :, :]
        
        config['n_pixel'] = nx + pad_width

        # Instantiate the model
        deconvolver = Deconvolution(config)
        
        # Deconvolve the data
        # It returns the deconvolved image, the Fourier filtered image and the loss
        rec, rec_H, loss = deconvolver.deconvolve(frames,                                                 
                                                    regularize_fourier='mask', 
                                                    diffraction_limit=0.90,
                                                    lambda_grad=lambda_grad, 
                                                    lambda_obj=lambda_obj,
                                                    lambda_spectral=lambda_spectral,
                                                    lambda_continuum=lambda_continuum,
                                                    lambda_wavelet=lambda_wavelet,
                                                    wavelet=wavelet)
        
        rec_all.append(rec)

    out = np.concatenate(rec_all, axis=1)    
        
    # Save the results    
    hdu = fits.PrimaryHDU(data=out)
    hdul = fits.HDUList([hdu])
    hdul.writeto('wavelet.fits', overwrite=True)