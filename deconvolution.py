import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.utils.data
import matplotlib.pyplot as pl
import zern
import util
from collections import OrderedDict
from tqdm import tqdm
from kornia.filters import median_blur, spatial_gradient
import ptwt
import pywt
try:
    from nvitop import Device
    NVITOP = True
except:
    NVITOP = False

class Deconvolution(nn.Module):
    def __init__(self, config):
        """

        Parameters
        ----------
        npix_apodization : int
            Total number of pixel for apodization (divisible by 2)
        device : str
            Device where to carry out the computations
        batch_size : int
            Batch size
        """
        super().__init__()

        self.config = config          
        
        self.cuda = torch.cuda.is_available()
        if self.config['gpu'] == -1:
            self.cuda = False
        self.device = torch.device(f"cuda:{self.config['gpu']}" if self.cuda else "cpu")        

        if (NVITOP):
            self.handle = Device.all()[self.config['gpu']]            
            print(f"Computing in {self.device} : {self.handle.name()} - mem: {self.handle.memory_used_human()}/{self.handle.memory_total_human()}")
        else:
            print(f"Computing in {self.device}")
        
        # Generate Hamming window function for WFS correlation
        self.npix_apod = self.config['npix_apodization']
        win = np.hanning(self.npix_apod)
        winOut = np.ones(self.config['n_pixel'])
        winOut[0:self.npix_apod//2] = win[0:self.npix_apod//2]
        winOut[-self.npix_apod//2:] = win[-self.npix_apod//2:]
        window = np.outer(winOut, winOut)

        self.pad_width = self.config['pad_width']
        self.n_iter_regularization = self.config['n_iter_regularization']

        # Compute the overfill to properly generate the PSFs from the wavefronts
        self.overfill = util.psf_scale(self.config['wavelength'], 
                                        self.config['diameter'], 
                                        self.config['pix_size'])

        if (self.overfill < 1.0):
            raise Exception(f"The pixel size is not small enough to model a telescope with D={self.telescope_diameter} cm")

        # Compute telescope aperture
        pupil = util.aperture(npix=self.config['n_pixel'], 
                        cent_obs = self.config['central_obs'] / self.config['diameter'], 
                        spider=0, 
                        overfill=self.overfill)
                
        # Define Zernike modes
        print("Computing Zernike modes...")
        Z_machine = zern.ZernikeNaive(mask=[])
        x = np.linspace(-1, 1, self.config['n_pixel'])
        xx, yy = np.meshgrid(x, x)
        rho = self.overfill * np.sqrt(xx ** 2 + yy ** 2)
        theta = np.arctan2(yy, xx)
        aperture_mask = rho <= 1.0

        self.n_modes = len(self.config['psf_modes'])

        basis = np.zeros((self.n_modes, self.config['n_pixel'], self.config['n_pixel']))
        
        # Precompute all Zernike modes except for piston
        for j in range(self.n_modes):
            n, m = zern.zernIndex(j+1)            
            Z = Z_machine.Z_nm(n, m, rho, theta, True, 'Jacobi')
            basis[j,:,:] = Z * aperture_mask
            # basis[j,...] /= np.max(np.abs(basis[j,...]))

        self.pupil = torch.tensor(pupil.astype('float32')).to(self.device)
        self.basis = torch.tensor(basis[0:self.n_modes, :, :].astype('float32')).to(self.device)        
        self.window = torch.tensor(window.astype('float32')).to(self.device)

        print("Computing PSF ...")
        self.psf_modes = torch.tensor(self.config['psf_modes'].astype('float32')).to(self.device)
        self.wavefront, self.psf, self.psf_ft = self.compute_psfs(self.psf_modes)
        
        self.cutoff = self.config['diameter'] / (self.config['wavelength'] * 1e-8) / 206265.0
        freq = np.fft.fftfreq(self.config['n_pixel'], d=self.config['pix_size']) / self.cutoff
        
        xx, yy = np.meshgrid(freq, freq)
        self.rho = np.sqrt(xx ** 2 + yy ** 2)
                
        if self.config['precision'] == 'float16':
            print("Working in float16...")
            self.use_amp = True
        else:
            print("Working in float32...")
            self.use_amp = False

        # Define the scaler for the automatic mixed precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
                     
    def compute_psfs(self, modes):
        """Compute the PSFs and their Fourier transform from a set of modes
        
        Args:
            wavefront_focused ([type]): wavefront of the focused image
            illum ([type]): pupil aperture
            diversity ([type]): diversity for this specific images
        
        """

        # --------------
        # Focused PSF
        # --------------
                
        # Compute wavefronts from estimated modes
        wavefront = torch.einsum('k,klm->lm', modes, self.basis)

        # Compute the complex phase
        phase = self.pupil * torch.exp(1j * wavefront)

        # Compute FFT of the pupil function and compute autocorrelation
        ft = torch.fft.fft2(phase)
        psf = (torch.conj(ft) * ft).real
        
        # Normalize PSF
        psf_norm = psf / torch.sum(psf)

        # FFT of the PSF
        psf_ft = torch.fft.fft2(psf_norm)

        return wavefront, psf_norm, psf_ft

    def lofdahl_scharmer_filter(self, image_ft, psf_ft):
                
        num = torch.abs(psf_ft)**2
        denom = torch.abs(image_ft.detach() * torch.conj(psf_ft))**2
        H = 1.0 - self.mask * self.config['n_pixel']**2 * self.sigma**2 * (num / denom)
                
        H[H > 1.0] = 1.0
        H[H < 0.2] = 0.0

        H = self.mask * median_blur(H[None, None, :, :], (3, 3)).squeeze()
        H = torch.nan_to_num(H)

        H[H < 0.2] = 0.0
        
        return H

    def forward(self, image):
        # Apodize frames and compute FFT
        mean_val = torch.mean(image, dim=(2, 3), keepdim=True)
        image_apod = image - mean_val
        image_apod *= self.window
        image_apod += mean_val
        image_ft = torch.fft.fft2(image)

        # Threshold
        if (self.regularize_fourier == 'scharmer'):
            H = self.lofdahl_scharmer_filter(image_ft, self.psf_ft)
        if (self.regularize_fourier == 'mask'):            
            H = self.mask[None, None, :, :]

        # Convolve estimated image with PSF            
        image_H_ft = H * image_ft
        convolved = torch.fft.ifft2(image_H_ft * self.psf_ft).real

        # Regularization
        grad = spatial_gradient(image, mode='sobel', order=1)

        return convolved, image_H_ft, grad
    
    def wavelet_loss(self, image, wavelet='db5'):
        coefs = ptwt.wavedec2(image, pywt.Wavelet(wavelet), level=4, mode="reflect")
        
        nlev = len(coefs)
        loss = 0.0
        for i in range(nlev-1):
            for j in range(3):
                loss += torch.mean(torch.abs(coefs[i+1][j]))

        return loss

    
    def deconvolve(self, 
                   frames, 
                   regularize_fourier='mask', 
                   diffraction_limit=0.95, 
                   lambda_grad=0.1, 
                   lambda_obj=0.0, 
                   lambda_spectral=0.0, 
                   lambda_continuum=0.0, 
                   lambda_wavelet=0.0, 
                   wavelet='haar'):
        """Deconvolve the image using the PSF

        Args:
            obs (tensor): [L,S,H,W] tensor to be convolved: L is the number of wavelengths, S is the number of Stokes, H and W are the image dimensions            
            regularize_fourier (str, optional): Type of Fourier masking for denoising. Defaults to 'mask'.
            diffraction_limit (float, optional): Diffraction limit in the Fourier masking in units of the diffraction limit. Defaults to 0.95.
            lambda_grad (float, optional): _description_. Defaults to 0.1.
            lambda_obj (float, optional): _description_. Defaults to 0.0.
            lambda_spectral (float, optional): _description_. Defaults to 0.0.
            lambda_continuum (float, optional): _description_. Defaults to 0.0.

        Returns:
            _type_: _description_
        """
        
                        
        # Pad the images and move arrays to tensors
        obs_pad = np.pad(frames, pad_width=((0, 0), (0, 0), (self.pad_width // 2, self.pad_width // 2), (self.pad_width // 2, self.pad_width // 2)), mode='symmetric')        
        obs = torch.tensor(obs_pad.astype('float32')).to(self.device)
        
        # Fourier mask
        mask = self.rho <= diffraction_limit
        self.mask = torch.tensor(mask.astype('float32')).to(self.device)

        lambda_grad = torch.tensor(lambda_grad).to(self.device)
        lambda_obj = torch.tensor(lambda_obj).to(self.device)
        lambda_spectral = torch.tensor(lambda_spectral).to(self.device)
        lambda_continuum = torch.tensor(lambda_continuum).to(self.device)
        lambda_wavelet = torch.tensor(lambda_wavelet).to(self.device)
        
        self.regularize_fourier = regularize_fourier
                            
        image = obs.clone().detach().requires_grad_(True).to(self.device)

        optimizer = torch.optim.AdamW([image], lr=0.1)

        losses = []

        t = tqdm(range(self.config['n_iter']))        
        
        for loop in t:

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
                
                # Get the convolved image using the current estimation and the PSF
                if self.config['checkpointing']:
                    convolved, image_H_ft, grad = torch.utils.checkpoint.checkpoint(self.forward, image, use_reentrant=False)
                else:                    
                    convolved, image_H_ft, grad = self.forward(image)

                # Regularization for the spatial gradient and the object
                if (lambda_grad > 0.0):
                    regul_grad = lambda_grad * torch.mean(grad**2)
                else:
                    regul_grad = torch.tensor(0.0).to(self.device)

                # Regularization for the object to force it to be zero
                if (lambda_obj > 0.0):
                    regul_obj = lambda_obj * torch.mean(image**2)
                else:
                    regul_obj = torch.tensor(0.0).to(self.device)

                # Regularization forcing spectral smoothness
                if (lambda_spectral > 0.0):                
                    grad_spectra = image[1:, ...] - image[:-1, ...]
                    regul_spectra = lambda_spectral * torch.mean(grad_spectra**2)
                else:
                    regul_spectra = torch.tensor(0.0).to(self.device)

                # Regularization forcing continuum to zero (continuum is last wavelength point)
                if (lambda_continuum > 0.0):
                    regul_continuum = lambda_continuum * torch.mean(image[-1, ...]**2)
                else:
                    regul_continuum = torch.tensor(0.0).to(self.device)

                if (lambda_wavelet > 0.0):
                    regul_wavelet = lambda_wavelet * self.wavelet_loss(image, wavelet=wavelet)
                else:
                    regul_wavelet = torch.tensor(0.0).to(self.device)
                
                # Compute the likelihood
                loss_mse = torch.mean( (obs - convolved)**2)

                # Add likelihood and regularization
                loss = loss_mse + regul_grad + regul_obj + regul_spectra + regul_continuum + regul_wavelet

                if loop > self.n_iter_regularization:
                    lambda_grad = 0.0
                    lambda_obj = 0.0
                    lambda_spectral = 0.0
                    lambda_continuum = 0.0
                    lambda_wavelet = 0.0
                                                                
            self.scaler.scale(loss).backward()

            # Update the parameters
            self.scaler.step(optimizer)
            self.scaler.update()
                        
            tmp = OrderedDict()
            if (NVITOP):
                tmp['gpu'] = f'{self.handle.gpu_utilization()}'                
                tmp['mem'] = f' {self.handle.memory_used_human()}/{self.handle.memory_total_human()}'
            tmp['L_mse'] = f'{loss_mse.item():.8f}'
            tmp['R_grad'] = f'{regul_grad.item():.8f}'
            tmp['R_obj'] = f'{regul_obj.item():.8f}'
            tmp['R_spec'] = f'{regul_spectra.item():.8f}'
            tmp['R_cont'] = f'{regul_continuum.item():.8f}'
            tmp['R_wave'] = f'{regul_wavelet.item():.8f}'
            tmp['L'] = f'{loss.item():.8f}'
            t.set_postfix(ordered_dict=tmp)

            losses.append(loss.item())

        losses = np.array(losses)

        # Final result after the optimization takes place
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):                
            # Get the convolved image using the current estimation and the PSF
            if self.config['checkpointing']:
                convolved, image_H_ft, grad = torch.utils.checkpoint.checkpoint(self.forward, image, use_reentrant=False)
            else:                    
                convolved, image_H_ft, grad = self.forward(image)

        # Recover filter image
        image_H = torch.fft.ifft2(image_H_ft).real
        image_H = image_H.detach().cpu().numpy()

        # Return the unfiltered image    
        image = image.detach().cpu().numpy()

        # Crop the padded region
        image = image[:, :, self.pad_width // 2:-self.pad_width // 2, self.pad_width // 2:-self.pad_width // 2]
        image_H = image_H[:, :, self.pad_width // 2:-self.pad_width // 2, self.pad_width // 2:-self.pad_width // 2]

        return image, image_H, losses