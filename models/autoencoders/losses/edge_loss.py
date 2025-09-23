
import cv2
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg
from torchvision.transforms import Grayscale, GaussianBlur
from skimage.morphology import remove_small_objects
from monai.networks.layers import GaussianFilter
from kornia.enhance import equalize_clahe

class EdgeDetector(nn.Module):
    def __init__(self,k_smoothing=3,sigma_spatial=0.6,sigma_color=1.0, kernel_size=3,mode='green',sigmas=[0.8,0.9,1.0,1.1],
                 alpha=1/3,beta=1/2,sigmoid_scale=1.0,low_threshold=None,high_threshold=0.05,max_iter=10,
                 clahe=False,
                 threshold=False,
                 visualise_steps=False):
        super().__init__()
        
        self.kernel_size = kernel_size

        # Meijering filter parameters
        self.visualise_steps = visualise_steps
        self.clahe = clahe
        self.threshold = threshold
        self.mode = mode
        self.sigmas = sigmas
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.sigmoid_scale = sigmoid_scale
        self.low_threshold = high_threshold if low_threshold is None else low_threshold 
        self.high_threshold = high_threshold
        self.k_smoothing = k_smoothing
        self.sigma_spatial = sigma_spatial
        self.sigma_color = sigma_color

    def trim_channels(self, img):
        """
        Trims channels of the input image tensor based on the specified mode,
        treating the third-last dimension as the channel dimension.
        The output tensor retains the same dimensions as the input.

        Parameters:
            img (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The modified image tensor.
        """
        channel_dim = -3  # Third last dimension is treated as the channel dimension.
        if self.mode == 'greyscale':
            img = Grayscale()(img)  # Convert to greyscale
        elif self.mode == 'green':
            img = img.select(channel_dim, 1).unsqueeze(channel_dim)
        elif self.mode == 'red':
            img = img.select(channel_dim, 0).unsqueeze(channel_dim)
        elif self.mode == 'blue':
            img = img.select(channel_dim, 2).unsqueeze(channel_dim)
        else:
            print('Invalid mode specified')
            sys.exit()

        return img
    
    def _hessian_matrix(self,image, sigma):
        """Compute the Hessian matrix of an image."""

        ss_image = GaussianFilter(spatial_dims=2,sigma=sigma,requires_grad=True,approx='scalespace')(image)

        kernel_xx, kernel_yy, kernel_xy = get_derivative_kernels(image.device,image.dtype)

        Gxx = F.conv2d(ss_image, kernel_xx, padding='same')
        Gyy = F.conv2d(ss_image, kernel_yy, padding='same')
        Gxy = F.conv2d(ss_image, kernel_xy, padding='same')
        return (Gxx, Gxy, Gyy)

    def _compute_eigenvalues(self, hessian):
        """Compute eigenvalues from the upper-diagonal entries of a symmetric
        matrix.

        Parameters
        ----------
        S_elems : list of torch.Tensor
            The upper-diagonal elements of the matrix, as returned by
            `hessian_matrix` or `structure_tensor`.

        Returns
        -------
        eigs : torch.Tensor
            The eigenvalues of the matrix, in decreasing order. The eigenvalues are
            the leading dimension. That is, ``eigs[i, j, k]`` contains the
            ith-largest eigenvalue at position (j, k).
        """
        assert len(hessian) == 3
        Gxx, Gxy, Gyy = hessian  # Unpack the Hessian components

        # Fast explicit formulas for 2D.
        eigs = (Gxx + Gyy) / 2
        hsqrtdet = torch.sqrt(Gxy**2 + ((Gxx - Gyy) / 2) ** 2)
        eig1 = eigs + hsqrtdet
        eig2 = eigs - hsqrtdet
        eigs_out = torch.stack([eig1, eig2], dim=0)

        return eigs_out

    def frangi(self,image):
        """
        Differentiable Frangi vesselness filter in PyTorch.

        Args:
            image (torch.Tensor): Single-channel input image of shape (1, H, W) or (1, D, H, W).
            sigmas (list of float): List of scales (sigma values) to use for filtering.
            alpha (float): Alpha parameter for plate sensitivity.
            beta (float): Beta parameter for blobness sensitivity.
            gamma (float or None): Gamma parameter for structuredness sensitivity. If None, it is calculated from the data.

        Returns:
            torch.Tensor: Filtered image with the same shape as the input image.
        """
        assert image.shape[-3] == 1, "Input image must be single-channel"

        filtered_max = torch.zeros_like(image).to(image.device)
        self.gamma = None 

        for sigma in self.sigmas:
            # Compute Hessian matrix and eigenvalues
            hessian = self._hessian_matrix(image, sigma)
            eigvals = self._compute_eigenvalues(hessian)

            # Sort eigenvalues by magnitude
            eigvals, _ = torch.sort(torch.abs(eigvals), dim=0)
            lambda1 = eigvals[0]

            lambda2 = torch.clamp(eigvals[1], min=1e-10)
            r_b = torch.abs(lambda1) / lambda2  # Blobness sensitivity (2D case)

            s = torch.sqrt(torch.sum(eigvals**2, dim=0))  # Structuredness

            # Compute gamma dynamically if not provided
            if self.gamma is None:
                self.gamma = s.max() / 2
                if self.gamma == 0:
                    self.gamma = 1

            # Compute the Frangi filter response
            #plate_sensitivity = 1.0 - torch.exp(-r_a**2 / (2 * self.alpha**2)) # NOTE: SKIPPED AS 1.0 FOR 2D CASE
            blobness = torch.exp(-r_b**2 / (2 * self.beta**2))  # Blobness term
            structuredness = 1.0 - torch.exp(-s**2 / (2 * self.gamma**2))  # Structuredness term
            vals = blobness * structuredness

            filtered_max = torch.max(filtered_max, vals)  # Take pixel-wise maximum over all sigmas

        return filtered_max

    def meijering(self, image):
        """Apply Meijering filter to enhance tubular structures in the image.

        Parameters
        ----------
        image : torch.Tensor
            Input single-channel image tensor.

        Returns
        -------
        filtered_max : torch.Tensor
            Filtered image tensor with enhanced tubular structures.
        """
        assert image.shape[-3] == 1, "Input image must be single-channel"
        ndim = 2  # Number of dimensions

        # Create the circulant matrix for eigenvalue combination
        mtx = torch.tensor(
            linalg.circulant([1] + [self.alpha] * (ndim - 1)), dtype=image.dtype,
        ).to(image.device)

        # Initialize tensor to store the maximum filtered values
        filtered_max = torch.zeros_like(image).to(image.device)

        for sigma in self.sigmas:
            # Compute Hessian matrix and eigenvalues
            hessian = self._hessian_matrix(image, sigma)
            eigvals = self._compute_eigenvalues(hessian)

            # Compute normalized eigenvalues: l_i = e_i + sum_{j!=i} alpha * e_j
            vals = torch.tensordot(mtx, eigvals, dims=1)  # Shape: (ndim, ...)

            # Get the largest normalized eigenvalue (by magnitude) at each pixel
            max_indices = vals.abs().argmax(0, keepdim=True)
            vals_selected = vals.gather(0, max_indices).squeeze(0)

            # Remove negative values and normalize to max=1
            vals_clamped = torch.clamp(vals_selected, min=0)
            max_val = vals_clamped.max()
            if max_val > 0:
                vals_clamped = vals_clamped / max_val

            # Update the maximum filtered values across all scales
            filtered_max = torch.maximum(filtered_max, vals_clamped)

        return filtered_max

    def double_thresholding(self, image):
        image = image * 255.0

        strong_edges = torch.sigmoid(self.sigmoid_scale * (image - self.high_threshold))
        strong_edges = self.soft_binary(strong_edges)
        weak_edges = torch.sigmoid(self.sigmoid_scale * (image - self.low_threshold)) * (1.0 - strong_edges)
        weak_edges = self.soft_binary(weak_edges)

        return strong_edges, weak_edges
    

    def hysteresis_thresholding(self, strong_edges, weak_edges):
        hysteresis_kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=strong_edges.device)
        for _ in range(self.max_iter):
            non_zero_neighbours = torch.sigmoid(self.sigmoid_scale * (F.conv2d(strong_edges, hysteresis_kernel, padding=1) - 0.5))
            edges = (torch.sigmoid(self.sigmoid_scale * (weak_edges - 0.5)) * non_zero_neighbours).float() + strong_edges
            weak_edges = weak_edges * (1.0 - non_zero_neighbours.float())
            strong_edges = edges
            if weak_edges.max() <= 1e-6:
                break
        return edges

    def color_patches(self,image,min_neighbours=4.0,eps=0.1):
        hysteresis_kernel = circular_kernel(1) + eps # adding 0.1 to enable almost-hard thresholding with sigmoids
        hysteresis_kernel[1,1]  = min_neighbours + eps # make sure that the center pixel stays its color if not zero
        weight = torch.tensor(hysteresis_kernel, device=image.device).unsqueeze(0).unsqueeze(0)
        non_zero_neighbours = F.conv2d(image, weight, bias=None, stride=1, padding=1)
        edges = torch.sigmoid(non_zero_neighbours-min_neighbours)
        edges = self.soft_binary(edges)
        return edges

    def soft_binary(self,image):
        return torch.sigmoid(1e3*(image-0.5))

    def hysteresis(self,image,min_neighbours=2.0,eps=0.1,k=5):

        edges = torch.zeros_like(image).to(image.device)
        for _ in range(self.max_iter):
            for hysteresis_kernel in [np.ones((k,k))+eps,horizontal_kernel(k,eps),vertical_kernel(k,eps),diagonal_kernel(k,eps),antidiagonal_kernel(k,eps)]:
                hysteresis_kernel[k//2,k//2]  = min_neighbours + eps # make sure that the center pixel stays its color if not zero
                weight = torch.tensor(hysteresis_kernel, device=image.device).unsqueeze(0).unsqueeze(0)
                non_zero_neighbours = F.conv2d(image, weight, bias=None, stride=1, padding='same') # if at least min_neighbour 
                edges = torch.sigmoid(non_zero_neighbours-min_neighbours) # make an edge if it was strong edge or if weak edge & connected to min_neighbours strong edges
                edges = self.soft_binary(edges)

        return edges

    def scale_filtered(self,filtered, threshold=1e-3, neg_scale=20000, pos_scale=2000):
        """
        Differentiable processing of raw vesselness scores to create a probability map.
        
        Parameters:
            filtered (torch.Tensor): Raw vesselness scores (output from Frangi-Net).
            threshold (float): The threshold `t`.
            neg_scale (float): Scaling factor for negative values.
            pos_scale (float): Scaling factor for positive values.
            
        Returns:
            torch.Tensor: The probability map after processing.
        """
        # Step 1: Subtract the threshold
        adjusted = filtered - threshold

        # Step 2: Smooth asymmetric scaling using a combination of positive and negative scales
        scaled = adjusted * (neg_scale * (adjusted < 0).float() + pos_scale * (adjusted >= 0).float())

        # Step 3: Apply the sigmoid function
        probability_map = torch.sigmoid(scaled)

        return probability_map

    def connect_disjoint_edges(self,image, threshold=50):
        # Create a copy of the image to draw the lines
        image = np.uint8(image)

        lines = cv2.HoughLinesP(image, 1, np.pi / 180, threshold, minLineLength=10, maxLineGap=5)
        result = image.copy()

        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Draw each detected line on the image
            cv2.line(result, (x1, y1), (x2, y2), (255, 0, 0), 2)

        return result

    def _ensure_dim(self,img):
        if img.shape[-3] !=1:
            img = self.trim_channels(img)
        if img.dim() == 3:
            img = img.unsqueeze(0)
        elif img.dim() == 4 and img.shape[-3] == 1:
            pass
        else:
            print('Invalid input shape')
            sys.exit()
        return img
    
    def forward(self, img):
        
        img = self._ensure_dim(img)

        out = []

        if self.clahe:
            clahe = equalize_clahe(img,grid_size=(15,15),slow_and_differentiable=False)
            out.append((clahe,'CLAHE'))
            img = clahe
        
        smoothed = GaussianBlur(kernel_size=self.k_smoothing,sigma=self.sigma_spatial)(img)
        out.append((smoothed,'Smoothed Image'))

        #[zeta,sigma_r] = logClassifier(img,0.3,[0.8,0.9,1.0,1.1],)
        #plt.imshow(zeta.squeeze().detach().cpu().numpy(),cmap='gray')
        #plt.savefig('zeta.png')
        #plt.close()

        #smoothed = fastABF(img,rho=0.3,sigma_r=sigma_r,theta=img+zeta,N=3)
        #out.append((smoothed,'Smoothed Image'))

        filtered = self.meijering(smoothed)

        out.append((filtered,'Filtered Image'))

        scaled = self.scale_filtered(filtered,threshold=self.high_threshold)
        out.append((scaled,'Scaled Image'))

        #connected = self.connect_disjoint_edges(scaled.squeeze().detach().cpu().numpy())
        #out.append((connected,'Connected Edges'))


        # ## DOUBLE THRESHOLDING

        if self.threshold:

            strong_edges, weak_edges = self.double_thresholding(filtered)
            out.append((weak_edges,'Weak Edges'))
            out.append((strong_edges,'Strong Edges'))

            hysteresis = self.hysteresis_thresholding(strong_edges,weak_edges)
            out.append((hysteresis,'Hysteresis'))

            final_scaled = torch.clamp(scaled,0,1)
            out.append((final_scaled,'Final Edges Scaled'))

            binary_mask = (final_scaled.squeeze() > 0.5).detach().cpu().numpy()
            removed_small = remove_small_objects(binary_mask, min_size=64)
            removed_small = torch.tensor(removed_small).unsqueeze(0).unsqueeze(0).to(final_scaled.device)
            removed_small = removed_small.float()

            last = removed_small
        else:
            last = filtered

        if self.visualise_steps:
            return out 
        else:
            return last 


def get_derivative_kernels(device,dtype):
    """
    Defines a (minimal) mask for discrete approximation of the 2nd order derivative in xx,yy,xy directions.

    Source: Discrete Scale Space and Scale-Space Derivative Toolbox for Python
    Author: Copyright (2023) Tony Lindeberg
    
    """
    kernel_xx = np.array([[1, -2, 1]])  # Second derivative in x direction
    kernel_yy = np.array([[1], [-2], [1]])  # Second derivative in y direction
    kernel_xy = np.array([[-1/4, 0, 1/4], 
                        [0, 0, 0], 
                        [1/4, 0, -1/4]])  # Mixed derivative

    # Convert to PyTorch tensors and reshape for convolution
    kernel_xx = torch.tensor(kernel_xx,dtype=dtype).unsqueeze(0).unsqueeze(0).to(device)  # Shape: (1, 1, 1, 3)
    kernel_yy = torch.tensor(kernel_yy,dtype=dtype).unsqueeze(0).unsqueeze(0).to(device)  # Shape: (1, 1, 3, 1)
    kernel_xy = torch.tensor(kernel_xy,dtype=dtype).unsqueeze(0).unsqueeze(0).to(device)  # Shape: (1, 1, 3, 3)

    return kernel_xx, kernel_yy, kernel_xy

def horizontal_kernel(kernel_size=3,eps=0.1):
    # Initialize a kernel filled with zeros
    kernel = np.zeros((kernel_size, kernel_size))
    
    # Set the middle row to ones to create a straight horizontal line at 0 degrees
    kernel[kernel_size // 2, :] = 1 + eps
    
    return kernel

def vertical_kernel(kernel_size=3,eps=0.1):
    # Initialize a kernel filled with zeros
    kernel = np.zeros((kernel_size, kernel_size))
    
    # Set the middle column to ones to create a straight vertical line at 0 degrees
    kernel[:, kernel_size // 2] = 1 + eps
    
    return kernel

def antidiagonal_kernel(kernel_size=3,eps=0.1):
    # Initialize a kernel filled with zeros
    kernel = np.zeros((kernel_size, kernel_size))
    
    # Set the anti-diagonal to ones to create a straight anti-diagonal line at 135 degrees
    np.fill_diagonal(np.fliplr(kernel), 1 + eps)
    
    return kernel

def diagonal_kernel(kernel_size=3,eps=0.1):
    # Initialize a kernel filled with zeros
    kernel = np.zeros((kernel_size, kernel_size))
    
    # Set the diagonal to ones to create a straight diagonal line at 45 degrees
    np.fill_diagonal(kernel, 1 + eps)
    
    return kernel


def circular_kernel(radius):
    """
    Create a circular binary kernel with the given radius using NumPy.

    Args:
        radius (int): Radius of the circle.

    Returns:
        np.ndarray: Circular kernel.
    """
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    kernel = mask.astype(np.float32)
    return kernel