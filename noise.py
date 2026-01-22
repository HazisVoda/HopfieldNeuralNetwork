import numpy as np
from typing import Tuple, Optional


def add_flip_noise(pattern: np.ndarray, noise_level: float = 0.2, 
                   random_seed: Optional[int] = None) -> np.ndarray:
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    noisy = pattern.copy()
    flip_mask = np.random.random(pattern.shape) < noise_level
    
    noisy[flip_mask] = -noisy[flip_mask]
    
    return noisy


def add_salt_pepper_noise(pattern: np.ndarray, noise_level: float = 0.2,
                          salt_ratio: float = 0.5,
                          random_seed: Optional[int] = None) -> np.ndarray:
    if random_seed is not None:
        np.random.seed(random_seed)
    
    noisy = pattern.copy()
    noise_mask = np.random.random(pattern.shape) < noise_level
    
    salt_mask = noise_mask & (np.random.random(pattern.shape) < salt_ratio)
    pepper_mask = noise_mask & ~salt_mask
    
    noisy[salt_mask] = 1
    noisy[pepper_mask] = -1
    
    return noisy


def add_mask_noise(pattern: np.ndarray, mask_fraction: float = 0.3,
                   mask_value: float = 0, mask_type: str = "random",
                   image_size: Optional[int] = None,
                   random_seed: Optional[int] = None) -> np.ndarray:
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    noisy = pattern.copy()
    n_pixels = len(pattern)
    n_mask = int(n_pixels * mask_fraction)
    
    if mask_type == "random":
        # Random pixel masking
        mask_indices = np.random.choice(n_pixels, n_mask, replace=False)
        
    elif mask_type == "block" and image_size is not None:
        # Square block mask
        block_size = int(np.sqrt(n_mask))
        max_start = image_size - block_size
        
        start_row = np.random.randint(0, max_start + 1)
        start_col = np.random.randint(0, max_start + 1)
        
        mask_indices = []
        for r in range(start_row, start_row + block_size):
            for c in range(start_col, start_col + block_size):
                mask_indices.append(r * image_size + c)
        mask_indices = np.array(mask_indices)
        
    elif mask_type == "left" and image_size is not None:
        # Mask left half
        mask_indices = []
        half_width = image_size // 2
        for r in range(image_size):
            for c in range(half_width):
                mask_indices.append(r * image_size + c)
        mask_indices = np.array(mask_indices)
        
    elif mask_type == "right" and image_size is not None:
        # Mask right half
        mask_indices = []
        half_width = image_size // 2
        for r in range(image_size):
            for c in range(half_width, image_size):
                mask_indices.append(r * image_size + c)
        mask_indices = np.array(mask_indices)
        
    elif mask_type == "top" and image_size is not None:
        # Mask top half
        mask_indices = []
        half_height = image_size // 2
        for r in range(half_height):
            for c in range(image_size):
                mask_indices.append(r * image_size + c)
        mask_indices = np.array(mask_indices)
        
    elif mask_type == "bottom" and image_size is not None:
        # Mask bottom half
        mask_indices = []
        half_height = image_size // 2
        for r in range(half_height, image_size):
            for c in range(image_size):
                mask_indices.append(r * image_size + c)
        mask_indices = np.array(mask_indices)
        
    else:
        # Default to random
        mask_indices = np.random.choice(n_pixels, n_mask, replace=False)
    
    # Apply mask
    if mask_value == "random":
        noisy[mask_indices] = np.random.choice([-1, 1], len(mask_indices))
    else:
        noisy[mask_indices] = mask_value
    
    return noisy


def add_gaussian_noise(pattern: np.ndarray, std: float = 0.5,
                       random_seed: Optional[int] = None) -> np.ndarray:

    if random_seed is not None:
        np.random.seed(random_seed)
    
    noisy_continuous = pattern + np.random.normal(0, std, pattern.shape)
    
    noisy = np.sign(noisy_continuous)
    noisy[noisy == 0] = 1  # Handle zero case
    
    return noisy


def add_noise(pattern: np.ndarray, noise_type: str = "flip", 
              noise_level: float = 0.2, image_size: Optional[int] = None,
              random_seed: Optional[int] = None, **kwargs) -> np.ndarray:

    if noise_type == "flip":
        return add_flip_noise(pattern, noise_level, random_seed)
    
    elif noise_type == "salt_pepper":
        return add_salt_pepper_noise(pattern, noise_level, 
                                      random_seed=random_seed, **kwargs)
    
    elif noise_type == "mask":
        mask_fraction = kwargs.get("mask_fraction", noise_level)
        mask_type = kwargs.get("mask_type", "random")
        mask_value = kwargs.get("mask_value", 0)
        return add_mask_noise(pattern, mask_fraction, mask_value, 
                              mask_type, image_size, random_seed)
    
    elif noise_type == "gaussian":
        return add_gaussian_noise(pattern, noise_level, random_seed)
    
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")


def compute_noise_ratio(original: np.ndarray, noisy: np.ndarray) -> float:
    return np.mean(original != noisy)