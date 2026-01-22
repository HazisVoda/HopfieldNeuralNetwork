import numpy as np
from typing import Tuple, Optional, List
import tensorflow as tf
import tensorflow_datasets as tfds
from scipy.ndimage import zoom
tf.get_logger().setLevel('ERROR')


def load_mnist(split: str = "train") -> Tuple[np.ndarray, np.ndarray]:
    ds = tfds.load('mnist', split=split, as_supervised=True)
    images = []
    labels = []
    
    for image, label in tfds.as_numpy(ds):
        images.append(image)
        labels.append(label)
    
    images = np.array(images).squeeze()  #Remove channel dimension
    labels = np.array(labels)
    images = images.astype(np.float32) / 255.0
    
    return images, labels


def crop_to_content(image: np.ndarray, padding: int = 2) -> np.ndarray:
    rows = np.any(image > 0.1, axis=1)
    cols = np.any(image > 0.1, axis=0)
    
    if not rows.any() or not cols.any():
        return image
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmin = max(0, rmin - padding)
    rmax = min(image.shape[0], rmax + padding + 1)
    cmin = max(0, cmin - padding)
    cmax = min(image.shape[1], cmax + padding + 1)
    
    return image[rmin:rmax, cmin:cmax]


def resize_image(image: np.ndarray, target_size: int) -> np.ndarray:    
    h, w = image.shape
    zoom_h = target_size / h
    zoom_w = target_size / w
    
    return zoom(image, (zoom_h, zoom_w), order=1)


def preprocess_for_hopfield(images: np.ndarray, target_size: int = 20) -> np.ndarray:
    processed = []
    
    for img in images:
        cropped = crop_to_content(img, padding=1)
        resized = resize_image(cropped, target_size)
        processed.append(resized)
    
    return np.array(processed)


def binarize_images(images: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    # Convert to bipolar: values > threshold become +1, else -1
    binary = (images > threshold).astype(np.float32)
    bipolar = 2 * binary - 1
    return bipolar


def downsample_images(images: np.ndarray, target_size: int) -> np.ndarray:
    if images.ndim == 2:
        images = images[np.newaxis, ...]
        squeeze = True
    else:
        squeeze = False
    
    n_images = images.shape[0]
    original_size = images.shape[1]
    factor = original_size // target_size
    
    if factor == 1:
        return images.squeeze() if squeeze else images
    
    downsampled = images.reshape(
        n_images, target_size, factor, target_size, factor
    ).mean(axis=(2, 4))
    
    return downsampled.squeeze() if squeeze else downsampled


def flatten_images(images: np.ndarray) -> np.ndarray:
    if images.ndim == 2:
        return images.flatten()
    return images.reshape(images.shape[0], -1)


def unflatten_images(vectors: np.ndarray, image_size: int) -> np.ndarray:
    if vectors.ndim == 1:
        return vectors.reshape(image_size, image_size)
    return vectors.reshape(vectors.shape[0], image_size, image_size)


def prepare_mnist_data(
    split: str = "train",
    downsample_size: Optional[int] = None,
    digits: Optional[List[int]] = None,
    num_samples: Optional[int] = None,
    random_seed: int = 42,
    crop_and_resize: bool = True,
    crop_size: int = 16
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    np.random.seed(random_seed)
    images, labels = load_mnist(split)
    
    if digits is not None:
        mask = np.isin(labels, digits)
        images = images[mask]
        labels = labels[mask]
    
    if num_samples is not None and num_samples < len(images):
        indices = np.random.choice(len(images), num_samples, replace=False)
        images = images[indices]
        labels = labels[indices]
    
    if crop_and_resize:
        images = preprocess_for_hopfield(images, target_size=crop_size)
        image_size = crop_size
    elif downsample_size is not None and downsample_size < 28:
        images = downsample_images(images, downsample_size)
        image_size = downsample_size
    else:
        image_size = 28
    
    original_images = images.copy()
    bipolar = binarize_images(images, threshold=0.2)  # Lower threshold to capture more of digit
    patterns = flatten_images(bipolar)
    
    return patterns, original_images, labels, image_size


def get_samples_per_digit(
    patterns: np.ndarray,
    labels: np.ndarray,
    samples_per_digit: int = 5,
    digits: Optional[List[int]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    if digits is None:
        digits = list(range(10))
    
    selected_patterns = []
    selected_labels = []
    
    for digit in digits:
        digit_mask = labels == digit
        digit_patterns = patterns[digit_mask]
        digit_labels = labels[digit_mask]

        n_available = len(digit_patterns)
        n_select = min(samples_per_digit, n_available)
        
        indices = np.random.choice(n_available, n_select, replace=False)
        selected_patterns.append(digit_patterns[indices])
        selected_labels.append(digit_labels[indices])
    
    return np.vstack(selected_patterns), np.concatenate(selected_labels)


def select_diverse_patterns(
    patterns: np.ndarray,
    labels: np.ndarray,
    num_patterns: int,
    min_distance: float = 0.3
) -> Tuple[np.ndarray, np.ndarray]:
    n_total = len(patterns)
    n_neurons = patterns.shape[1]
    
    selected_indices = [np.random.randint(n_total)]
    candidates = list(range(n_total))
    candidates.remove(selected_indices[0])
    
    while len(selected_indices) < num_patterns and candidates:
        best_candidate = None
        best_min_distance = -1
        
        for c in candidates:
            distances = []
            for s in selected_indices:
                dist = np.mean(patterns[c] != patterns[s])
                distances.append(dist)
            
            min_dist = min(distances)
            
            if min_dist > best_min_distance:
                best_min_distance = min_dist
                best_candidate = c
        
        if best_candidate is not None and best_min_distance >= min_distance:
            selected_indices.append(best_candidate)
            candidates.remove(best_candidate)
        else:
            if best_candidate is not None:
                selected_indices.append(best_candidate)
                candidates.remove(best_candidate)
            else:
                break
    
    selected_indices = np.array(selected_indices)
    return patterns[selected_indices], labels[selected_indices]


def select_prototypical_patterns(
    patterns: np.ndarray,
    labels: np.ndarray,
    digits: Optional[List[int]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    if digits is None:
        digits = list(range(10))
    
    selected_patterns = []
    selected_labels = []
    
    for digit in digits:
        mask = labels == digit
        digit_patterns = patterns[mask]
        
        if len(digit_patterns) == 0:
            continue

        mean_pattern = np.mean(digit_patterns, axis=0)
        similarities = np.dot(digit_patterns, mean_pattern)
        best_idx = np.argmax(similarities)
        
        selected_patterns.append(digit_patterns[best_idx])
        selected_labels.append(digit)
    
    return np.array(selected_patterns), np.array(selected_labels)