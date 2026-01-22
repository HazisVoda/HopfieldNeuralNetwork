# Image settings
IMAGE_SIZE = 28
CROP_SIZE = 16
USE_CROPPING = True

# Network settings
NUM_PATTERNS = 10
THRESHOLD = 0

# Training settings
MAX_ITERATIONS = 100  #Maximum iterations for pattern recall
CONVERGENCE_CHECK = True
ASYNC_UPDATE = True

# Noise settings
NOISE_TYPE = "flip"  #"flip", "salt_pepper", "mask"
NOISE_LEVEL = 0.1
MASK_SIZE = 0.3

# Evaluation settings
NUM_TEST_SAMPLES = 20
RANDOM_SEED = 42

# Visualization settings
FIGURES_DIR = "figures"
SHOW_PLOTS = True
SAVE_PLOTS = True

# Dataset settings
MNIST_DIGITS = list(range(10))