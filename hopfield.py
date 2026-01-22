import numpy as np
from typing import Tuple, List, Optional


class HopfieldNetwork:
    def __init__(self, n_neurons: int):
        self.n_neurons = n_neurons
        self.weights = np.zeros((n_neurons, n_neurons))
        self.patterns_stored = 0
        self.stored_patterns = None
        
    def train(self, patterns: np.ndarray) -> None:
        num_patterns = patterns.shape[0]
        self.patterns_stored = num_patterns
        self.stored_patterns = patterns.copy()
        self.weights = np.dot(patterns.T, patterns) / self.n_neurons
        
        np.fill_diagonal(self.weights, 0)
        
        capacity = int(0.138 * self.n_neurons)
        if num_patterns > capacity:
            print(f"Warning: Storing {num_patterns} patterns exceeds "
                  f"theoretical capacity of ~{capacity} patterns.")
    
    def energy(self, state: np.ndarray) -> float:

        return -0.5 * np.dot(state, np.dot(self.weights, state))
    
    def recall(self, pattern: np.ndarray, max_iter: int = 100, 
               track_energy: bool = False) -> Tuple[np.ndarray, List[float]]:
        
        state = pattern.copy().astype(np.float64)
        energy_history = []
        
        if track_energy:
            energy_history.append(self.energy(state))
        
        for iteration in range(max_iter):
            state_before = state.copy()
            for i in np.random.permutation(self.n_neurons):
                h_i = np.dot(self.weights[i], state)
                state[i] = 1.0 if h_i >= 0 else -1.0
            
            if track_energy:
                energy_history.append(self.energy(state))
            
            if np.array_equal(state, state_before):
                break
                
        return state, energy_history
    
    def compute_overlap(self, state: np.ndarray) -> np.ndarray:
        if self.stored_patterns is None:
            return np.array([])
        return np.dot(self.stored_patterns, state) / self.n_neurons
    
    def get_storage_info(self) -> dict:
        capacity = int(0.138 * self.n_neurons)
        return {
            "n_neurons": self.n_neurons,
            "patterns_stored": self.patterns_stored,
            "theoretical_capacity": capacity,
            "load_factor": self.patterns_stored / capacity if capacity > 0 else 0
        }