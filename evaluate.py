import numpy as np
from typing import Dict, List, Tuple, Optional
import os
from hopfield import HopfieldNetwork
from data_utils import prepare_mnist_data
from noise import add_noise, compute_noise_ratio
from visualization import (
    plot_denoising_result, 
    plot_multiple_denoising_results,
    plot_energy_convergence,
    plot_accuracy_vs_noise
)
import config


def compute_recovery_accuracy(original: np.ndarray, recovered: np.ndarray) -> float:
    return np.mean(original == recovered)


def compute_bit_error_rate(original: np.ndarray, recovered: np.ndarray) -> float:
    return np.mean(original != recovered)


def evaluate_single_pattern(
    network: HopfieldNetwork,
    original: np.ndarray,
    noise_type: str = "flip",
    noise_level: float = 0.2,
    image_size: Optional[int] = None,
    max_iter: int = 100,
    track_energy: bool = False
) -> Dict:
    noisy = add_noise(original, noise_type, noise_level, image_size)
    actual_noise = compute_noise_ratio(original, noisy)
    recovered, energy_history = network.recall(noisy, max_iter=max_iter, track_energy=track_energy)
    
    accuracy = compute_recovery_accuracy(original, recovered)
    ber = compute_bit_error_rate(original, recovered)
    perfect_recall = np.array_equal(original, recovered)
    
    return {
        'original': original,
        'noisy': noisy,
        'recovered': recovered,
        'actual_noise_level': actual_noise,
        'accuracy': accuracy,
        'bit_error_rate': ber,
        'perfect_recall': perfect_recall,
        'energy_history': energy_history,
        'iterations': len(energy_history) - 1 if energy_history else 0
    }


def evaluate_batch(
    network: HopfieldNetwork,
    patterns: np.ndarray,
    noise_type: str = "flip",
    noise_level: float = 0.2,
    image_size: Optional[int] = None,
    max_iter: int = 100
) -> Dict:
    results = []
    
    for pattern in patterns:
        result = evaluate_single_pattern(
            network, pattern, noise_type, noise_level,
            image_size, max_iter
        )
        results.append(result)
    
    accuracies = [r['accuracy'] for r in results]
    bers = [r['bit_error_rate'] for r in results]
    perfect_recalls = [r['perfect_recall'] for r in results]
    
    return {
        'individual_results': results,
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'mean_ber': np.mean(bers),
        'std_ber': np.std(bers),
        'perfect_recall_rate': np.mean(perfect_recalls),
        'num_patterns': len(patterns)
    }


def evaluate_noise_robustness(
    network: HopfieldNetwork,
    patterns: np.ndarray,
    noise_levels: List[float],
    noise_type: str = "flip",
    image_size: Optional[int] = None,
    max_iter: int = 100
) -> Dict:
    results = {
        'noise_levels': noise_levels,
        'mean_accuracies': [],
        'std_accuracies': [],
        'perfect_recall_rates': []
    }
    
    for level in noise_levels:
        batch_result = evaluate_batch(
            network, patterns, noise_type, level,
            image_size, max_iter
        )
        
        results['mean_accuracies'].append(batch_result['mean_accuracy'])
        results['std_accuracies'].append(batch_result['std_accuracy'])
        results['perfect_recall_rates'].append(batch_result['perfect_recall_rate'])
    
    return results


def run_evaluation(
    network: HopfieldNetwork,
    test_patterns: np.ndarray,
    test_labels: np.ndarray,
    image_size: int,
    noise_type: str = "flip",
    noise_level: float = 0.2,
    num_samples: int = 10,
    save_dir: str = "figures",
    show_plots: bool = True
) -> Dict:
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nEvaluating on {len(test_patterns)} patterns...")
    print(f"Noise type: {noise_type}, Noise level: {noise_level}")
    batch_results = evaluate_batch(
        network, test_patterns, noise_type, noise_level,
        image_size, config.MAX_ITERATIONS
    )
    
    print(f"\nResults:")
    print(f"  Mean accuracy: {batch_results['mean_accuracy']:.2%} ± {batch_results['std_accuracy']:.2%}")
    print(f"  Mean BER: {batch_results['mean_ber']:.4f}")
    print(f"  Perfect recall rate: {batch_results['perfect_recall_rate']:.2%}")
    
    originals = np.array([r['original'] for r in batch_results['individual_results'][:num_samples]])
    noisys = np.array([r['noisy'] for r in batch_results['individual_results'][:num_samples]])
    recovereds = np.array([r['recovered'] for r in batch_results['individual_results'][:num_samples]])
    
    plot_multiple_denoising_results(
        originals, noisys, recovereds, image_size,
        labels=test_labels[:num_samples],
        n_samples=min(num_samples, 5),
        save_path=os.path.join(save_dir, "denoising_results.png"),
        show=show_plots
    )
    
    single_result = evaluate_single_pattern(
        network, test_patterns[0], noise_type, noise_level,
        image_size, config.MAX_ITERATIONS,
        track_energy=True
    )
    
    plot_denoising_result(
        single_result['original'],
        single_result['noisy'],
        single_result['recovered'],
        image_size,
        title=f"Denoising Example (Digit: {test_labels[0]})",
        save_path=os.path.join(save_dir, "single_denoising.png"),
        show=show_plots
    )
    
    if single_result['energy_history']:
        plot_energy_convergence(
            single_result['energy_history'],
            title="Energy During Pattern Recall",
            save_path=os.path.join(save_dir, "energy_convergence.png"),
            show=show_plots
        )
    
    noise_levels = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    robustness_results = evaluate_noise_robustness(
        network, test_patterns[:20], noise_levels,
        noise_type, image_size
    )
    
    plot_accuracy_vs_noise(
        robustness_results['noise_levels'],
        robustness_results['mean_accuracies'],
        title="Recovery Accuracy vs Noise Level",
        save_path=os.path.join(save_dir, "noise_robustness.png"),
        show=show_plots
    )
    
    return {
        'batch_results': batch_results,
        'robustness_results': robustness_results
    }


def main():
    from train import load_network
    print("Hopfield Network Evaluation")
    
    # Load trained model
    print("\n[1] Loading trained model...")
    try:
        network, train_patterns, train_labels, image_size = load_network("hopfield_model.pkl")
        print(f"Loaded network with {network.patterns_stored} stored patterns")
    except FileNotFoundError:
        print("Error: No trained model found. Please run train.py first.")
        return
    
    # IMPORTANT: Test on the STORED patterns, not new unseen patterns!
    # Hopfield networks can only recall patterns they have memorized
    print("\n[2] Preparing test data...")
    print("  Note: Testing on stored patterns (Hopfield networks recall memorized patterns)")
    
    test_patterns = train_patterns
    test_labels = train_labels
    
    # Run evaluation
    print("\n[3] Running evaluation...")
    results = run_evaluation(
        network,
        test_patterns,
        test_labels,
        image_size,
        noise_type=config.NOISE_TYPE,
        noise_level=config.NOISE_LEVEL,
        num_samples=min(config.NUM_TEST_SAMPLES, len(test_patterns)),
        save_dir=config.FIGURES_DIR,
        show_plots=config.SHOW_PLOTS
    )
    print("\nEvaluation complete! Figures saved to:", config.FIGURES_DIR)
    
    return results


if __name__ == "__main__":
    main()