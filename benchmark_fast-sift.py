"""
PWODE Benchmark Comparison: SIFT, FAST, and PWODE
===================================================
Compares feature detection performance across three methods:
- SIFT (Scale-Invariant Feature Transform): High accuracy, slow
- FAST (Features from Accelerated Segment Test): High speed, lower precision
- PWODE (Prime Wave Order-Detection Engine): Balanced speed/precision

Metrics:
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
- Runtime: Average processing time per image
- Throughput: Images processed per second

Author: Adrian Sutton
Date: 2025-10-27
Version: 1.0
"""

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import time
import logging
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import sys

# Import PWODE (ensure pwode_v8.3.py is in the same directory)
try:
    from pwode_v8_3 import (
        pwode_filter_2d,
        add_noise,
        get_image_array,
        load_cifar10
    )

    PWODE_AVAILABLE = True
except ImportError:
    print("Warning: pwode_v8_3.py not found. PWODE benchmark will be skipped.")
    PWODE_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ============================================================================
# CONFIGURATION
# ============================================================================

BENCHMARK_CONFIG = {
    'num_images': 100,  # Number of test images
    'noise_levels': [0.0, 0.1, 0.15, 0.2, 0.3],  # Gaussian noise std
    'distance_threshold': 5.0,  # Pixels for TP matching
    'output_dir': './benchmark_results',
    'save_visualizations': True,
    'num_visualizations': 5,
    'opencv_version': cv2.__version__,
}


# ============================================================================
# GROUND TRUTH GENERATION
# ============================================================================

def generate_synthetic_keypoints(img_shape, num_keypoints=50, min_distance=10):
    """
    Generate synthetic ground truth keypoints with spatial constraints.

    Args:
        img_shape: (H, W, C) tuple
        num_keypoints: Number of keypoints to generate
        min_distance: Minimum distance between keypoints (avoids clustering)

    Returns:
        keypoints: List of (x, y) tuples
    """
    H, W = img_shape[:2]
    keypoints = []
    attempts = 0
    max_attempts = num_keypoints * 100

    while len(keypoints) < num_keypoints and attempts < max_attempts:
        x = np.random.randint(min_distance, W - min_distance)
        y = np.random.randint(min_distance, H - min_distance)

        # Check minimum distance constraint
        if len(keypoints) == 0:
            keypoints.append((x, y))
        else:
            distances = [np.sqrt((x - kx) ** 2 + (y - ky) ** 2) for kx, ky in keypoints]
            if min(distances) >= min_distance:
                keypoints.append((x, y))

        attempts += 1

    return keypoints


def create_synthetic_image_with_keypoints(keypoints, shape=(32, 32, 3),
                                          signal_intensity=0.9, background_std=0.1):
    """
    Create synthetic image with Gaussian blobs at keypoint locations.

    Args:
        keypoints: List of (x, y) ground truth locations
        shape: Image dimensions
        signal_intensity: Peak intensity at keypoint centers
        background_std: Background noise level

    Returns:
        image: Synthetic image with embedded keypoints
    """
    H, W, C = shape
    image = np.random.normal(0.2, background_std, shape)

    # Add Gaussian blobs at each keypoint
    for x, y in keypoints:
        y_grid, x_grid = np.ogrid[-y:H - y, -x:W - x]
        mask = x_grid ** 2 + y_grid ** 2 <= 9  # Radius = 3 pixels
        for c in range(C):
            image[mask, c] = signal_intensity

    return np.clip(image, 0, 1)


# ============================================================================
# FEATURE DETECTORS
# ============================================================================

class SIFTDetector:
    """SIFT (Scale-Invariant Feature Transform) wrapper."""

    def __init__(self):
        """Initialize SIFT detector with default parameters."""
        try:
            self.detector = cv2.SIFT_create()
        except AttributeError:
            # Fallback for older OpenCV versions
            self.detector = cv2.xfeatures2d.SIFT_create()
        self.name = "SIFT"

    def detect(self, image):
        """
        Detect keypoints using SIFT.

        Args:
            image: Input image (HxWxC, float32, range [0,1])

        Returns:
            keypoints: List of (x, y) tuples
            runtime: Detection time in seconds
        """
        # Convert to uint8 grayscale for OpenCV
        gray = (image.mean(axis=2) * 255).astype(np.uint8)

        start_time = time.time()
        kp = self.detector.detect(gray, None)
        runtime = time.time() - start_time

        keypoints = [(int(k.pt[0]), int(k.pt[1])) for k in kp]
        return keypoints, runtime


class FASTDetector:
    """FAST (Features from Accelerated Segment Test) wrapper."""

    def __init__(self, threshold=10):
        """
        Initialize FAST detector.

        Args:
            threshold: Corner detection threshold (lower = more detections)
        """
        self.detector = cv2.FastFeatureDetector_create(threshold=threshold)
        self.name = "FAST"

    def detect(self, image):
        """
        Detect keypoints using FAST.

        Args:
            image: Input image (HxWxC, float32, range [0,1])

        Returns:
            keypoints: List of (x, y) tuples
            runtime: Detection time in seconds
        """
        gray = (image.mean(axis=2) * 255).astype(np.uint8)

        start_time = time.time()
        kp = self.detector.detect(gray, None)
        runtime = time.time() - start_time

        keypoints = [(int(k.pt[0]), int(k.pt[1])) for k in kp]
        return keypoints, runtime


class PWODEDetector:
    """PWODE (Prime Wave Order-Detection Engine) wrapper."""

    def __init__(self, modulus=30, qcs_threshold=0.5):
        """
        Initialize PWODE detector.

        Args:
            modulus: Wheel factorization modulus
            qcs_threshold: Quadratic coherence score threshold
        """
        if not PWODE_AVAILABLE:
            raise ImportError("PWODE not available. Check pwode_v8_3.py import.")
        self.modulus = modulus
        self.qcs_threshold = qcs_threshold
        self.name = "PWODE"

    def detect(self, image):
        """
        Detect keypoints using PWODE.

        Args:
            image: Input image (HxWxC, float32, range [0,1])

        Returns:
            keypoints: List of (x, y) tuples
            runtime: Detection time in seconds
        """
        start_time = time.time()
        validated_signals, stats = pwode_filter_2d(
            image,
            modulus=self.modulus,
            qcs_threshold=self.qcs_threshold,
            verbose=False
        )
        runtime = time.time() - start_time

        # Extract (x, y) coordinates from validated signals
        keypoints = [(signal['coords'][1], signal['coords'][0])
                     for signal in validated_signals]  # (col, row) -> (x, y)

        return keypoints, runtime


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def compute_matches(detected_kp, ground_truth_kp, distance_threshold=5.0):
    """
    Match detected keypoints to ground truth using nearest neighbor with threshold.

    Args:
        detected_kp: List of (x, y) detected keypoints
        ground_truth_kp: List of (x, y) ground truth keypoints
        distance_threshold: Maximum distance for a valid match (pixels)

    Returns:
        tp: True positives (matched detections)
        fp: False positives (unmatched detections)
        fn: False negatives (unmatched ground truth)
        matches: List of (detected_idx, gt_idx, distance) tuples
    """
    if len(detected_kp) == 0:
        return 0, 0, len(ground_truth_kp), []

    if len(ground_truth_kp) == 0:
        return 0, len(detected_kp), 0, []

    # Compute pairwise distances
    detected_arr = np.array(detected_kp)
    gt_arr = np.array(ground_truth_kp)

    distances = np.sqrt(
        ((detected_arr[:, np.newaxis, 0] - gt_arr[np.newaxis, :, 0]) ** 2) +
        ((detected_arr[:, np.newaxis, 1] - gt_arr[np.newaxis, :, 1]) ** 2)
    )

    # Greedy matching: assign each detection to nearest GT (if within threshold)
    matched_gt = set()
    matched_detected = set()
    matches = []

    # Sort by distance (ascending)
    det_idx, gt_idx = np.unravel_index(np.argsort(distances, axis=None), distances.shape)

    for d_idx, g_idx in zip(det_idx, gt_idx):
        if d_idx in matched_detected or g_idx in matched_gt:
            continue
        if distances[d_idx, g_idx] <= distance_threshold:
            matched_detected.add(d_idx)
            matched_gt.add(g_idx)
            matches.append((d_idx, g_idx, distances[d_idx, g_idx]))

    tp = len(matched_detected)
    fp = len(detected_kp) - tp
    fn = len(ground_truth_kp) - len(matched_gt)

    return tp, fp, fn, matches


def compute_metrics(tp, fp, fn):
    """
    Compute precision, recall, and F1-score.

    Args:
        tp: True positives
        fp: False positives
        fn: False negatives

    Returns:
        metrics: Dictionary with precision, recall, f1
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


# ============================================================================
# BENCHMARK EXECUTION
# ============================================================================

def run_single_benchmark(detector, image, ground_truth_kp, distance_threshold=5.0):
    """
    Run detector on single image and evaluate.

    Args:
        detector: Feature detector instance (SIFT/FAST/PWODE)
        image: Input image
        ground_truth_kp: List of ground truth keypoints
        distance_threshold: Matching threshold

    Returns:
        result: Dictionary with metrics and runtime
    """
    detected_kp, runtime = detector.detect(image)
    tp, fp, fn, matches = compute_matches(detected_kp, ground_truth_kp, distance_threshold)
    metrics = compute_metrics(tp, fp, fn)

    return {
        'detector': detector.name,
        'runtime': runtime,
        'num_detected': len(detected_kp),
        'num_ground_truth': len(ground_truth_kp),
        **metrics,
        'detected_kp': detected_kp,
        'matches': matches
    }


def run_benchmark_suite(num_images=100, noise_levels=[0.0, 0.15],
                        distance_threshold=5.0, save_results=True):
    """
    Run comprehensive benchmark comparing SIFT, FAST, and PWODE.

    Args:
        num_images: Number of test images
        noise_levels: List of Gaussian noise standard deviations
        distance_threshold: Matching threshold for TP/FP
        save_results: Save results to CSV and JSON

    Returns:
        results_df: Pandas DataFrame with all results
        summary: Dictionary with aggregate statistics
    """
    logging.info("=" * 80)
    logging.info("PWODE BENCHMARK SUITE - SIFT vs FAST vs PWODE")
    logging.info("=" * 80)
    logging.info(f"Configuration:")
    logging.info(f"  Images: {num_images}")
    logging.info(f"  Noise levels: {noise_levels}")
    logging.info(f"  Distance threshold: {distance_threshold} pixels")
    logging.info(f"  OpenCV version: {BENCHMARK_CONFIG['opencv_version']}")
    logging.info("=" * 80)

    # Initialize detectors
    detectors = [
        SIFTDetector(),
        FASTDetector(threshold=10),
    ]
    if PWODE_AVAILABLE:
        detectors.append(PWODEDetector(modulus=30, qcs_threshold=0.5))

    # Results storage
    all_results = []

    # Run benchmark
    for noise_std in noise_levels:
        logging.info(f"\nTesting with noise std = {noise_std:.2f}")
        logging.info("-" * 80)

        for img_idx in range(num_images):
            # Generate synthetic image with known keypoints
            ground_truth_kp = generate_synthetic_keypoints(
                (32, 32, 3),
                num_keypoints=50,
                min_distance=5
            )
            clean_image = create_synthetic_image_with_keypoints(
                ground_truth_kp,
                shape=(32, 32, 3)
            )

            # Add noise
            if noise_std > 0:
                noisy_image = clean_image + np.random.normal(0, noise_std, clean_image.shape)
                noisy_image = np.clip(noisy_image, 0, 1)
            else:
                noisy_image = clean_image

            # Test each detector
            for detector in detectors:
                result = run_single_benchmark(
                    detector,
                    noisy_image,
                    ground_truth_kp,
                    distance_threshold
                )
                result['image_idx'] = img_idx
                result['noise_std'] = noise_std
                all_results.append(result)

            # Progress logging
            if (img_idx + 1) % 20 == 0:
                logging.info(f"  Processed {img_idx + 1}/{num_images} images...")

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)

    # Compute summary statistics
    summary = {}
    for detector_name in results_df['detector'].unique():
        detector_results = results_df[results_df['detector'] == detector_name]
        summary[detector_name] = {
            'avg_precision': detector_results['precision'].mean(),
            'std_precision': detector_results['precision'].std(),
            'avg_recall': detector_results['recall'].mean(),
            'std_recall': detector_results['recall'].std(),
            'avg_f1': detector_results['f1'].mean(),
            'std_f1': detector_results['f1'].std(),
            'avg_runtime': detector_results['runtime'].mean(),
            'std_runtime': detector_results['runtime'].std(),
            'throughput': 1.0 / detector_results['runtime'].mean(),  # images/sec
        }

    # Display summary
    logging.info("\n" + "=" * 80)
    logging.info("BENCHMARK SUMMARY")
    logging.info("=" * 80)
    logging.info(
        f"{'Detector':<10} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<12} | {'Runtime (s)':<12} | {'Throughput (img/s)':<12}")
    logging.info("-" * 80)

    for detector_name, stats in summary.items():
        logging.info(
            f"{detector_name:<10} | "
            f"{stats['avg_precision']:.3f}±{stats['std_precision']:.3f}  | "
            f"{stats['avg_recall']:.3f}±{stats['std_recall']:.3f}  | "
            f"{stats['avg_f1']:.3f}±{stats['std_f1']:.3f}  | "
            f"{stats['avg_runtime']:.4f}±{stats['std_runtime']:.4f} | "
            f"{stats['throughput']:.1f}"
        )

    logging.info("=" * 80)

    # Speedup analysis
    if 'SIFT' in summary and PWODE_AVAILABLE and 'PWODE' in summary:
        sift_time = summary['SIFT']['avg_runtime']
        pwode_time = summary['PWODE']['avg_runtime']
        speedup = sift_time / pwode_time

        precision_diff = summary['PWODE']['avg_precision'] - summary['SIFT']['avg_precision']

        logging.info(f"\nPWODE vs SIFT:")
        logging.info(f"  Speedup: {speedup:.1f}×")
        logging.info(f"  Precision difference: {precision_diff:+.3f}")
        logging.info(f"  Trade-off: {speedup:.1f}× faster with {precision_diff:+.1%} precision")

    # Save results
    if save_results:
        output_dir = Path(BENCHMARK_CONFIG['output_dir'])
        output_dir.mkdir(exist_ok=True)

        # Save CSV
        csv_path = output_dir / 'benchmark_results.csv'
        results_df.to_csv(csv_path, index=False)
        logging.info(f"\nResults saved to: {csv_path}")

        # Save JSON summary
        json_path = output_dir / 'benchmark_summary.json'
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logging.info(f"Summary saved to: {json_path}")

    return results_df, summary


# ============================================================================
# NOISE ROBUSTNESS ANALYSIS
# ============================================================================

def analyze_noise_robustness(results_df):
    """
    Analyze detector performance across different noise levels.

    Args:
        results_df: DataFrame from run_benchmark_suite

    Returns:
        noise_analysis: Dictionary with per-detector noise curves
    """
    logging.info("\n" + "=" * 80)
    logging.info("NOISE ROBUSTNESS ANALYSIS")
    logging.info("=" * 80)

    noise_analysis = {}

    for detector_name in results_df['detector'].unique():
        detector_results = results_df[results_df['detector'] == detector_name]
        noise_curve = detector_results.groupby('noise_std').agg({
            'precision': ['mean', 'std'],
            'recall': ['mean', 'std'],
            'f1': ['mean', 'std']
        })
        noise_analysis[detector_name] = noise_curve

        logging.info(f"\n{detector_name}:")
        logging.info(noise_curve.to_string())

    return noise_analysis


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_detections(image, ground_truth_kp, detected_kp, matches,
                         detector_name, save_path=None):
    """
    Visualize detection results with ground truth overlay.

    Args:
        image: Input image
        ground_truth_kp: Ground truth keypoints
        detected_kp: Detected keypoints
        matches: List of (det_idx, gt_idx, distance) tuples
        detector_name: Name of detector
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image)

    # Plot ground truth (green circles)
    for x, y in ground_truth_kp:
        circle = plt.Circle((x, y), 3, color='green', fill=False, linewidth=2)
        ax.add_patch(circle)

    # Plot detections (red crosses)
    detected_arr = np.array(detected_kp)
    if len(detected_arr) > 0:
        ax.scatter(detected_arr[:, 0], detected_arr[:, 1],
                   marker='x', color='red', s=100, linewidths=2)

    # Draw matches (blue lines)
    for det_idx, gt_idx, dist in matches:
        det_x, det_y = detected_kp[det_idx]
        gt_x, gt_y = ground_truth_kp[gt_idx]
        ax.plot([det_x, gt_x], [det_y, gt_y], 'b-', alpha=0.5, linewidth=1)

    ax.set_title(f"{detector_name} Detection Results\n"
                 f"GT: {len(ground_truth_kp)}, Detected: {len(detected_kp)}, Matched: {len(matches)}")
    ax.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_comparison_plots(results_df, output_dir):
    """
    Create comparison plots for benchmark results.

    Args:
        results_df: Results DataFrame
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # 1. Precision-Recall scatter
    fig, ax = plt.subplots(figsize=(10, 8))
    for detector_name in results_df['detector'].unique():
        detector_data = results_df[results_df['detector'] == detector_name]
        ax.scatter(
            detector_data['recall'],
            detector_data['precision'],
            label=detector_name,
            alpha=0.6,
            s=50
        )
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Comparison', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'precision_recall_scatter.png', dpi=150)
    plt.close()

    # 2. Runtime comparison boxplot
    fig, ax = plt.subplots(figsize=(10, 6))
    runtime_data = [
        results_df[results_df['detector'] == name]['runtime'].values
        for name in results_df['detector'].unique()
    ]
    ax.boxplot(runtime_data, labels=results_df['detector'].unique())
    ax.set_ylabel('Runtime (seconds)', fontsize=12)
    ax.set_title('Runtime Comparison', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    plt.savefig(output_dir / 'runtime_boxplot.png', dpi=150)
    plt.close()

    # 3. Noise robustness curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = ['precision', 'recall', 'f1']

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        for detector_name in results_df['detector'].unique():
            detector_data = results_df[results_df['detector'] == detector_name]
            noise_curve = detector_data.groupby('noise_std')[metric].agg(['mean', 'std'])
            ax.errorbar(
                noise_curve.index,
                noise_curve['mean'],
                yerr=noise_curve['std'],
                label=detector_name,
                marker='o',
                capsize=5
            )
        ax.set_xlabel('Noise Std Dev', fontsize=12)
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.set_title(f'{metric.capitalize()} vs Noise Level', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'noise_robustness_curves.png', dpi=150)
    plt.close()

    logging.info(f"\nVisualization plots saved to: {output_dir}")


# ============================================================================
# CIFAR-10 REAL-WORLD TEST
# ============================================================================

def run_cifar10_benchmark(num_images=100, noise_std=0.15):
    """
    Run benchmark on real CIFAR-10 images (no ground truth, qualitative only).

    Args:
        num_images: Number of CIFAR-10 images to test
        noise_std: Gaussian noise level

    Returns:
        cifar_results: Dictionary with runtime and detection count statistics
    """
    logging.info("\n" + "=" * 80)
    logging.info("CIFAR-10 REAL-WORLD BENCHMARK (Qualitative)")
    logging.info("=" * 80)

    trainloader = load_cifar10(batch_size=1)

    detectors = [
        SIFTDetector(),
        FASTDetector(threshold=10),
    ]
    if PWODE_AVAILABLE:
        detectors.append(PWODEDetector(modulus=30, qcs_threshold=0.5))

    cifar_results = defaultdict(list)

    for img_idx, (images, _) in enumerate(trainloader):
        if img_idx >= num_images:
            break

        img_array = get_image_array(images)
        noisy_img = add_noise(img_array, gaussian_std=noise_std)

        for detector in detectors:
            detected_kp, runtime = detector.detect(noisy_img)
            cifar_results[detector.name].append({
                'runtime': runtime,
                'num_detected': len(detected_kp)
            })

        if (img_idx + 1) % 20 == 0:
            logging.info(f"Processed {img_idx + 1}/{num_images} CIFAR-10 images...")

    # Summary
    logging.info("\nCIFAR-10 Results:")
    logging.info(f"{'Detector':<10} | {'Avg Detections':<15} | {'Avg Runtime (s)':<15}")
    logging.info("-" * 50)

    for detector_name, results in cifar_results.items():
        avg_detections = np.mean([r['num_detected'] for r in results])
        avg_runtime = np.mean([r['runtime'] for r in results])
        logging.info(f"{detector_name:<10} | {avg_detections:>15.1f} | {avg_runtime:>15.4f}")

    return cifar_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main benchmark execution."""
    import argparse

    parser = argparse.ArgumentParser(description='PWODE Benchmark Suite')
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'quick', 'cifar10', 'visualize'],
                        help='Benchmark mode')
    parser.add_argument('--num_images', type=int, default=100,
                        help='Number of test images')
    parser.add_argument('--output_dir', type=str, default='./benchmark_results',
                        help='Output directory for results')

    args = parser.parse_args()

    # Update config
    BENCHMARK_CONFIG['num_images'] = args.num_images
    BENCHMARK_CONFIG['output_dir'] = args.output_dir

    if args.mode == 'full':
        # Full benchmark with multiple noise levels
        results_df, summary = run_benchmark_suite(
            num_images=args.num_images,
            noise_levels=[0.0, 0.05, 0.10, 0.15, 0.20, 0.30],
            distance_threshold=5.0,
            save_results=True
        )

        # Noise robustness analysis
        noise_analysis = analyze_noise_robustness(results_df)

        # Create visualizations
        create_comparison_plots(results_df, args.output_dir)

    elif args.mode == 'quick':
        # Quick benchmark (fewer images, single noise level)
        results_df, summary = run_benchmark_suite(
            num_images=20,
            noise_levels=[0.15],
            distance_threshold=5.0,
            save_results=True
        )

    elif args.mode == 'cifar10':
        # CIFAR-10 qualitative test
        cifar_results = run_cifar10_benchmark(
            num_images=args.num_images,
            noise_std=0.15
        )

    elif args.mode == 'visualize':
        # Generate sample visualizations
        logging.info("Generating visualization samples...")

        detectors = [SIFTDetector(), FASTDetector(), PWODEDetector()]
        output_dir = Path(args.output_dir) / 'visualizations'
        output_dir.mkdir(parents=True, exist_ok=True)

        for i in range(5):
            # Generate test image
            ground_truth_kp = generate_synthetic_keypoints((32, 32, 3), 50, 5)
            clean_image = create_synthetic_image_with_keypoints(ground_truth_kp)
            noisy_image = add_noise(clean_image, gaussian_std=0.15)

            for detector in detectors:
                result = run_single_benchmark(detector, noisy_image, ground_truth_kp)
                visualize_detections(
                    noisy_image,
                    ground_truth_kp,
                    result['detected_kp'],
                    result['matches'],
                    detector.name,
                    save_path=output_dir / f'sample_{i}_{detector.name}.png'
                )

        logging.info(f"Visualizations saved to: {output_dir}")

    logging.info("\nBenchmark complete!")


if __name__ == "__main__":
    main()
