"""
PWODE V8.3 - Prime Wave Order-Detection Engine
Complete implementation with all three phases:
- Phase 1: Static Mod-30 Resonance Filter (Deterministic Sparsification)
- Phase 2: Adaptive Threshold with Patch-based Refinement
- Phase 3: Quadratic Coherence Signature (QCS) Validation

Author: Adrian Sutton
Date: 2025-10-27
Version: 8.3 (Full Pipeline)
"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import math
import time
import logging
from collections import defaultdict

# Set up logging with detailed output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'modulus': 30,  # Wheel factorization modulus
    'initial_threshold': 0.3,  # Initial intensity threshold
    'qcs_threshold': 0.5,  # Quadratic Coherence Score threshold
    'tolerance': 0.5,  # Echo search tolerance
    'density_threshold': 0.05,  # Patch density threshold for sieving
    'iterations': 2,  # Iterative refinement passes
    'jitter_sigma': 0.5,  # Phase jitter parameter
    'patch_size': 12,  # Adaptive threshold patch size
    'gaussian_std': 0.15,  # Noise model parameter
    'harmonic_bases': [4, 9],  # Structured noise frequencies
    'harmonic_amp': 0.2,  # Harmonic noise amplitude
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_cifar10(batch_size=1):
    """Load CIFAR-10 dataset with minimal preprocessing."""
    transform = transforms.ToTensor()
    trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True
    )
    return trainloader


def get_image_array(image_tensor):
    """Extract image from batch tensor and convert to HxWxC format."""
    image = image_tensor[0]  # Remove batch dimension
    return image.numpy().transpose(1, 2, 0)  # CxHxW -> HxWxC


# ============================================================================
# NOISE MODEL
# ============================================================================

def add_noise(img_array, gaussian_std=0.15, harmonic_bases=[4, 9],
              harmonic_amp=0.2, jitter_sigma=0.5):
    """
    Add compound noise to image:
    - Gaussian: Sensor/thermal noise
    - Harmonic: Structured interference (row/column artifacts)
    - Jitter: Phase/timing instability
    """
    noisy = img_array + np.random.normal(0, gaussian_std, img_array.shape)
    H, W, C = img_array.shape

    # Add 2D structured harmonic noise
    for base in harmonic_bases:
        for r in range(base, H, base):
            noisy[r, :, :] += harmonic_amp * np.sin(2 * np.pi * r / base)
        for c in range(base, W, base):
            noisy[:, c, :] += harmonic_amp * np.sin(2 * np.pi * c / base)

    # Add phase jitter
    noisy += np.random.normal(0, jitter_sigma, img_array.shape)
    return np.clip(noisy, 0, 1)


def contrast_stretch(img_array, factor=0.1):
    """Enhance contrast for dim images (mean intensity < 0.2)."""
    if np.mean(img_array) < 0.2:
        min_val, max_val = np.min(img_array), np.max(img_array)
        stretched = img_array - min_val
        stretched = stretched / (max_val - min_val + 1e-7) * (1 - factor) + factor / 2
        return np.clip(stretched, 0, 1)
    return img_array


# ============================================================================
# PHASE 1: STATIC MOD-30 RESONANCE FILTER
# ============================================================================

# Precomputed wheel factorization sets (coprime residues)
ADMISSIBLE_SETS = {
    6: {1, 5},
    30: {1, 7, 11, 13, 17, 19, 23, 29},
    210: {r for r in range(1, 210) if math.gcd(r, 210) == 1},
    2310: {r for r in range(1, 2310) if math.gcd(r, 2310) == 1}
}

PRIME_LIMITS = {6: 3, 30: 5, 210: 7, 2310: 11}


def resonance_filter_2d(i, modulus=30, verbose=False):
    """
    Deterministic sparsification using wheel factorization.

    Returns 1 if index i is in admissible set (potential candidate),
    0 if fixed arithmetic noise (deterministically composite).

    Theoretical reduction: 1 - φ(modulus)/modulus
    - Mod-6:  66.7% reduction
    - Mod-30: 73.3% reduction
    """
    if modulus not in ADMISSIBLE_SETS:
        raise ValueError(f"Unsupported modulus: {modulus}")

    admissible = ADMISSIBLE_SETS[modulus]
    prime_limit = PRIME_LIMITS[modulus]

    if i > prime_limit and i % modulus in admissible:
        return 1  # Admissible candidate
    else:
        if verbose:
            logging.debug(f"Fixed Arithmetic Noise (Discarded) at index {i}")
        return 0  # Fixed arithmetic noise


# ============================================================================
# PHASE 2: ADAPTIVE THRESHOLD WITH PATCH REFINEMENT
# ============================================================================

def compute_adaptive_threshold(img_noisy, patch_size=12):
    """
    Compute local adaptive threshold using patch statistics.
    Avoids global threshold bias in non-uniform illumination.
    """
    H, W, C = img_noisy.shape
    patches = [
        img_noisy[i:i + patch_size, j:j + patch_size]
        for i in range(0, H, patch_size)
        for j in range(0, W, patch_size)
    ]
    # Threshold = mean + 0.5*std per patch, then average
    adaptive_thresh = np.mean([np.mean(p) + 0.5 * np.std(p) for p in patches])
    return adaptive_thresh


def apply_phase1_phase2(img_noisy, modulus=30, initial_threshold=0.3,
                        patch_size=12, verbose=False):
    """
    Combined Phase 1 (Mod-30) and Phase 2 (Adaptive Threshold).

    Returns:
        candidates: List of (r, c, ch) tuples for admissible high-intensity pixels
        adaptive_thresh: Computed threshold for Phase 3
        stats: Dictionary with phase-wise metrics
    """
    H, W, C = img_noisy.shape
    N = H * W * C
    flat_noisy = img_noisy.flatten()

    stats = {'N': N, 'phase1_discarded': 0, 'phase2_candidates': 0}

    # Adjust threshold for dim images
    if np.mean(img_noisy) < 0.2:
        initial_threshold = 0.5
        if np.where(flat_noisy > initial_threshold)[0].size == 0:
            img_noisy = contrast_stretch(img_noisy)
            initial_threshold = 0.3
            flat_noisy = img_noisy.flatten()

    # Compute adaptive threshold (Phase 2)
    adaptive_thresh = compute_adaptive_threshold(img_noisy, patch_size)

    # Combined Phase 1 + 2: Mod-30 filter + intensity threshold
    candidates = []
    for i in range(max(4, modulus), N):
        # Phase 1: Check admissibility
        if resonance_filter_2d(i, modulus, verbose=verbose) == 0:
            stats['phase1_discarded'] += 1
            continue

        # Phase 2: Check intensity
        if flat_noisy[i] > initial_threshold:
            r = i // (W * C)
            c = (i // C) % W
            ch = i % C
            candidates.append((r, c, ch))

    stats['phase2_candidates'] = len(candidates)
    return candidates, adaptive_thresh, img_noisy, stats


# ============================================================================
# PHASE 3: QUADRATIC COHERENCE SIGNATURE (QCS) VALIDATION
# ============================================================================

def time_warping(signal, fp, fp2_pred, jitter_sigma):
    """
    Apply linear time-warping correction for phase jitter compensation.
    Simulates signal alignment under timing instability.
    """
    delta_t = jitter_sigma
    t = np.arange(len(signal))
    w_t = t + delta_t * np.sin(2 * np.pi * t / len(signal))

    # Sort and interpolate to original time grid
    sorted_indices = np.argsort(w_t)
    w_t_sorted = w_t[sorted_indices]
    signal_sorted = signal[sorted_indices]
    warped = np.interp(t, w_t_sorted, signal_sorted)
    return warped


def compute_scs(fp, fp2_energy, noise_floor):
    """
    Quadratic Coherence Score (QCS): Measures signal-to-noise at predicted echo.

    Formula: SCS = E(fp²) / (noise_floor * (1/ln(fp)))

    High SCS (>0.5) indicates coherent signal with quadratic self-similarity.
    """
    ln_fp = np.log(fp + 1e-10) if fp > 1 else 1.0
    scs = fp2_energy / (noise_floor * (1 / ln_fp) + 1e-10)
    return scs


def compute_rpe(fp, fp2_obs, fp2_pred, signal, jitter_sigma=0.5):
    """
    Residual Phase Error (RPE): Quantifies phase alignment quality.

    Low RPE indicates stable, predictable signal phase.
    """
    fft_signal = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal))

    # Find phases at observed and predicted frequencies
    phase_obs = np.angle(fft_signal[np.argmin(np.abs(freqs - fp2_obs))])
    phase_pred = np.angle(fft_signal[np.argmin(np.abs(freqs - fp2_pred))])

    # Compute minimum angular distance (wrap-around aware)
    delta_phi = min(
        abs(phase_obs - phase_pred),
        2 * np.pi - abs(phase_obs - phase_pred)
    )

    delta_t = jitter_sigma
    rpe = delta_phi / (2 * np.pi * fp * delta_t + 1e-10) % 1
    return rpe


def compute_sli(fp, signal, fs, N):
    """
    Spectral Leakage Index (SLI): Measures frequency resolution quality.

    Low SLI indicates signal falls close to DFT bin center (good localization).
    """
    delta_f = fs / N
    k_near = round(fp / delta_f)
    delta = abs(fp - k_near * delta_f) / (delta_f + 1e-10)
    sli = np.sin(np.pi * delta) ** 2
    return sli


def apply_phase3_qcs(candidates, img_noisy, adaptive_thresh,
                     qcs_threshold=0.5, jitter_sigma=0.5, verbose=False):
    """
    Phase 3: Quadratic Coherence Signature (QCS) Validation.

    For each candidate from Phase 1+2:
    1. Predict "echo" location at fp² (scaled to image dimensions)
    2. Compute coherence score (SCS), phase error (RPE), leakage (SLI)
    3. Validate if SCS > threshold

    Returns:
        validated_signals: List of (fp, scs, rpe, sli) tuples for validated features
        stats: Dictionary with Phase 3 metrics
    """
    H, W, C = img_noisy.shape
    N = H * W * C
    flat_noisy = img_noisy.flatten()
    fs = 1.0  # Normalized sampling frequency

    validated_signals = []
    stats = {
        'phase3_checked': len(candidates),
        'phase3_validated': 0,
        'phase3_rejected': 0,
        'scs_values': [],
        'rpe_values': [],
        'sli_values': []
    }

    for (r, c, ch) in candidates:
        # Convert spatial coordinates to flattened index (fp)
        fp = r * W * C + c * C + ch

        # Predict quadratic echo location (scaled to image dimensions)
        # Using PNT approximation: fp² ≈ fp * ln(fp) for realistic spacing
        fp2_pred = fp * np.log(fp + 1e-10) if fp > 1 else fp
        fp2_pred = int(fp2_pred) % N  # Wrap to valid index

        # Extract echo energy at predicted location
        fp2_energy = flat_noisy[fp2_pred]

        # Compute Quadratic Coherence Score
        scs = compute_scs(fp, fp2_energy, adaptive_thresh)
        stats['scs_values'].append(scs)

        if scs > qcs_threshold:
            # Apply time-warping to entire channel for phase correction
            channel_signal = img_noisy[:, :, ch].flatten()
            warped_signal = time_warping(channel_signal, fp, fp2_pred, jitter_sigma)

            # Compute additional metrics on warped signal
            rpe = compute_rpe(fp, fp2_pred, fp2_pred, warped_signal, jitter_sigma)
            sli = compute_sli(fp, warped_signal, fs, len(warped_signal))

            stats['rpe_values'].append(rpe)
            stats['sli_values'].append(sli)

            validated_signals.append({
                'fp': fp,
                'coords': (r, c, ch),
                'scs': scs,
                'rpe': rpe,
                'sli': sli
            })
            stats['phase3_validated'] += 1

            if verbose:
                logging.debug(
                    f"Validated fp={fp} at ({r},{c},{ch}): "
                    f"SCS={scs:.3f}, RPE={rpe:.3f}, SLI={sli:.3f}"
                )
        else:
            stats['phase3_rejected'] += 1

    return validated_signals, stats


# ============================================================================
# FULL PWODE PIPELINE
# ============================================================================

def pwode_filter_2d(img_noisy, modulus=30, initial_threshold=0.3,
                    qcs_threshold=0.5, jitter_sigma=0.5,
                    patch_size=12, verbose=False):
    """
    Complete PWODE-2D filtering with three-phase cascade.

    Phase 1: Static Mod-30 Resonance Filter (Deterministic Sparsification)
    Phase 2: Adaptive Threshold with Patch-based Refinement
    Phase 3: Quadratic Coherence Signature (QCS) Validation

    Returns:
        validated_signals: List of validated feature dictionaries
        all_stats: Combined statistics from all phases
    """
    # Phase 1 + 2
    candidates, adaptive_thresh, img_processed, stats_p12 = apply_phase1_phase2(
        img_noisy, modulus, initial_threshold, patch_size, verbose
    )

    # Phase 3
    validated_signals, stats_p3 = apply_phase3_qcs(
        candidates, img_processed, adaptive_thresh,
        qcs_threshold, jitter_sigma, verbose
    )

    # Combine statistics
    all_stats = {**stats_p12, **stats_p3}
    all_stats['reduction_phase1'] = (stats_p12['phase1_discarded'] / stats_p12['N']) * 100
    all_stats['reduction_phase2'] = (1 - stats_p12['phase2_candidates'] / (
                stats_p12['N'] - stats_p12['phase1_discarded'])) * 100
    all_stats['reduction_phase3'] = (1 - stats_p3['phase3_validated'] / stats_p12['phase2_candidates']) * 100 if \
    stats_p12['phase2_candidates'] > 0 else 0
    all_stats['total_reduction'] = (1 - stats_p3['phase3_validated'] / stats_p12['N']) * 100

    return validated_signals, all_stats


# ============================================================================
# BENCHMARKING AND TESTING
# ============================================================================

def run_pwode_benchmark(num_images=10, modulus=30, qcs_threshold=0.5, verbose=False):
    """
    Run PWODE on CIFAR-10 with detailed performance tracking.

    Returns:
        results: List of per-image results with timing and statistics
        summary: Aggregate statistics across all images
    """
    trainloader = load_cifar10(batch_size=1)
    results = []

    logging.info("=" * 80)
    logging.info("PWODE V8.3 - Full Pipeline Benchmark")
    logging.info("=" * 80)
    logging.info(f"Configuration: Modulus={modulus}, QCS Threshold={qcs_threshold}")
    logging.info(f"Testing on {num_images} images from CIFAR-10")
    logging.info("=" * 80)

    total_start = time.time()

    for batch_idx, (images, labels) in enumerate(trainloader):
        if batch_idx >= num_images:
            break

        # Get image and add noise
        img_array = get_image_array(images)
        noisy_img = add_noise(
            img_array,
            gaussian_std=CONFIG['gaussian_std'],
            harmonic_bases=CONFIG['harmonic_bases'],
            harmonic_amp=CONFIG['harmonic_amp'],
            jitter_sigma=CONFIG['jitter_sigma']
        )

        # Run PWODE with timing
        start_time = time.time()
        validated_signals, stats = pwode_filter_2d(
            noisy_img,
            modulus=modulus,
            qcs_threshold=qcs_threshold,
            verbose=verbose
        )
        runtime = time.time() - start_time

        # Extract QCS metrics
        avg_scs = np.mean(stats['scs_values']) if stats['scs_values'] else 0
        avg_rpe = np.mean(stats['rpe_values']) if stats['rpe_values'] else 0
        avg_sli = np.mean(stats['sli_values']) if stats['sli_values'] else 0

        # Log results
        logging.info(
            f"Image {batch_idx:2d} | "
            f"Phase1: {stats['phase1_discarded']:4d} discarded ({stats['reduction_phase1']:5.1f}%) | "
            f"Phase2: {stats['phase2_candidates']:4d} candidates ({stats['reduction_phase2']:5.1f}%) | "
            f"Phase3: {stats['phase3_validated']:4d} validated ({stats['reduction_phase3']:5.1f}%) | "
            f"Total: {stats['total_reduction']:5.1f}% reduction | "
            f"Time: {runtime:.4f}s"
        )

        logging.info(
            f"         | "
            f"Avg SCS: {avg_scs:.3f} | "
            f"Avg RPE: {avg_rpe:.3f} | "
            f"Avg SLI: {avg_sli:.3f}"
        )

        results.append({
            'image_idx': batch_idx,
            'label': labels.item(),
            'runtime': runtime,
            'validated_count': stats['phase3_validated'],
            'stats': stats,
            'avg_scs': avg_scs,
            'avg_rpe': avg_rpe,
            'avg_sli': avg_sli,
            'validated_signals': validated_signals
        })

    total_time = time.time() - total_start

    # Compute summary statistics
    summary = {
        'total_time': total_time,
        'avg_time_per_image': total_time / num_images,
        'avg_validated_count': np.mean([r['validated_count'] for r in results]),
        'std_validated_count': np.std([r['validated_count'] for r in results]),
        'avg_scs': np.mean([r['avg_scs'] for r in results]),
        'avg_rpe': np.mean([r['avg_rpe'] for r in results]),
        'avg_sli': np.mean([r['avg_sli'] for r in results]),
        'avg_total_reduction': np.mean([r['stats']['total_reduction'] for r in results])
    }

    logging.info("=" * 80)
    logging.info("BENCHMARK SUMMARY")
    logging.info("=" * 80)
    logging.info(f"Total Runtime:        {summary['total_time']:.2f}s")
    logging.info(f"Avg Time/Image:       {summary['avg_time_per_image']:.4f}s")
    logging.info(f"Avg Validated:        {summary['avg_validated_count']:.1f} ± {summary['std_validated_count']:.1f}")
    logging.info(f"Avg Total Reduction:  {summary['avg_total_reduction']:.1f}%")
    logging.info(f"Avg SCS:              {summary['avg_scs']:.3f}")
    logging.info(f"Avg RPE:              {summary['avg_rpe']:.3f}")
    logging.info(f"Avg SLI:              {summary['avg_sli']:.3f}")
    logging.info("=" * 80)

    return results, summary


# ============================================================================
# ABLATION STUDY
# ============================================================================

def ablation_study(num_images=10):
    """
    Ablation study to measure contribution of each phase.

    Tests:
    1. Baseline: Adaptive threshold only
    2. + Mod-30: Phase 1 + 2
    3. + QCS: Full PWODE (Phase 1 + 2 + 3)
    """
    trainloader = load_cifar10(batch_size=1)

    logging.info("=" * 80)
    logging.info("ABLATION STUDY - Component Contribution Analysis")
    logging.info("=" * 80)

    ablation_results = {
        'baseline': [],
        'mod30': [],
        'full_pwode': []
    }

    for batch_idx, (images, _) in enumerate(trainloader):
        if batch_idx >= num_images:
            break

        img_array = get_image_array(images)
        noisy_img = add_noise(img_array)
        H, W, C = noisy_img.shape
        N = H * W * C
        flat_noisy = noisy_img.flatten()

        # 1. Baseline: Adaptive threshold only
        t0 = time.time()
        adaptive_thresh = compute_adaptive_threshold(noisy_img)
        baseline_candidates = np.where(flat_noisy > adaptive_thresh)[0]
        baseline_time = time.time() - t0

        # 2. + Mod-30: Phase 1 + 2
        t0 = time.time()
        mod30_candidates, _, _, mod30_stats = apply_phase1_phase2(noisy_img, modulus=30)
        mod30_time = time.time() - t0

        # 3. Full PWODE: Phase 1 + 2 + 3
        t0 = time.time()
        validated, pwode_stats = pwode_filter_2d(noisy_img, modulus=30, qcs_threshold=0.5)
        pwode_time = time.time() - t0

        ablation_results['baseline'].append({
            'count': len(baseline_candidates),
            'time': baseline_time
        })
        ablation_results['mod30'].append({
            'count': len(mod30_candidates),
            'time': mod30_time
        })
        ablation_results['full_pwode'].append({
            'count': len(validated),
            'time': pwode_time
        })

    # Summary
    logging.info("Configuration          | Avg Candidates | Avg Runtime | Speedup vs Baseline")
    logging.info("-" * 80)

    baseline_avg_count = np.mean([r['count'] for r in ablation_results['baseline']])
    baseline_avg_time = np.mean([r['time'] for r in ablation_results['baseline']])

    mod30_avg_count = np.mean([r['count'] for r in ablation_results['mod30']])
    mod30_avg_time = np.mean([r['time'] for r in ablation_results['mod30']])

    pwode_avg_count = np.mean([r['count'] for r in ablation_results['full_pwode']])
    pwode_avg_time = np.mean([r['time'] for r in ablation_results['full_pwode']])

    logging.info(
        f"Baseline (Adapt Thresh) | {baseline_avg_count:14.1f} | {baseline_avg_time:11.4f}s | 1.0×"
    )
    logging.info(
        f"+ Mod-30 Filter         | {mod30_avg_count:14.1f} | {mod30_avg_time:11.4f}s | "
        f"{baseline_avg_time / mod30_avg_time:.1f}×"
    )
    logging.info(
        f"+ QCS (Full PWODE)      | {pwode_avg_count:14.1f} | {pwode_avg_time:11.4f}s | "
        f"{baseline_avg_time / pwode_avg_time:.1f}×"
    )
    logging.info("=" * 80)

    return ablation_results


# ============================================================================
# NULL HYPOTHESIS TEST (QCS Validation)
# ============================================================================

def qcs_null_test(num_images=50, qcs_threshold=0.5):
    """
    Test QCS discriminative power on null images (uniform random noise).

    Null hypothesis: QCS validates random noise at same rate as signal.
    Expected: Validation rate < 5% for random images.
    """
    logging.info("=" * 80)
    logging.info("QCS NULL HYPOTHESIS TEST - Random Noise Rejection")
    logging.info("=" * 80)
    logging.info(f"Testing on {num_images} uniform random images")
    logging.info(f"QCS Threshold: {qcs_threshold}")
    logging.info("=" * 80)

    validation_rates = []

    for i in range(num_images):
        # Generate null image (uniform random)
        null_img = np.random.uniform(0, 1, (32, 32, 3))

        # Run PWODE
        validated, stats = pwode_filter_2d(
            null_img,
            modulus=30,
            qcs_threshold=qcs_threshold,
            verbose=False
        )

        validation_rate = (stats['phase3_validated'] / stats['N']) * 100
        validation_rates.append(validation_rate)

    avg_rate = np.mean(validation_rates)
    max_rate = np.max(validation_rates)
    std_rate = np.std(validation_rates)

    logging.info(f"Avg Validation Rate: {avg_rate:.2f}% ± {std_rate:.2f}%")
    logging.info(f"Max Validation Rate: {max_rate:.2f}%")
    logging.info(f"Expected (5% threshold): {'PASS' if max_rate < 5.0 else 'FAIL'}")
    logging.info("=" * 80)

    return validation_rates


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import sys

    # Parse command line arguments
    mode = sys.argv[1] if len(sys.argv) > 1 else "benchmark"

    if mode == "benchmark":
        # Standard benchmark on 10 images
        results, summary = run_pwode_benchmark(
            num_images=10,
            modulus=30,
            qcs_threshold=0.5,
            verbose=False
        )

    elif mode == "ablation":
        # Ablation study
        ablation_results = ablation_study(num_images=10)

    elif mode == "null_test":
        # QCS validation test
        validation_rates = qcs_null_test(num_images=50, qcs_threshold=0.5)

    elif mode == "verbose":
        # Detailed verbose run on 3 images
        results, summary = run_pwode_benchmark(
            num_images=3,
            modulus=30,
            qcs_threshold=0.5,
            verbose=True
        )

    else:
        print("Usage: python pwode_v8.3.py [benchmark|ablation|null_test|verbose]")
        sys.exit(1)
