import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import savgol_filter, find_peaks
import math
import logging
from pathlib import Path
from collections import defaultdict
import time
import os

# Set up logging with detailed output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIGURATION ---
CONFIG = {
    'modulus': 30,  # Final Optimized Filter Modulus
    'qcs_threshold': 0.6,  # Final Optimized Coherence Score threshold
    'echo_function': 'pnt',  # Final Functional Echo: 'pnt' (i·ln(i))
    'baseline_method': 'percentile',  # Standard for spectral baseline fitting
    'max_trials': 10,  # Number of trials for calculating CI
    'data_dir': Path('./data/'),  # Relative path for input data
    'output_dir': Path('./results/')  # Relative path for output results
}

# Precomputed admissible sets
ADMISSIBLE_SETS = {
    30: {1, 7, 11, 13, 17, 19, 23, 29},
}
PRIME_LIMITS = {30: 5}


# ============================================================================
# DATA HANDLING AND STRUCTURE
# ============================================================================

class SpectralData:
    """Container for 1D energy spectrum data."""

    def __init__(self, energies, intensities, metadata=None):
        self.energies = np.asarray(energies, dtype=np.float64)
        self.intensities = np.asarray(intensities, dtype=np.float64)
        self.N = len(energies)
        self.metadata = metadata or {}
        self.max_intensity = np.max(self.intensities)
        self.min_intensity = np.min(self.intensities)

        # Normalize intensities (0 to 1)
        if self.max_intensity > 0:
            self.normalized_intensities = self.intensities / self.max_intensity
        else:
            self.normalized_intensities = self.intensities


def load_spectral_data(file_name, data_dir: Path, energy_col='Energy(eV)', intensity_col='Total_DOS'):
    """Loads 1D spectral data from a file using relative path."""
    file_path = data_dir / file_name

    if not file_path.exists():
        logging.error(f"File not found: {file_path}. Ensure file is in the '{data_dir}' subfolder.")
        return None

    try:
        df = pd.read_csv(file_path, sep=r'\s+', engine='python')

        if energy_col not in df.columns or intensity_col not in df.columns:
            logging.error(f"Missing required columns: Expected '{energy_col}' and '{intensity_col}'.")
            return None

        energies = df[energy_col].values
        intensities = df[intensity_col].values

        metadata = {'source': file_name, 'units': energy_col.split('(')[1].split(')')[0],
                    'material': file_name.split('_')[0]}

        return SpectralData(energies, intensities, metadata=metadata)

    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        return None


# ============================================================================
# PHASE 1: MOD-N FILTER (DETERMINISTIC SPARSIFICATION)
# ============================================================================

def spectral_resonance_filter(index, modulus=30):
    """1D wheel factorization: Check if spectral index is admissible (prime-like)."""
    admissible = ADMISSIBLE_SETS[modulus]
    prime_limit = PRIME_LIMITS[modulus]

    if index > prime_limit and index % modulus in admissible:
        return 1
    else:
        return 0


def get_admissible_indices(N, modulus=30):
    """Returns a list of admissible indices for a given spectrum length N."""
    admissible_indices = [
        i for i in range(PRIME_LIMITS[modulus], N)
        if spectral_resonance_filter(i, modulus=modulus) == 1
    ]
    return admissible_indices


# ============================================================================
# PHASE 2: ADAPTIVE BASELINE & PEAK DETECTION (BASELINES)
# ============================================================================

def compute_adaptive_baseline(spectrum: SpectralData, method='percentile', percentile=10, window=50):
    """Compute adaptive baseline for spectral data."""
    intensities = spectrum.normalized_intensities
    N = len(intensities)

    if method == 'percentile':
        baseline = np.zeros(N)
        half_window = window // 2
        for i in range(N):
            start = max(0, i - half_window)
            end = min(N, i + half_window)
            baseline[i] = np.percentile(intensities[start:end], percentile)
        return baseline

    elif method == 'savitzky_golay':
        # Savitzky-Golay smoothing for baseline comparison
        window_size = min(N - (N % 2) - 1, 51)  # Must be odd, max 51 or N-1
        poly_order = min(window_size - 1, 3)
        return savgol_filter(intensities, window_size, poly_order, mode='mirror')

    else:  # Default or Fixed
        return np.full(N, np.min(intensities))


def detect_spectral_peaks(spectrum: SpectralData, baseline, prominence_factor=0.5, detector_type='SCIPY'):
    """Detect peaks above adaptive baseline using scipy.signal.find_peaks."""

    # Baseline detector (SCIPY/SAVGOL) uses the filtered signal
    if detector_type != 'PWODE':
        normalized = spectrum.normalized_intensities - baseline
    else:
        # PWODE baseline is only used for comparison metrics, not peak finding itself
        normalized = spectrum.normalized_intensities

    threshold = np.mean(baseline) * prominence_factor

    # Handle the potential for zero/negative values in the normalized array for height check
    valid_data_indices = np.where(normalized > 0)[0]

    if len(valid_data_indices) == 0:
        return np.array([]), 0

    # Find peaks on the *raw* intensity adjusted by the local threshold
    peak_indices, _ = signal.find_peaks(
        spectrum.normalized_intensities,
        height=baseline + threshold,
        distance=5,
        prominence=0.01
    )

    return peak_indices, len(peak_indices)


# ============================================================================
# PHASE 3: QUADRATIC COHERENCE SIGNATURE (QCS)
# ============================================================================

def compute_echo_index(primary_index, N, echo_function='pnt'):
    """Compute predicted "echo" location in spectral domain (1D)."""
    i = primary_index

    if i < 2: return 0

    if echo_function == 'pnt':
        # PNT-inspired: i·ln(i) mod N (Slower growth matches quantization)
        ln_i = np.log(i + 1)
        return int(i * ln_i) % N

    else:
        return 0


def compute_spectral_qcs(spectrum: SpectralData, primary_idx, echo_idx, baseline, qcs_threshold=0.6, echo_tolerance=5):
    """Compute Quadratic Coherence Score for spectral domain (1D)."""
    I_primary = spectrum.normalized_intensities[primary_idx]
    B_primary = baseline[primary_idx]

    echo_start = max(0, echo_idx - echo_tolerance)
    echo_end = min(spectrum.N, echo_idx + echo_tolerance + 1)

    I_echo_region = spectrum.normalized_intensities[echo_start:echo_end]
    B_echo_region = baseline[echo_start:echo_end]

    if len(I_echo_region) == 0:
        return 0.0, False

    I_echo_max = np.max(I_echo_region)
    B_echo_avg = np.mean(B_echo_region)

    primary_norm = (I_primary - B_primary) / (B_primary + 1e-6)
    echo_norm = (I_echo_max - B_echo_avg) / (B_echo_avg + 1e-6)

    # QCS: Geometric mean of normalized heights
    qcs = np.sqrt(primary_norm * echo_norm) / (np.max([primary_norm, echo_norm, 1e-6]) + 1e-6)
    qcs = np.clip(qcs, 0, 1)

    # Echo found if score meets threshold
    echo_found = (qcs >= qcs_threshold)

    return qcs, echo_found


# ============================================================================
# PWODE PIPELINE
# ============================================================================

def pwode_spectral_analysis(spectrum: SpectralData, modulus=30, qcs_threshold=0.6, echo_function='pnt'):
    """Complete PWODE spectral analysis pipeline for 1D data."""
    N = spectrum.N

    # Phase 2: Baseline Detection
    baseline = compute_adaptive_baseline(spectrum, method=CONFIG['baseline_method'])
    all_peaks, _ = detect_spectral_peaks(spectrum, baseline, prominence_factor=0.5, detector_type='SCIPY')

    # Phase 1: Apply resonance filter to all detected peaks (SCIPY peaks are candidates)
    admissible_indices = get_admissible_indices(N, modulus)
    admissible_peaks = [p for p in all_peaks if p in admissible_indices]

    # Phase 3: QCS validation on admissible peaks
    validated_peaks = []

    for peak_idx in admissible_peaks:
        # Check against adaptive baseline (Phase 2 refinement)
        if spectrum.normalized_intensities[peak_idx] - baseline[peak_idx] > 0.01:

            echo_idx = compute_echo_index(peak_idx, N, echo_function)
            qcs, echo_found = compute_spectral_qcs(spectrum, peak_idx, echo_idx, baseline, qcs_threshold)

            if qcs >= qcs_threshold and echo_found:
                validated_peaks.append({
                    'index': peak_idx,
                    'energy': spectrum.energies[peak_idx],
                    'intensity': spectrum.intensities[peak_idx],
                    'qcs': qcs,
                    'echo_index': echo_idx,
                })

    stats = {
        'N': N,
        'all_peaks': len(all_peaks),
        'admissible_peaks': len(admissible_peaks),
        'validated_peaks': len(validated_peaks),
        'admissible_indices': len(admissible_indices),
        'validation_rate': len(validated_peaks) / len(admissible_peaks) * 100 if admissible_peaks else 0,
        'qcs_threshold_used': qcs_threshold
    }

    return {
        'validated_peaks': validated_peaks,
        'baseline': baseline,
        'stats': stats
    }


# ============================================================================
# VALIDATION SUITE - METRICS AND EXECUTION
# ============================================================================

def run_detector_analysis(spectrum, detector_name):
    """Runs a single detector (PWODE/SCIPY/SAVGOL) and returns the core metrics."""

    baseline_percentile = compute_adaptive_baseline(spectrum, method='percentile')
    baseline_savgol = compute_adaptive_baseline(spectrum, method='savitzky_golay')

    if detector_name == 'PWODE':
        analysis = pwode_spectral_analysis(spectrum, modulus=CONFIG['modulus'],
                                           qcs_threshold=CONFIG['qcs_threshold'],
                                           echo_function=CONFIG['echo_function'])
        peaks = analysis['validated_peaks']
        peak_count = len(peaks)
        val_rate = analysis['stats']['validation_rate']

    elif detector_name == 'SCIPY':
        peaks_indices, peak_count = detect_spectral_peaks(spectrum, baseline_percentile, 0.5, 'SCIPY')
        peaks = [{'energy': spectrum.energies[i]} for i in peaks_indices]
        val_rate = 100.0

    elif detector_name == 'SAVGOL':
        peaks_indices, peak_count = detect_spectral_peaks(spectrum, baseline_savgol, 0.5, 'SAVGOL')
        peaks = [{'energy': spectrum.energies[i]} for i in peaks_indices]
        val_rate = 100.0

    return {'peak_count': peak_count, 'val_rate': val_rate}


def run_multi_trial_validation(spectrum, detector_name):
    """Runs multiple trials to get mean and std dev for stability metrics."""
    results = defaultdict(list)
    trial_count = CONFIG['max_trials']

    for _ in range(trial_count):
        start_time = time.time()
        res = run_detector_analysis(spectrum, detector_name)
        runtime = time.time() - start_time

        results['runtime'].append(runtime)
        results['peak_count'].append(res['peak_count'])
        results['val_rate'].append(res['val_rate'])

    # Compute Final CI/Mean
    summary = {
        'Detector': detector_name,
        'Validated_Mean': np.mean(results['peak_count']),
        'Validated_Std': np.std(results['peak_count']),
        'PNT_Coherence_Mean': np.mean(results['val_rate']),
        'PNT_Coherence_Std': np.std(results['val_rate']),
        'Runtime_Mean': np.mean(results['runtime']) * 1000,  # ms
        'Runtime_Std': np.std(results['runtime']) * 1000,
    }

    return summary


def run_validation_suite(diamond_regions, germanium_regions):
    """Runs the full comparison suite for Diamond and Germanium E-DOS."""

    # --- 1. DIAMOND E-DOS ANALYSIS (mp-66) ---
    diamond_file = "mp-66_dos.txt"
    diamond_spectrum = load_spectral_data(diamond_file, CONFIG['data_dir'], energy_col='Energy(eV)',
                                          intensity_col='Total_DOS')

    if diamond_spectrum is None:
        logging.error(f"Skipping Diamond analysis: Data file not loaded.")
        return []

    diamond_results = []

    logging.info("=" * 80)
    logging.info(f"PWODE V9.4 FINAL VALIDATION: {diamond_spectrum.metadata['material']} E-DOS")
    logging.info(f"Relevant Regions: E_g Edges {diamond_regions}")
    logging.info("-" * 80)

    for det in ['PWODE', 'SCIPY', 'SAVGOL']:
        diamond_results.append(run_multi_trial_validation(diamond_spectrum, det))

    # --- 2. GERMANIUM E-DOS ANALYSIS (mp-149) ---
    germanium_file = "mp-149_dos.txt"
    germanium_spectrum = load_spectral_data(germanium_file, CONFIG['data_dir'], energy_col='Energy(eV)',
                                            intensity_col='Total_DOS')

    if germanium_spectrum is None:
        logging.error(f"Skipping Germanium analysis: Data file not loaded.")
        return []

    germanium_results = []

    logging.info("\n" + "=" * 80)
    logging.info(f"PWODE V9.4 FINAL VALIDATION: {germanium_spectrum.metadata['material']} E-DOS")
    logging.info(f"Relevant Regions: E_g Edges {germanium_regions}")
    logging.info("-" * 80)

    for det in ['PWODE', 'SCIPY', 'SAVGOL']:
        germanium_results.append(run_multi_trial_validation(germanium_spectrum, det))

    return diamond_results, germanium_results


def format_final_table(results_list):
    """Formats the final comparison table for the report."""
    table_data = []
    for res in results_list:
        # Use Validation Rate as the measure of PNT Coherence
        coherence_rate = f"{res['PNT_Coherence_Mean']:.1f} ± {res['PNT_Coherence_Std']:.1f}"

        table_data.append({
            'Detector': res['Detector'],
            'Validated Peaks': f"{res['Validated_Mean']:.1f} ± {res['Validated_Std']:.1f}",
            'PNT Coherence (%)': coherence_rate,
            'Runtime (ms)': f"{res['Runtime_Mean']:.2f} ± {res['Runtime_Std']:.2f}"
        })
    return pd.DataFrame(table_data)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    # Create required directory structure
    for path in [CONFIG['data_dir'], CONFIG['output_dir']]:
        os.makedirs(path, exist_ok=True)

    # Band Gap regions (DFT values)
    DIAMOND_REGIONS = [(0.0, 4.12)]
    GERMANIUM_REGIONS = [(0.0, 0.67)]

    # Execute the Validation Suite
    diamond_res, germanium_res = run_validation_suite(DIAMOND_REGIONS, GERMANIUM_REGIONS)

    # --- Print Final Tables ---
    if diamond_res:
        logging.info("\n" + "=" * 80)
        logging.info("FINAL TABLE: DIAMOND (E-DOS mp-66)")
        logging.info("=" * 80)
        print(format_final_table(diamond_res).to_markdown(index=False))

    if germanium_res:
        logging.info("\n" + "=" * 80)
        logging.info("FINAL TABLE: GERMANIUM (E-DOS mp-149)")
        logging.info("=" * 80)
        print(format_final_table(germanium_res).to_markdown(index=False))