# ⚛️ PWODE Project: Prime-Wave Orbit Density Engineering

The PWODE project develops unconventional computational tools for **quantum spectral analysis** by leveraging principles from **Prime Number Theory (PNT)**. Our goal is to provide superior spectral sparsification, noise reduction, and automated feature detection in electronic density of states (E-DOS) and band structure calculations.

The **Prime Resonance** phenomenon utilizes the mathematical structure inherent in prime distribution to efficiently model the complex, non-linear patterns found in quantum spectra, drastically improving computational efficiency and prediction purity compared to traditional methods.

## 🚀 PWODE V10.0: The Spectral Tuner (Current Release)

The V10.0 release is a major architectural upgrade that introduces **dynamic material tuning**, solving the primary limitation of previous versions. The core of this version is the **Material Parameter Manager**, which automatically selects the optimal coherence threshold ($\\text{QCS}_{\\text{min}}$) for each material class, moving PWODE from a specialized filter to a **universal, tunable spectral engine.**

### Breakthrough: Tunability and Coherence

Our validation demonstrated that the coherence strength varies with bonding type. V10.0 automatically adapts to these empirical observations:

| Material Class | Bonding Type | QCSₘᵢₙ Threshold | V10.0 Status |
|---|---|---|---|
| **Covalent** | Si, Diamond, GaN | 0.60 | **OPTIMAL:** Strong Coherence required. |
| **Layered** | MoS₂ | 0.40 | **TUNED:** Lower coherence threshold needed. |
| **Ionic/Wide-Gap** | KCl | 0.25 | **TUNED:** Highly relaxed threshold for weak coherence. |
| **Metallic/Gapless** | Cu, Bi₂Te₃ | 0.01 | **VALIDATION:** Rejects noisy, gapless systems by default. |


## 🛠️ Getting Started

### 1\. Requirements

-   Python (3.8+)
    
-   Standard scientific libraries: `numpy`, `pandas`, `scipy`, `pathlib`.
    
-   Materials Project API Key: Required for the included download utility (`e-dos_download_v10.py`).
    

### 2\. File Structure

```
pwode-project/
```

### 3\. Usage: Running the Spectral Tuner

The core logic resides in `pwode_v10_spectral_tuner.py`. This single script performs the end-to-end analysis using the dynamically loaded parameters.

1.  **Prepare Data:** Ensure all necessary DOS files (available in the `/data` folder) are downloaded. We recommend using the provided `e_dos_download_v10.py` script.
    
2.  **Execute the Tuner:** Run the main analysis pipeline. The script automatically iterates through all files in the `/data` directory, fetching the correct $\\text{QCS}$ threshold for each one.
    

```
python pwode_v10_spectral_tuner.py
```

The output will be a comprehensive report table showing the dramatic **peak reduction** achieved by PWODE V10.0 compared to baseline methods ($\\text{SCIPY/SAVGOL}$), confirming the efficiency of the Prime Resonance approach on a material-by-material basis.

## 📚 Theory and Documentation

For a detailed technical breakdown of the $\\text{Modulus}=30$ wheel factorization, the $i \\cdot \\ln(i)$ echo function, and the derivation of the material-specific $\\text{QCS}$ values, please consult the official documentation:

-   **[PWODE V10.0 Technical Documentation](https://github.com/Tusk-Bilasimo/pwode-project/blob/main/docs/PWODE-SPECTRAL%20V.10.0.md)**
    

**We are not claiming primes are "fundamental" to quantum mechanics; we have demonstrated that prime number theory provides superior computational tools for quantum spectral analysis.**
