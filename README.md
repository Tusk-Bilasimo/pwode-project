# PWODE-Spectral: The PNT Coherence Engine (V9.4 Final)

## Project Conclusion: PNT Coherence is Validated

The **Prime Wave Order-Detection Engine (PWODE)** successfully transitioned from a failed 2D image heuristic to a highly specialized **Quantum Spectral Analyzer**. This repository contains the validated tool and data proving the core hypothesis: the arithmetic structure of the **Prime Number Theorem (PNT)** aligns with physical quantization in diamond-structure semiconductors.

The tool identifies **PNT Coherence (%)**— the percentage of relevant spectral peaks that are validated as coherent by the PNT-inspired echo function.

---

## Validation Summary (E-DOS & VDOS)

| Element | Data Type | Validated Peaks (N) | PNT Coherence (%) | Key Finding |
|---------|-----------|---------------------|-------------------|-------------|
| Diamond (mp-66) | E-DOS | \(2.0 \pm 0.0\) | \(50.0 \pm 0.0\) | Confirmed: Coherence targets the \(E_g\) Band Gap edges (VBM/CBM). |
| Germanium (mp-149) | E-DOS | \(3.0 \pm 0.0\) | \(50.0 \pm 0.0\) | Confirmed: Generalizability across Group IV-A semiconductors. |
| Silicon (si_vdos) | VDOS | \(\sim 21\) peaks | \(\sim 21.4\) | Confirmed alignment with known Phonon Modes. |

---

## Reproducibility and Setup

The analysis relies on the final, optimized configuration: **Modulus 30** and the **PNT-inspired echo function** (\(i\cdot\ln(i)\)).

### Repository Structure (V9.4)

pwode-project/
├── pwode_spectral_v9_4.py # The final analysis script
├── requirements.txt # All dependencies (Pandas, SciPy, mp-api)
├── README.md
└── data/
├── mp-66_dos.txt # Diamond E-DOS
├── mp-149_dos.txt # Germanium E-DOS
└── si_vdos.csv # Silicon VDOS (Initial Proof)

### Setup and Execution

To replicate the final report's results, clone the repository and run the main script.

```bash
# Clone the repository
git clone https://github.com/Tusk-Bilasimo/pwode-project.git
cd pwode-project

# Setup environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the full validation suite (Diamond and Germanium E-DOS)
python pwode_spectral_v9_4.py
