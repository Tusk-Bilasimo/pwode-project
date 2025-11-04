from mp_api.client import MPRester
import numpy as np
import os
from pathlib import Path

# --- Configuration ---
# Your Materials Project API key is kept consistent from previous chats.
API_KEY = "YOUR_API_KEY_HERE"

# Define the local folder where you want to save the files.
DOWNLOAD_PATH = Path.home() / "Downloads" / "PWODE_V10_Data"

# Define all 11 materials, including the 5 new challenge targets.
MATERIALS = {
    # Group IV (Initial Set)
    "mp-66": "Carbon_Diamond",
    "mp-149": "Silicon",
    "mp-32": "Germanium",
    "mp-33": "Tin_alpha",

    # Group III-V (First Extension)
    "mp-2534": "Gallium_Arsenide_GaAs_Cubic",
    "mp-804": "Gallium_Nitride_GaN_Hexagonal",

    # Phase 2 Challenge Materials (New Targets)
    "mp-1434": "Molybdenum_Disulfide_MoS2",  # Layered Hexagonal Structure
    "mp-568371": "Bismuth_Telluride_Bi2Te3",  # Topological Insulator
    "mp-2133": "Zinc_Oxide_ZnO",  # II-VI High Ionicity
    "mp-23193": "Potassium_Chloride_KCl",  # Ionic Insulator
    "mp-30": "Copper_Cu",  # Metal
}

print("=== Ensuring Download Directory ===")
DOWNLOAD_PATH.mkdir(parents=True, exist_ok=True)
print(f"Saving files to: {DOWNLOAD_PATH}")

print("\n=== Downloading Electronic DOS Data ===")

with MPRester(API_KEY) as mpr:
    for mp_id, name in MATERIALS.items():
        print(f"\nDownloading {name} ({mp_id})...")

        try:
            # Get electronic DOS
            dos_data = mpr.get_dos_by_material_id(mp_id)

            # Process data: shift energies by Fermi level
            energies = dos_data.energies - dos_data.efermi
            total_dos = dos_data.get_densities()

            # Define filename including the path
            filename = DOWNLOAD_PATH / f"{name}_{mp_id}_dos.txt"

            # Save to file
            with open(filename, "w") as f:
                f.write("Energy(eV) Total_DOS\n")

                # Write energy and DOS values
                for e, d in zip(energies, total_dos):
                    f.write(f"{e:.6f} {d:.6f}\n")

            print(f"✓ Successfully saved {filename.name} to {DOWNLOAD_PATH.name}")
            print(f"  Data points: {len(energies)}")
            print(f"  Energy range: {energies[0]:.2f} to {energies[-1]:.2f} eV")

        except Exception as e:
            print(f"✗ Failed to download {name} ({mp_id}): {e}")

print(f"\n=== Download Complete ===")