import json
import glob
import os
import numpy as np

DATA_DIR = "raman_blood_dataset"
MICRONS_PER_PIXEL = 0.5 
DEPTH_MICRONS = 10
VOL_PER_IMG_ML = (128*0.5*1e-4) * (128*0.5*1e-4) * (10*1e-4)

def generate_report():
    files = glob.glob(os.path.join(DATA_DIR, "metadata", "*.json"))
    if not files: return print("No data found.")
    
    total_particles = 0
    types = []
    sizes = []
    
    for f in files:
        with open(f) as jf:
            data = json.load(jf)
            for p in data['particles']:
                total_particles += 1
                types.append(p['type'])
                # Convert pixel area to diameter
                d_um = 2 * np.sqrt(p['size'] / np.pi) * MICRONS_PER_PIXEL
                sizes.append(d_um)

    # Metrics
    conc = total_particles / (len(files) * VOL_PER_IMG_ML)
    dom_type = max(set(types), key=types.count) if types else "None"
    avg_size = np.mean(sizes) if sizes else 0
    
    qty = "High" if conc > 10000 else "Medium" if conc > 5000 else "Low"
    
    print("-" * 40)
    print(" RAMAN SPECTROSCOPIC ANALYSIS REPORT")
    print("-" * 40)
    print(f"MP Quantity:   {qty} ({int(conc)} / mL)")
    print(f"Dominant Type: {dom_type}")
    print(f"Avg Size:      {avg_size:.2f} microns")
    print("-" * 40)

if __name__ == "__main__":
    generate_report()