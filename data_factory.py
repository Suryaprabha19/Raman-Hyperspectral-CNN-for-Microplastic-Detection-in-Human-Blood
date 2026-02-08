import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.spatial import ConvexHull
import json
import os
import random
import shutil

# Try importing scikit-image
try:
    from skimage.draw import polygon, circle_perimeter, line_aa
except ImportError:
    print("Error: Library 'scikit-image' is missing. Run: pip install scikit-image")
    exit()

# ==========================================
# CONFIGURATION
# ==========================================
DATASET_SIZE = 350       # Number of patient samples
OUTPUT_DIR = "raman_blood_dataset"
IMG_H, IMG_W = 128, 128
N_BANDS = 100           # Higher resolution for Raman peaks
START_WN = 400          # Start Wavenumber (cm^-1)
END_WN = 3500           # End Wavenumber (cm^-1)

class RamanFactory:
    def __init__(self, output_dir):
        self.out_dir = output_dir
        self.wavenumbers = np.linspace(START_WN, END_WN, N_BANDS)
        
        # Directories
        self.dirs = {
            "cube": os.path.join(output_dir, "raman_cubes"),
            "img": os.path.join(output_dir, "microscope_images"),
            "mask": os.path.join(output_dir, "ground_truth_masks"),
            "meta": os.path.join(output_dir, "metadata")
        }
        
        if os.path.exists(output_dir): shutil.rmtree(output_dir)
        for d in self.dirs.values(): os.makedirs(d)

        # --- RAMAN LIBRARIES (The Physics) ---
        self.blood_spec = self._generate_blood_background()
        self.plastic_specs = {
            'PET': self._generate_raman_peaks([1615, 1730, 1290, 860, 2960]),
            'PS':  self._generate_raman_peaks([1001, 1031, 1602, 3054, 2900]),
            'PE':  self._generate_raman_peaks([1060, 1130, 1296, 1441, 2850, 2880]),
            'PMMA': self._generate_raman_peaks([810, 1450, 1730, 2950]),
            'PP':  self._generate_raman_peaks([809, 841, 972, 1151, 1450, 2900])
        }

    def _generate_raman_peaks(self, peaks):
        """Generates specific sharp peaks on a flat baseline."""
        spec = np.zeros_like(self.wavenumbers)
        for center in peaks:
            # Gaussian Peak Function
            # Peak width (sigma) is randomized slightly for realism
            sigma = random.uniform(10, 25) 
            height = random.uniform(0.5, 1.0)
            spec += height * np.exp(-((self.wavenumbers - center)**2) / (2 * sigma**2))
        return np.clip(spec, 0, 1)

    def _generate_blood_background(self):
        """Simulates biological fluorescence (a rising curve) + weak protein peaks."""
        x = self.wavenumbers
        # Fluorescence slope (common in biological Raman)
        baseline = 0.2 + 0.5 * (x / 3500)**2 
        # Weak peaks for Hemoglobin/Proteins (Amide I and III)
        protein_peaks = 0.1 * np.exp(-((x - 1650)**2)/(2*40**2)) + \
                        0.05 * np.exp(-((x - 1250)**2)/(2*40**2))
        return np.clip(baseline + protein_peaks, 0, 1)

    def _create_rgb_image(self):
        """Simulates Brightfield Microscopy."""
        img = np.ones((IMG_H, IMG_W, 3)) * np.array([255, 220, 200])/255.0
        # Add cellular texture
        noise = gaussian_filter(np.random.normal(0, 1, (IMG_H, IMG_W)), sigma=2)
        img[:,:,0] -= 0.05 * noise
        img[:,:,1] -= 0.15 * noise
        return np.clip(img, 0, 1)

    def generate_sample(self, idx):
        # 1. Particles Geometry
        n_particles = random.randint(3, 8)
        full_mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        particles = []
        
        for i in range(n_particles):
            p_type = random.choice(list(self.plastic_specs.keys()))
            cy, cx = np.random.randint(20, IMG_H-20), np.random.randint(20, IMG_W-20)
            size = random.randint(4, 12)
            
            # Simple Sphere Shape for robustness
            rr, cc = circle_perimeter(cy, cx, size, shape=(IMG_H, IMG_W))
            temp_mask = np.zeros((IMG_H, IMG_W))
            temp_mask[rr, cc] = 1
            temp_mask = distance_transform_edt(1-temp_mask) < size
            
            if temp_mask.sum() == 0: continue
            
            # Update Mask
            full_mask[temp_mask.astype(bool)] = i + 1
            
            # Abundance (Mixing)
            dist = distance_transform_edt(temp_mask)
            abundance = dist / (dist.max() + 1e-6)
            
            particles.append({
                'id': i+1, 'type': p_type, 
                'mask': temp_mask.astype(bool), 
                'abundance': abundance,
                'size_px': int(temp_mask.sum())
            })

        # 2. Build Raman Cube (The "Scan")
        # Start with Blood Background everywhere
        cube = np.zeros((IMG_H, IMG_W, N_BANDS))
        bg_noise = gaussian_filter(np.random.normal(1, 0.02, (IMG_H, IMG_W)), sigma=3)
        for b in range(N_BANDS):
            cube[:,:,b] = self.blood_spec[b] * bg_noise

        # Add Plastic Signals
        for p in particles:
            spec = self.plastic_specs[p['type']]
            mask = p['mask']
            alpha = p['abundance'][..., np.newaxis] # Expand for broadcasting
            
            # Linear Mixing: Signal = (1-alpha)*Blood + alpha*Plastic
            # Note: In Raman, signals often ADD on top of background
            current_region = cube[mask]
            plastic_signal = spec * alpha[mask]
            
            # Add plastic peaks ON TOP of blood fluorescence
            cube[mask] += plastic_signal

        # Add Sensor Noise (Shot Noise)
        cube += np.random.normal(0, 0.01, cube.shape)

        # 3. Build RGB Image (Visual)
        rgb = self._create_rgb_image()
        for p in particles:
            mask = p['mask']
            alpha = p['abundance'][..., np.newaxis]
            # Color tints
            if p['type'] == 'PET': col = [0.9, 0.9, 0.95]
            elif p['type'] == 'PE': col = [0.7, 0.8, 0.9]
            else: col = [0.6, 0.6, 0.6]
            
            rgb[mask] = rgb[mask] * (1 - alpha[mask]*0.6) + np.array(col) * alpha[mask]*0.6

        # 4. Save
        id_str = f"sample_{idx:04d}"
        np.save(os.path.join(self.dirs["cube"], f"{id_str}.npy"), cube.astype(np.float32))
        plt.imsave(os.path.join(self.dirs["img"], f"{id_str}.png"), rgb)
        plt.imsave(os.path.join(self.dirs["mask"], f"{id_str}_mask.png"), full_mask, cmap='gray')
        
        meta = {"id": id_str, "particles": [{"type": p['type'], "size": p['size_px']} for p in particles]}
        with open(os.path.join(self.dirs["meta"], f"{id_str}.json"), 'w') as f:
            json.dump(meta, f)
            
        return f"Generated {id_str}"

# Execution
print("Initializing Raman Lab Simulation...")
factory = RamanFactory(OUTPUT_DIR)
for i in range(DATASET_SIZE):
    factory.generate_sample(i)
print(f"Done! Saved to {OUTPUT_DIR}")