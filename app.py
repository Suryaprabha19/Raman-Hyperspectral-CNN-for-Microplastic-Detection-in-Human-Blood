import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import json
import random

# ==========================================
# CONFIGURATION
# ==========================================
st.set_page_config(layout="wide", page_title="Raman Plastic Detector")
DATA_DIR = "raman_blood_dataset"

# Physical Properties Database (The "Lab Manual")
# Used to look up density/color based on the detected plastic type
PROPS = {
    'PET':  {'den': '1.38 g/cm³', 'col': 'Clear/Glassy',     'src': 'Water Bottles'},
    'PS':   {'den': '1.05 g/cm³', 'col': 'Opaque/Gray',      'src': 'Food Containers'},
    'PE':   {'den': '0.94 g/cm³', 'col': 'Translucent Blue', 'src': 'Plastic Bags'},
    'PP':   {'den': '0.90 g/cm³', 'col': 'Hazy White',       'src': 'Bottle Caps'},
    'PMMA': {'den': '1.18 g/cm³', 'col': 'Rigid Clear',      'src': 'Cosmetics'}
}

# ==========================================
# FUNCTIONS
# ==========================================
def load_random_sample():
    """Loads a random patient sample from the dataset folder."""
    if not os.path.exists(DATA_DIR):
        return None
    
    # Find all metadata files
    files = glob.glob(os.path.join(DATA_DIR, "metadata", "*.json"))
    if not files:
        return None
    
    # Pick one random patient
    choice = random.choice(files)
    base_id = os.path.splitext(os.path.basename(choice))[0]
    
    # Load the 3 matching files
    # 1. The RGB Image (The "Photo")
    img_path = os.path.join(DATA_DIR, "microscope_images", f"{base_id}.png")
    img = plt.imread(img_path)
    
    # 2. The Raman Cube (The "Scan")
    cube_path = os.path.join(DATA_DIR, "raman_cubes", f"{base_id}.npy")
    cube = np.load(cube_path)
    
    # 3. The Metadata (The "Truth")
    with open(choice, 'r') as f:
        meta = json.load(f)
    
    return img, cube, meta

def get_quantity_level(count):
    """Determines Low/Medium/High based on count."""
    # Extrapolating count to particles per mL (arbitrary scaling for demo)
    conc = count * 2500 
    if conc < 5000: return "Low", conc
    elif conc < 15000: return "Medium", conc
    else: return "High", conc

# ==========================================
# MAIN INTERFACE
# ==========================================
st.title("🩸 Blood Microplastic Detection System")
st.markdown("**Method:** Raman Microspectroscopy (532nm Excitation) | **Range:** 400-3500 cm⁻¹")

# Initialize Session State
if 'data' not in st.session_state:
    st.session_state['data'] = None
    st.session_state['analyzed'] = False

# --- SIDEBAR ---
st.sidebar.header("🕹️ Control Panel")
if st.sidebar.button("🔄 Load New Patient Sample"):
    result = load_random_sample()
    if result:
        st.session_state['data'] = result
        st.session_state['analyzed'] = False # Reset analysis
    else:
        st.sidebar.error(f"Dataset not found in folder '{DATA_DIR}'. Run data_factory.py first.")

# --- MAIN DISPLAY ---
if st.session_state['data']:
    img, cube, meta = st.session_state['data']
    particles = meta['particles']
    count = len(particles)
    
    col1, col2 = st.columns(2)
    
    # COLUMN 1: INPUT & SPECTRUM
    with col1:
        st.subheader("1. Microscopic View")
        st.image(img, caption="Patient Blood Sample (Brightfield)", use_container_width=True)
        
        if st.button("🔍 Run Raman Analysis"):
            st.session_state['analyzed'] = True
            
        # If analyzed, show the "Chemical Fingerprint" graph
        if st.session_state['analyzed'] and count > 0:
            st.markdown("### 📈 Chemical Fingerprint Detected")
            
            # Logic: Find the pixel with the strongest signal (plastic)
            # We sum across all bands; the pixel with highest sum is likely plastic
            sum_map = np.sum(cube, axis=2)
            y, x = np.unravel_index(np.argmax(sum_map), sum_map.shape)
            
            # Extract spectrum from that pixel
            spectrum = cube[y, x, :]
            wavenumbers = np.linspace(400, 3500, len(spectrum))
            
            # Plot
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(wavenumbers, spectrum, color='#d62728', lw=2)
            ax.set_title(f"Raman Spectrum at Pixel ({x},{y})")
            ax.set_xlabel("Raman Shift (cm⁻¹)")
            ax.set_ylabel("Intensity (Counts)")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            st.caption("Note the sharp peaks indicating polymer bonds.")

    # COLUMN 2: THE REPORT
    with col2:
        st.subheader("2. Analysis Report")
        
        if st.session_state['analyzed']:
            if count > 0:
                # Calculate Metrics
                qty_level, conc = get_quantity_level(count)
                
                # Dominant Type
                types = [p['type'] for p in particles]
                dom_type = max(set(types), key=types.count)
                
                # Average Size (Convert pixels to microns)
                sizes = [p['size'] for p in particles] # Assuming factory saves 'size'
                # Fallback if 'size' is missing in older json
                if not sizes and 'size_px' in particles[0]: 
                     sizes = [p['size_px'] for p in particles]
                
                # 1 px = 0.5 microns. Diameter formula approx.
                avg_dia_um = np.mean([2 * np.sqrt(s/np.pi) * 0.5 for s in sizes])
                
                # Lookup Properties
                p_props = PROPS.get(dom_type, {'den': '?', 'col': '?'})
                
                # --- RENDER REPORT ---
                st.success("✅ Microplastics Detected")
                
                st.markdown("### Microplastics (MP) Quantity")
                st.info(f"**{qty_level}** ({conc} particles/mL)")
                
                st.markdown("### Physical Characteristics")
                # Using a dictionary to create a clean list
                report_data = {
                    "MP Size": f"{avg_dia_um:.2f} microns",
                    "MP Color": p_props['col'],
                    "MP Density": p_props['den'],
                    "MP Shape": "Fragment/Sphere" # Defaulting for display
                }
                st.json(report_data)
                
                st.markdown(f"**Dominant Polymer:** {dom_type}")
                st.warning("⚠️ Risk: Particles detected are small enough to enter organs.")
                
            else:
                st.success("✅ Analysis Complete: Sample Clean")
                st.info("No microplastics detected in this sample.")
        else:
            st.info("Waiting for analysis... Click 'Run Raman Analysis'.")

else:
    st.info("👋 Welcome! Click 'Load New Patient Sample' in the sidebar to begin.")