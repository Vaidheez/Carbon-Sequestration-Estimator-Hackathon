import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import detectree as dtr
import rasterio as rio
from rasterio import plot
import tempfile
import os
from PIL import Image
import io as python_io
import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Carbon Sequestration Estimation",
    page_icon="🌲",
    layout="wide"
)

# Add custom CSS for styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .title {
        font-size: 2.5rem;
        margin-bottom: 1.5rem;
    }
    .stButton > button {
        width: 100%;
        height: 3rem;
        font-size: 1.2rem;
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
    }
    .image-container {
        background-color: #f5f5f5;
        border-radius: 10px;
        padding: 20px;
        min-height: 300px;
        margin-bottom: 20px;
    }
    .placeholder-text {
        text-align: center;
        color: #888;
        padding-top: 100px;
    }
    .report-container {
        background-color: #f0f8ff;
        border-left: 5px solid #4682b4;
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 600px;
        background-color: #f9f9f9;
        color: #333;
        text-align: left;
        border-radius: 6px;
        padding: 15px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -300px;
        opacity: 0;
        transition: opacity 0.3s;
        box-shadow: 0px 0px 15px rgba(0,0,0,0.2);
        border: 1px solid #ddd;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    .results-section {
        background-color: #fff;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

#==============================================================================
# SECTION 1: VEGETATION DETECTION & IMAGE PROCESSING UTILITIES
# Functions for NDVI calculation, mask creation, and contour processing
#==============================================================================

def calculate_ndvi_from_rgb(image):
    """Calculate pseudo-NDVI from RGB image"""
    blue = image[:, :, 0].astype(float)
    green = image[:, :, 1].astype(float)
    red = image[:, :, 2].astype(float)
    
    epsilon = 1e-10
    pseudo_ndvi = (green - red) / (green + red + blue + epsilon)
    return pseudo_ndvi

def get_red_contour_mask(biased_y_pred, binary_mask):
    """Create the mask from red contours with transparency"""
    # Clean the mask
    kernel = np.ones((11, 11), np.uint8)
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
    
    # Get green contours
    _, thresholded_mask = cv2.threshold(binary_mask, 240, 255, cv2.THRESH_BINARY)
    contours_green, _ = cv2.findContours(thresholded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area_green = 5
    filtered_contours_green = [cnt for cnt in contours_green if cv2.contourArea(cnt) > min_area_green]
    
    # Get red contours
    contours_red, _, _ = find_buffered_black_region_contours(cleaned_mask, buffer_distance=10, min_area=700)
    
    # Create final mask (red filled areas that overlap with vegetation)
    final_mask = np.zeros_like(binary_mask)
    
    for cnt_red in contours_red:
        # Check overlap with vegetation (green contours)
        overlap_ratio = calculate_intersection_area(cnt_red, filtered_contours_green, binary_mask)
        if overlap_ratio >= 0.01:  # Using threshold
            cv2.drawContours(final_mask, [cnt_red], -1, 255, thickness=cv2.FILLED)
    
    return (final_mask > 0).astype(np.uint8)

def find_buffered_black_region_contours(cleaned_mask, buffer_distance=10, min_area=100):
    inverted = cv2.bitwise_not(cleaned_mask)
    dist_transform = cv2.distanceTransform(inverted, cv2.DIST_L2, 3)
    buffered_mask = np.where(dist_transform < buffer_distance, 255, 0).astype(np.uint8)
    contours, hierarchy = cv2.findContours(buffered_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    return filtered_contours, buffered_mask, dist_transform

def calculate_intersection_area(cnt_red, green_contours, binary_mask):
    mask_red = np.zeros_like(binary_mask, dtype=np.uint8)
    mask_green = np.zeros_like(binary_mask, dtype=np.uint8)
    cv2.drawContours(mask_red, [cnt_red], -1, 255, thickness=cv2.FILLED)
    cv2.drawContours(mask_green, green_contours, -1, 255, thickness=cv2.FILLED)
    intersection = cv2.bitwise_and(mask_red, mask_green)
    return np.count_nonzero(intersection) / np.count_nonzero(mask_red) if np.count_nonzero(mask_red) > 0 else 0

#==============================================================================
# SECTION 2: SPECIES DATA MODELING
# Define synthetic species parameters and probability distributions
#==============================================================================

def create_species_data():
    """Define synthetic species parameters (revised for realism)"""
    return {
        'Oak': {
            'type': 'Tree',
            'ndvi_range': (0.7, 0.9),
            'biomass': 18.0,             # kgC/m² total
            'seq_rate': 0.025,           # 2.5% of biomass/year
            'height': (15, 25),
            'canopy_factor': 1.8,
            'default_coverage': 0.30
        },
        'Pine': {
            'type': 'Tree',
            'ndvi_range': (0.6, 0.8),
            'biomass': 12.5,
            'seq_rate': 0.03,            # Slightly more due to fast growth
            'height': (10, 20),
            'canopy_factor': 1.5,
            'default_coverage': 0.25
        },
        'Maple': {
            'type': 'Tree',
            'ndvi_range': (0.5, 0.7),
            'biomass': 10.0,
            'seq_rate': 0.02,
            'height': (8, 15),
            'canopy_factor': 1.2,
            'default_coverage': 0.15
        },
        'Rhododendron': {
            'type': 'Shrub',
            'ndvi_range': (0.4, 0.6),
            'biomass': 3.0,
            'seq_rate': 0.015,
            'height': (2, 4),
            'canopy_factor': 0.7,
            'default_coverage': 0.15
        },
        'Juniper': {
            'type': 'Shrub',
            'ndvi_range': (0.3, 0.5),
            'biomass': 2.0,
            'seq_rate': 0.01,
            'height': (1, 3),
            'canopy_factor': 0.5,
            'default_coverage': 0.10
        },
        'Fescue': {
            'type': 'Grass',
            'ndvi_range': (0.1, 0.3),
            'biomass': 0.5,
            'seq_rate': 0.02,   # faster turnover
            'height': (0.1, 0.3),
            'canopy_factor': 0.2,
            'default_coverage': 0.05
        }
    }

def get_species_data_table():
    """Convert species data to a DataFrame for better display"""
    species_data = create_species_data()
    table_data = []
    
    for species, data in species_data.items():
        row = {
            'Species': species,
            'Type': data['type'],
            'NDVI Range': f"{data['ndvi_range'][0]:.1f}-{data['ndvi_range'][1]:.1f}",
            'Biomass (kg/m²)': data['biomass'],
            'Seq Rate (kgC/m²/yr)': data['seq_rate'],
            'Height (m)': f"{data['height'][0]}-{data['height'][1]}",
            'Coverage (%)': f"{data['default_coverage']*100:.0f}%"
        }
        table_data.append(row)
    
    return pd.DataFrame(table_data)

def calculate_species_probabilities(ndvi, species_data):
    """Gaussian probability distribution for species assignment"""
    probs = {}
    for name, data in species_data.items():
        mu = np.mean(data['ndvi_range'])  # Midpoint of NDVI range
        sigma = 0.15  # Controls distribution width
        # Gaussian weighted by default coverage
        probs[name] = data['default_coverage'] * np.exp(-(ndvi - mu)**2 / (2 * sigma**2))
    # Normalize to sum=1
    total = sum(probs.values())
    return {k: v/total for k, v in probs.items()}

#==============================================================================
# SECTION 3: CARBON ESTIMATION CORE ALGORITHMS
# Carbon sequestration calculations based on vegetation types and NDVI
#==============================================================================

def estimate_carbon(vegetation_mask, ndvi, pixel_area_m2=0.25, return_co2=True):
    """
    Carbon estimation pipeline with more realistic sequestration logic.
    If return_co2=True, converts carbon to CO₂ equivalent.
    """
    species_data = create_species_data()
    urban_efficiency_factor = 0.65  # Represents average stress & maintenance losses in suburban areas
    total_carbon = 0.0
    carbon_map = np.zeros_like(vegetation_mask, dtype=np.float32)
    
    CO2_CONVERSION_FACTOR = 3.67  # kgCO2 / kgC
    
    # Get vegetation pixels
    veg_pixels = np.where(vegetation_mask > 0)
    
    for y, x in zip(*veg_pixels):
        pixel_ndvi = ndvi[y, x]
        if pixel_ndvi <= 0:
            continue
        
        # Get species probabilities
        probs = calculate_species_probabilities(pixel_ndvi, species_data)
        
        pixel_carbon = 0.0
        
        for species, prob in probs.items():
            data = species_data[species]
            
            # Conservative biomass adjustment based on NDVI ratio to max
            max_ndvi = data['ndvi_range'][1]
            ndvi_factor = np.clip(pixel_ndvi / max_ndvi, 0.3, 1.0)
            adj_biomass = data['biomass'] * ndvi_factor
            
            # Sequestration as a fraction of biomass
            annual_seq = adj_biomass * data['seq_rate']
            pixel_carbon += prob * annual_seq
        
        # Scale to pixel area
        pixel_carbon *= pixel_area_m2 * urban_efficiency_factor
        if return_co2:
            pixel_carbon *= CO2_CONVERSION_FACTOR
        
        carbon_map[y, x] = pixel_carbon
        total_carbon += pixel_carbon
    
    return total_carbon, carbon_map

def calculate_csi(total_carbon_kg, vegetated_area_m2):
    """
    Calculates the Carbon Sequestration Index (CSI) on a logarithmic scale
    where typical values range 0-4 but can extend higher.
    
    Args:
        total_carbon_kg: Total annual carbon sequestration in kg
        vegetated_area_m2: Total vegetated area in square meters
        
    Returns:
        csi: Unitless index value (typically 0-4, but unbounded)
    """
    # Convert to metric tons per hectare per year
    area_ha = vegetated_area_m2 / 10000
    if area_ha == 0:
        return 0.0
    
    carbon_t_ha_yr = (total_carbon_kg / 1000) / area_ha
    
    # Base reference: 1.0 CSI = 2.5 tC/ha/yr (mixed temperate forest)
    # Using logarithmic scaling similar to Richter magnitude
    csi = np.log10(carbon_t_ha_yr / 0.25 + 1)
    
    return csi

def display_results(total_carbon, vegetated_area, csi):
    """Formatted output of carbon metrics with both vegetated and total region rates"""
    area_ha = vegetated_area / 10000
    total_region_area_ha = 100  # Fixed 1 km² reference area
    
    # Calculate both rates
    carbon_t_ha_yr = (total_carbon / 1000) / area_ha if area_ha > 0 else 0
    carbon_t_total_region = (total_carbon / 1000) / total_region_area_ha
    
    results = {
        "Vegetated Area": f"{area_ha:.2f} hectares",
        "Total Annual Sequestration": f"{total_carbon/1000:.1f} metric tons CO₂e",
        "Areal Rate (Vegetated)": f"{carbon_t_ha_yr:.2f} tCO₂e/ha/yr",
        "Areal Rate (Total Region)": f"{carbon_t_total_region:.2f} tCO₂e/ha/yr",
        "Carbon Sequestration Index (CSI)": f"{csi:.2f}"
    }
    
    interpretation = {
        "0.0-0.5": "Degraded/Bare land",
        "0.5-1.5": "Grasslands/Sparse vegetation",
        "1.5-2.5": "Shrublands/Young plantations", 
        "2.5-3.5": "Established forests",
        "3.5-4.5": "Mature native forests",
        ">4.5": "Exceptional carbon sinks"
    }
    
    return results, interpretation

def display_editable_species_table():
    """Display editable species table with collapsible sliders"""
    species_data = create_species_data()
    
    st.markdown("### Vegetation Species Parameters")
    
    # Display the read-only table by default
    table_data = []
    for species, data in species_data.items():
        row = {
            'Species': species,
            'Type': data['type'],
            'NDVI Range': f"{data['ndvi_range'][0]:.1f}-{data['ndvi_range'][1]:.1f}",
            'Biomass (kg/m²)': data['biomass'],
            'Seq Rate (kgC/m²/yr)': data['seq_rate'],
            'Height (m)': f"{data['height'][0]}-{data['height'][1]}",
            'Coverage (%)': f"{data['default_coverage']*100:.1f}%"
        }
        table_data.append(row)
    
    st.dataframe(pd.DataFrame(table_data), use_container_width=True)
    
    # Collapsible edit section
    with st.expander("Adjust Species Coverage", expanded=False):
        st.markdown("Adjust the species distribution percentages below (must sum to 100%):")

        # Fake a narrow container using 5 columns — only use the middle one
        left_spacer, mid_left_spacer, content_col, mid_right_spacer, right_spacer = st.columns([1, 1, 2, 1, 1])

        with content_col:
            total = 0.0
            coverage_values = {}

            for species, data in species_data.items():
                default_val = data['default_coverage'] * 100
                coverage_values[species] = st.slider(
                    f"{species} coverage",
                    min_value=0.0,
                    max_value=100.0,
                    value=default_val,
                    step=0.5,
                    key=f"coverage_{species}"
                )
                total += coverage_values[species]

            if abs(total - 100.0) > 0.1:
                st.warning(f"Current total: {total:.1f}%. Adjust values to sum to exactly 100%.")
                st.stop()

            for species, coverage in coverage_values.items():
                species_data[species]['default_coverage'] = coverage / 100.0

    
    return species_data

#==============================================================================
# SECTION 4: MAIN PROCESSING PIPELINE
# End-to-end workflow from image input to carbon metrics and visualization
#==============================================================================

def process_image(jp2_file_path, tif_file_path):
    """Complete processing pipeline with carbon estimation and visualization."""
    try:
        # Step 1: Load RGB image using rasterio for JP2 file
        with rio.open(jp2_file_path) as src:
            # Read as RGB (assuming 3-band JP2)
            image = src.read()
            image = np.transpose(image, (1, 2, 0))
            if image.shape[2] > 3:  # If there are more bands, take first 3 for RGB
                image = image[:, :, :3]
        
        # Get shape information
        height, width = image.shape[:2]
        
        # Convert to RGB array if needed
        if len(image.shape) == 2:
            # Convert grayscale to pseudo-RGB
            image = np.stack((image,) * 3, axis=-1)
        
        # Create temporary TIFF file for detectree
        temp_tif = tempfile.NamedTemporaryFile(suffix='.tif', delete=False).name
        with rio.open(
            temp_tif, 'w',
            driver='GTiff',
            height=height,
            width=width,
            count=3,
            dtype=image.dtype
        ) as dst:
            dst.write(np.transpose(image, (2, 0, 1)))
        
        # Step 2: Get vegetation mask using detectree
        y_pred = dtr.Classifier().predict_img(temp_tif)
        os.remove(temp_tif)  # Clean up temp file
        
        biased_y_pred = np.where(y_pred > 0.1, 1, 0)
        binary_mask = (biased_y_pred * 255).astype(np.uint8)
        
        # Step 3: Get red contour mask
        red_contour_mask = get_red_contour_mask(biased_y_pred, binary_mask)
        
        # Step 4: Calculate NDVI
        ndvi = calculate_ndvi_from_rgb(image)
        
        # Step 5: Apply red contour mask to NDVI
        masked_ndvi = ndvi * red_contour_mask
        
        # Normalize only the masked areas
        masked_values = masked_ndvi[red_contour_mask > 0]
        if len(masked_values) > 0:
            min_val, max_val = masked_values.min(), masked_values.max()
            if max_val > min_val:
                masked_ndvi = np.where(
                    red_contour_mask > 0,
                    (masked_ndvi - min_val) / (max_val - min_val), 
                    0
                )
        
        # Step 6: Read segmentation mask from TIF file
        with rio.open(tif_file_path) as seg_src:
            seg_data = seg_src.read(1)  # Read first band
        
        # Apply colormap (similar to matplotlib's viridis) to segmentation mask
        norm_seg = (seg_data - seg_data.min()) / (seg_data.max() - seg_data.min() + 1e-8)
        colored_seg = plt.cm.viridis(norm_seg)
        colored_seg = (colored_seg[:,:,:3] * 255).astype(np.uint8)
        
        # Apply colormap to vegetation detection results (purple to yellow gradient)
        # Create a colored version of the vegetation detection result
        veg_colored = plt.cm.viridis(y_pred)  # Using viridis colormap like the notebook
        veg_colored = (veg_colored[:,:,:3] * 255).astype(np.uint8)
        
        # Step 7: Convert to 8-bit and apply colormap for visualization
        ndvi_viz = (masked_ndvi * 255).astype(np.uint8)
        ndvi_colored = cv2.applyColorMap(ndvi_viz, cv2.COLORMAP_JET)
        ndvi_colored[red_contour_mask == 0] = 0  # Set non-masked areas to black
        
        # Step 8: Calculate carbon metrics
        # Pixel size is typically provided by the raster metadata
        with rio.open(jp2_file_path) as src:
            # Get pixel dimensions in meters
            transform = src.transform
            pixel_size_x = abs(transform[0])
            pixel_size_y = abs(transform[4])
            pixel_area = pixel_size_x * pixel_size_y
        
        # Use 1.0 square meter as default if pixel size is unrealistically small
        if pixel_area < 0.01:
            pixel_area = 1.0
            
        total_carbon, carbon_map = estimate_carbon(red_contour_mask, masked_ndvi, pixel_area)
        
        # Calculate vegetated area in m²
        vegetated_pixels = np.sum(red_contour_mask > 0)
        vegetated_area_m2 = vegetated_pixels * pixel_area
        
        # Calculate CSI
        csi = calculate_csi(total_carbon, vegetated_area_m2)
        
        # Get formatted results
        results, interpretation = display_results(total_carbon, vegetated_area_m2, csi)
        
        # Convert carbon map to visualization
        carbon_norm = carbon_map / (np.max(carbon_map) + 1e-10)
        carbon_viz = (carbon_norm * 255).astype(np.uint8)
        carbon_viz_colored = cv2.applyColorMap(carbon_viz, cv2.COLORMAP_VIRIDIS)
        carbon_viz_rgb = cv2.cvtColor(carbon_viz_colored, cv2.COLOR_BGR2RGB)
        
        # Convert NDVI to RGB for display
        ndvi_rgb = cv2.cvtColor(ndvi_colored, cv2.COLOR_BGR2RGB)
        
        return {
            'original_image': image,
            'vegetation_mask': veg_colored,  # Now using the colorized version
            'segmentation_mask': colored_seg,  # Now using the colorized version
            'ndvi_image': ndvi_rgb,
            'carbon_map': carbon_viz_rgb,
            'results': results,
            'interpretation': interpretation,
            'csi': csi,
            'total_carbon': total_carbon,
            'vegetated_area': vegetated_area_m2
        }
    except Exception as e:
        st.error(f"Error in image processing: {str(e)}")
        raise e

# Function to save uploaded file to temp location
def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        return temp_file.name

# Main Streamlit App
def main():
    # Title and description
    st.markdown("<h1 class='title'>Carbon Sequestration Estimation</h1>", unsafe_allow_html=True)
    st.markdown("Upload aerial and segmentation images to analyze vegetation health and carbon sequestration potential.")
    
    # Initialize session state variables if they don't exist
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'results' not in st.session_state:
        st.session_state.results = None

    # File Upload Section
    st.markdown("### Upload Required Files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        jp2_file = st.file_uploader("Upload JP2 Image File", type=["jp2"], key="jp2_uploader", 
                                    help="Upload your JP2 aerial or satellite image")
    
    with col2:
        tif_file = st.file_uploader("Upload TIF Segmentation File", type=["tif"], key="tif_uploader",
                                    help="Upload your TIF segmentation mask file")
    
    # Display original image if uploaded (regardless of processing)
    if jp2_file is not None:
        st.markdown("### Input JP2 Image")
        try:
            # For JP2 files, read with rasterio and convert for display
            jp2_temp_path = save_uploaded_file(jp2_file)
            with rio.open(jp2_temp_path) as src:
                jp2_img = src.read()
                jp2_img = np.transpose(jp2_img, (1, 2, 0))
                if jp2_img.shape[2] > 3:  # If more than 3 bands, take first 3
                    jp2_img = jp2_img[:, :, :3]

            # Normalize for visualization
            percentiles = np.percentile(jp2_img, [2, 98])
            img_display = np.clip(jp2_img, percentiles[0], percentiles[1])
            img_display = (img_display - percentiles[0]) / (percentiles[1] - percentiles[0])
            img_display = (img_display * 255).astype(np.uint8)

            st.image(img_display, caption="Original JP2 Image", use_container_width=True)
            os.unlink(jp2_temp_path)
        except Exception as e:
            st.error(f"Error displaying JP2 image: {str(e)}")
            st.image("https://via.placeholder.com/800x400.png?text=Error+Loading+JP2+Image")

    # Process button - only show if both files are uploaded
    if jp2_file is not None and tif_file is not None:
        # Display editable species table
        species_data = display_editable_species_table()
        
        # Single Process button below the table
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Process"):
                with st.spinner("Processing images... This may take a moment."):
                    try:
                        # Save uploaded files to temporary locations
                        jp2_temp_path = save_uploaded_file(jp2_file)
                        tif_temp_path = save_uploaded_file(tif_file)

                        # Process the images with updated species data
                        st.session_state.results = process_image(jp2_temp_path, tif_temp_path)
                        st.session_state.processed = True

                        # Clean up temporary files
                        os.unlink(jp2_temp_path)
                        os.unlink(tif_temp_path)

                        # Force refresh
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing images: {str(e)}")

    # Display results after processing
    if st.session_state.processed and st.session_state.results is not None:
        results = st.session_state.results
        
        # ---- Carbon Sequestration Report - Modern Card Design ----
        st.markdown("""
        <style>
            .report-card {
                border-radius: 12px;
                padding: 2rem;
                background: linear-gradient(135deg, #f5f7fa 0%, #e4f0f8 100%);
                box-shadow: 0 4px 20px rgba(0,0,0,0.08);
                border-left: 6px solid #2ecc71;
                margin-bottom: 2rem;
            }
            .metric-card {
                background: white;
                border-radius: 8px;
                padding: 1rem;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                margin: 0.5rem 0;
            }
            .metric-title {
                font-size: 0.9rem;
                color: #7f8c8d;
                margin-bottom: 0.3rem;
            }
            .metric-value {
                font-size: 1.5rem;
                font-weight: 700;
                color: #2c3e50;
            }
            .carbon-highlight {
                font-size: 2rem;
                color: #27ae60;
                font-weight: 800;
            }
            .info-button {
                position: absolute;
                top: 10px;
                right: 10px;
                background: #3498db;
                color: white;
                border: none;
                border-radius: 50%;
                width: 24px;
                height: 24px;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                font-size: 0.8rem;
            }
            .info-tooltip {
                display: none;
                position: absolute;
                right: 0;
                top: 30px;
                background: white;
                padding: 1rem;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                width: 300px;
                z-index: 100;
            }
            .info-button:hover + .info-tooltip {
                display: block;
            }
            .csi-scale {
                height: 12px;
                background: linear-gradient(90deg, #e74c3c, #f39c12, #f1c40f, #2ecc71, #27ae60);
                border-radius: 6px;
                margin: 0.5rem 0;
                position: relative;
            }
            .csi-marker {
                position: absolute;
                top: -5px;
                width: 2px;
                height: 22px;
                background: #2c3e50;
            }
            .csi-labels {
                display: flex;
                justify-content: space-between;
                font-size: 0.7rem;
                color: #7f8c8d;
            }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown("### Carbon Sequestration Report")
        
        # Main report card with info button
        st.markdown("""
        <div style="position: relative;">
            <div class="report-card">
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
                    <div class="metric-card">
                        <div class="metric-title">Vegetated Area</div>
                        <div class="metric-value">{vegetated_area}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Sequestration Rate (Vegetated)</div>
                        <div class="metric-value">{areal_rate_veg}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Sequestration Rate (Total Region)</div>
                        <div class="metric-value">{areal_rate_total}</div>
                    </div>
                    <div class="metric-card" style="grid-column: span 2;">
                        <div class="metric-title">Total Annual Carbon Sequestration</div>
                        <div class="carbon-highlight">{total_sequestration}</div>
                    </div>
                </div>
                <div style="margin-top: 1.5rem;">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <div style="font-weight: 600; color: #2c3e50;">Carbon Sequestration Index</div>
                        <div style="font-size: 1.2rem; font-weight: 700; color: #27ae60;">{csi}</div>
                    </div>
                    <div class="csi-scale">
                        <div class="csi-marker" style="left: calc({csi_percent}% * 0.9);"></div>
                    </div>
                    <div class="csi-labels">
                        <div>0.0</div>
                        <div>1.0</div>
                        <div>2.0</div>
                        <div>3.0</div>
                        <div>4.0</div>
                        <div>5.0+</div>
                    </div>
                </div>
            </div>
        </div>
        """.format(
            vegetated_area=results['results']["Vegetated Area"],
            areal_rate_veg=results['results']["Areal Rate (Vegetated)"],
            areal_rate_total=results['results']["Areal Rate (Total Region)"],
            total_sequestration=results['results']["Total Annual Sequestration"],
            csi=results['results']["Carbon Sequestration Index (CSI)"],
            csi_percent=float(results['results']["Carbon Sequestration Index (CSI)"].split()[0]) * 20
        ), unsafe_allow_html=True)

        # Separate the info button and tooltip to avoid HTML rendering issues
        st.markdown("""
        <div style="position: relative;">
            <button class="info-button" title="Interpretation Guide">i</button>
            <div class="info-tooltip">
                <h4 style="margin-top: 0; color: #2c3e50;">CSI Interpretation Guide</h4>
                <div style="display: flex; flex-direction: column; gap: 0.5rem;">
                    <div style="display: flex; align-items: center;">
                        <div style="width: 12px; height: 12px; background: #e74c3c; border-radius: 2px; margin-right: 0.5rem;"></div>
                        <div>0.0-0.5: Degraded/Bare land</div>
                    </div>
                    <div style="display: flex; align-items: center;">
                        <div style="width: 12px; height: 12px; background: #f39c12; border-radius: 2px; margin-right: 0.5rem;"></div>
                        <div>0.5-1.5: Grasslands/Sparse vegetation</div>
                    </div>
                    <div style="display: flex; align-items: center;">
                        <div style="width: 12px; height: 12px; background: #f1c40f; border-radius: 2px; margin-right: 0.5rem;"></div>
                        <div>1.5-2.5: Shrublands/Young plantations</div>
                    </div>
                    <div style="display: flex; align-items: center;">
                        <div style="width: 12px; height: 12px; background: #2ecc71; border-radius: 2px; margin-right: 0.5rem;"></div>
                        <div>2.5-3.5: Established forests</div>
                    </div>
                    <div style="display: flex; align-items: center;">
                        <div style="width: 12px; height: 12px; background: #27ae60; border-radius: 2px; margin-right: 0.5rem;"></div>
                        <div>3.5-4.5: Mature native forests</div>
                    </div>
                    <div style="display: flex; align-items: center;">
                        <div style="width: 12px; height: 12px; background: #16a085; border-radius: 2px; margin-right: 0.5rem;"></div>
                        <div>>4.5: Exceptional carbon sinks</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # First row
        col1, col2 = st.columns(2)
        with col1:
            st.image(results['vegetation_mask'], caption="Detectree Vegetation Detection", use_container_width=True)
        with col2:
            st.image(results['segmentation_mask'], caption="Segmentation Mask (from TIF file)", use_container_width=True)
        
        # Second row
        col1, col2 = st.columns(2)
        with col1:
            st.image(results['ndvi_image'], caption="NDVI (Normalized Difference Vegetation Index)", use_container_width=True)
        with col2:
            st.image(results['carbon_map'], caption=f"Carbon Sequestration Map (Total: {results['total_carbon']/1000:.1f} tCO₂/yr)", use_container_width=True)
        
        # Add download buttons for results
        st.markdown("### Download Results")
        
        # Convert results to downloadable formats
        buf = python_io.BytesIO()
        Image.fromarray(results['ndvi_image']).save(buf, format="PNG")
        ndvi_bytes = buf.getvalue()
        
        buf = python_io.BytesIO()
        Image.fromarray(results['carbon_map']).save(buf, format="PNG")
        carbon_bytes = buf.getvalue()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="Download NDVI Map",
                data=ndvi_bytes,
                file_name="ndvi_map.png",
                mime="image/png"
            )
        
        with col2:
            st.download_button(
                label="Download Carbon Map",
                data=carbon_bytes,
                file_name="carbon_map.png",
                mime="image/png"
            )
        
        # Generate a CSV with results
        results_df = pd.DataFrame([results['results']])
        csv = results_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="Download Report as CSV",
            data=csv,
            file_name="carbon_sequestration_report.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()