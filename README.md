# Carbon Sequestration Estimation Tool

## Project Overview

Welcome to the **Carbon Sequestration Estimation Tool**! This project, developed by a team of five for the 'AI to GLOW' Hackathon (Artificial Intelligence to Generate Limitless Opportunities WorldWide) hosted by the International Centre for Education and Research (ICER), VIT-Bangalore (March 26 - April 6, 2025), is an innovative application designed to estimate carbon sequestration from aerial imagery. We delved into an unexplored area of research, combining advanced image analysis with environmental modeling to create a tool that can help organizations, especially NGOs, quantitatively measure the impact of their tree planting and conservation efforts and set data-driven goals.

We are proud to have reached the **final round** of this highly competitive hackathon!

## Features

Our tool provides a comprehensive pipeline, from image processing to detailed carbon metrics and visualization, all accessible through an interactive Streamlit interface:

* **Image Ingestion & Pre-processing:** Handles high-resolution `.jp2` (JPEG2000) aerial images and `.tif` (GeoTIFF) segmentation masks.
* **Intelligent Vegetation Detection:** Leverages `detectree` and OpenCV for precise identification and masking of vegetated areas.
* **Vegetation Health Assessment (Pseudo-NDVI):** Calculates a pseudo-Normalized Difference Vegetation Index (NDVI) from RGB images to assess vegetation vigor.
* **Dynamic Species Modeling:** Utilizes a synthetic species data model with a Gaussian probability distribution to assign species likelihoods to pixels, enhancing carbon calculation realism.
* **Accurate Carbon Sequestration Estimation:** Estimates annual carbon sequestration at a pixel level, factoring in species probabilities, NDVI-adjusted biomass, and real-world efficiency factors, then converts it to CO₂ equivalent.
* **Carbon Sequestration Index (CSI):** Introduces a novel logarithmic index (inspired by the Richter scale) to intuitively compare carbon capture potential across different areas.
* **Interactive Streamlit Web Application:** Provides a user-friendly interface for image uploads, visualization of intermediate results (vegetation masks, NDVI, carbon maps), and downloadable reports.

## How It Works

The tool processes aerial images to:
1.  **Identify Vegetation:** Using computer vision techniques, it accurately outlines areas covered by plants.
2.  **Assess Health:** It analyzes the color information in the image to determine the health and density of the vegetation (using pseudo-NDVI).
3.  **Model Species:** Based on vegetation health, our model estimates the likely types of plants present in each area.
4.  **Calculate Carbon:** Using established scientific principles and our species model, it calculates the amount of carbon dioxide absorbed by the vegetation annually.
5.  **Report & Visualize:** All findings are presented in an easy-to-understand report with visual maps, showing where carbon is being sequestered most effectively.

## Getting Started

Follow these steps to get a copy of the project up and running on your local machine.

### Prerequisites

You'll need Python 3.8+ installed. The project relies on the following key Python libraries:

* `streamlit`
* `opencv-python`
* `numpy`
* `matplotlib`
* `scikit-image`
* `detectree`
* `rasterio`
* `Pillow`
* `pandas`

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/carbon-sequestration-hackathon.git](https://github.com/your-username/carbon-sequestration-hackathon.git)
    cd carbon-sequestration-hackathon
    ```
    (Note: Replace `https://github.com/your-username/carbon-sequestration-hackathon.git` with your actual repository URL and recommended name, e.g., `carbon-sequestration-hackathon`).

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    (You'll need to create a `requirements.txt` file manually if not already present. You can generate it using `pip freeze > requirements.txt` after installing all dependencies.)

    **Example `requirements.txt` content:**
    ```
    streamlit
    opencv-python
    numpy
    matplotlib
    scikit-image
    detectree
    rasterio
    Pillow
    pandas
    ```

### Usage

1.  **Run the Streamlit application:**
    ```bash
    streamlit run hackathon_proj.py
    ```
2.  Your web browser will automatically open to the Streamlit application.
3.  Upload your `.jp2` aerial image and corresponding `.tif` segmentation mask file.
4.  Adjust species coverage parameters if desired.
5.  Click the "Process" button to initiate the carbon sequestration estimation.
6.  View the results, including various maps and metrics, and download reports.
