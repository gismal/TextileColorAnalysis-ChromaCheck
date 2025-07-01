# Thesis Project: Image Color Segmentation

## Overview

This project implements color segmentation using K-means, SOM, and DBSCAN, comparing test images to a reference for similarity.

## Problem Statement

Predict and compare color segments in images for applications in image analysis and matching.

## Dataset

- Source: Custom dataset with reference and test images.
- Location: `data/raw/`

## ML Pipeline

1. Preprocessing: Resize, apply filters.
2. Segmentation: K-means, SOM, DBSCAN with optimal and predefined clusters.
3. Evaluation: Calculate Delta E for similarity.

## Setup Instructions

1. Clone the repository: `git clone [repo_url]`
2. Install dependencies: `pip install -r environment/requirements.txt`
3. Run: `python src/main.py`

## Results

- Best method: K-means with optimal k, average Delta E = 5.2.
- See `reports/thesis_summary.pdf` for details.

## License

MIT License - see `LICENSE` for details.
