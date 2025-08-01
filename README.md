# Perspective Correction Pipeline

## Project Overview

This repository implements a pipeline for detecting and correcting perspective distortion in document images. Given a warped, skewed, or partially corrupted photo of a document, the system recovers a flattened, frontal view that aligns with the undistorted ground truth. The core idea combines classical computer vision methods—edge and line detection, robust model fitting, and geometric warping—to infer document boundaries and perform rectification. :contentReference[oaicite:15]{index=15}

## Key Components

### 1. Line Detection via Hough Transform  
Edge maps are processed to extract straight-line candidates with the Hough Transform. Peaks in accumulator space suggest potential document edges, but raw outputs are noisy and overcomplete, so later refinement is essential. :contentReference[oaicite:16]{index=16}

### 2. Robust Refinement with RANSAC  
The system applies RANSAC to filter and authenticate the line hypotheses from the Hough stage. By iteratively sampling data and evaluating consensus, it retains only those lines with strong support, reducing the influence of outliers and spurious detections. :contentReference[oaicite:17]{index=17}

### 3. Geometric Transformation  
Intersections of the refined lines produce candidate corners; these are processed (e.g., convex hull, polygon approximation) to extract a consistent quadrilateral representing the document. A homography is computed to map the warped quadrilateral to a canonical rectangle, and bilinear interpolation is used to resample the corrected image. :contentReference[oaicite:18]{index=18}

## Evaluation

Correction quality is quantified using the Structural Similarity Index (SSIM) against ground truth. The system was tested across several distortion types—curved, fold, incomplete, perspective, random, and rotate—with average SSIMs reflecting varying difficulty levels. :contentReference[oaicite:19]{index=19}

| Distortion Type | Average SSIM |
|-----------------|--------------|
| Curved          | 0.45         |
| Fold            | 0.41         |
| Incomplete      | 0.39         |
| Perspective     | 0.42         |
| Random          | 0.40         |
| Rotate          | 0.35         | :contentReference[oaicite:20]{index=20}

Visual examples illustrate both successful corrections and failure modes, highlighting the relative robustness across categories. :contentReference[oaicite:21]{index=21}

## Challenges & Limitations

- **Custom component sensitivity:** The homemade implementations of Hough and RANSAC exhibit sensitivity to noise and parameter choices, sometimes leading to missed or incorrect boundary estimation. :contentReference[oaicite:22]{index=22}  
- **Hyperparameter dependence:** Performance varies with thresholds and sampling strategies; lack of adaptive tuning can degrade results on heterogeneous inputs. :contentReference[oaicite:23]{index=23}  
- **Noisy / difficult distortions:** Categories like rotate, which introduce additional noise or degraded structures, show lower SSIM and more frequent quadrilateral extraction failures. 

## Results Summary

The pipeline consistently recovers document geometry in moderate to challenging distortion scenarios, with best results on curved and perspective warps and weaker outcomes on highly degraded or incomplete inputs. Overall SSIMs fall in the mid-0.3 to 0.5 range depending on category difficulty. :contentReference[oaicite:25]{index=25}

## Future Directions

- Integrate advanced denoising and edge enhancement to provide cleaner input for boundary estimation. 
- Replace or augment the custom Hough/RANSAC pipeline with optimized or learning-based alternatives for more stable detection. 
- Introduce automated hyperparameter tuning or adaptive parameter selection per image to improve consistency.


## Repository Structure (suggested)

