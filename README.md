# NETRA RAKSHAK: AUTONOMOUS RETINAL DIAGNOSTIC SUITE AND TELEMEDICINE CDSS

Multi-Stage Explainable AI Architecture, Deterministic Retinal Biomarker Segmentation, and Discrete-Event Telemedicine Resource Optimization for Diabetic Retinopathy Screening.

Developed under the MathWorks Smart India Hackathon Track (Problem Statement 26038).

---

## TABLE OF CONTENTS

1. [Executive Summary](#1-executive-summary)
2. [Clinical Background and Problem Statement](#2-clinical-background-and-problem-statement)
3. [System Architecture](#3-system-architecture)
4. [Core Technical Modules](#4-core-technical-modules)
   - [Phase 1: Ingestion and Image Quality Assessment (IQA)](#phase-1-ingestion-and-image-quality-assessment-iqa)
   - [Phase 2: Deterministic Retinal Structure and Biomarker Segmentation](#phase-2-deterministic-retinal-structure-and-biomarker-segmentation)
   - [Phase 3: Class-Balanced Deep Convolutional Grading](#phase-3-class-balanced-deep-convolutional-grading)
   - [Phase 4: Explainable AI via Grad-CAM Activation Mapping](#phase-4-explainable-ai-via-grad-cam-activation-mapping)
   - [Phase 5: Discrete-Event Telemedicine Simulation and Dispatch](#phase-5-discrete-event-telemedicine-simulation-and-dispatch)
5. [Clinical Validation and Multi-Cohort Benchmarking](#5-clinical-validation-and-multi-cohort-benchmarking)
   - [APTOS 2019 Calibration and Training Progression](#aptos-2019-calibration-and-training-progression)
   - [Referable Diabetic Retinopathy Clinical Triage Metrics](#referable-diabetic-retinopathy-clinical-triage-metrics)
   - [DRIVE Dataset Ground-Truth Benchmark](#drive-dataset-ground-truth-benchmark)
   - [Messidor-2 Out-of-Distribution Multi-Patient Audit](#messidor-2-out-of-distribution-multi-patient-audit)
6. [Master Clinical Diagnostic Station](#6-master-clinical-diagnostic-station)
7. [Repository Structure](#7-repository-structure)
8. [Prerequisites and Execution Guide](#8-prerequisites-and-execution-guide)
9. [Clinical Safety and Governance](#9-clinical-safety-and-governance)
10. [Authors and Acknowledgments](#10-authors-and-acknowledgments)

---

## 1. EXECUTIVE SUMMARY

Netra Rakshak is an end-to-end computational ophthalmology and clinical decision support system (CDSS) engineered to deliver scalable, explainable, and resource-optimized Diabetic Retinopathy (DR) screening in resource-constrained healthcare environments, such as Primary Health Centres (PHCs) and rural district hospitals.

The platform integrates two complementary diagnostic engines:
- A deterministic computer vision pipeline that isolates physiological structures (vascular tree, optic disc) and pathological hallmarks (microaneurysms, blot hemorrhages, hard exudates) according to established ophthalmological criteria.
- A deep convolutional neural network (ResNet-50) with inverse-frequency class weighting to grade severity according to the International Clinical Diabetic Retinopathy (ICDR) scale (Grades 0 to 4), backed by Gradient-weighted Class Activation Mapping (Grad-CAM) to verify decision provenance.

Beyond classification, the system includes a discrete-event telemedicine operational engine modeled in MATLAB Simulink. The simulation demonstrates that triaging scans locally at the edge eliminates over 90% of cellular transmission bandwidth and relieves over 75% of tertiary ophthalmologist workload across an annual cohort of 100,000 patients.

---

## 2. CLINICAL BACKGROUND AND PROBLEM STATEMENT

Diabetic Retinopathy is one of the leading global causes of preventable adult visual impairment and blindness. In India, where the diabetic population exceeds 101 million individuals (ICMR-INDIAB Study), regular annual fundus screening is required to intercept sight-threatening microvascular complications before irreversible retinal detachment or macular edema occurs.

### Contemporary Telemedicine Bottlenecks

Traditional centralized tele-ophthalmology models suffer from three structural failures:
1. Bandwidth Saturation: Transmitting raw, high-resolution fundus images (8.5+ MB per uncompressed acquisition) across rural 2G/3G cellular uplinks (nominal throughput 1.5 Mbps) causes recurrent transmission dropouts and high operational latency.
2. Specialist Queue Collapse: Approximately 75% to 80% of rural screening cohorts present with no apparent retinopathy (Grade 0) or mild non-referable alterations (Grade 1). Routing all scans to district hospital ophthalmologists creates severe queue backlogs, delaying intervention for critical proliferative cases.
3. Black-Box Opacity: Generic deep learning classifiers output probability vectors without anatomical justification, leaving clinicians unable to audit why an image was categorized as diseased.

Netra Rakshak resolves these bottlenecks through an edge-computed triage model:
- Non-Referable Cases (Grades 0 and 1): Screened, documented, and discharged locally at the PHC with zero cellular uplink usage.
- Referable Cases (Grades 2, 3, and 4): Compact diagnostic packages (0.65 MB containing compressed scans, Grad-CAM overlays, and biomarker telemetry) are securely dispatched with automated priority tags (P1 Emergency, P2 Routine Specialist Review).

---

## 3. SYSTEM ARCHITECTURE

The Netra Rakshak framework operates through a standardized sequential pipeline:

```
[Patient Retinal Acquisition (Zeiss / Canon / Portable Fundus Camera)]
                                |
                                v
+-------------------------------------------------------------------+
| STAGE 1: AUTOMATED INGESTION & IMAGE QUALITY ASSESSMENT (IQA)    |
| - Laplacian edge variance focus scoring (Var(Laplacian) >= 8.0)  |
| - Dynamic illumination variance verification (StdDev >= 15.0)    |
| - Green-channel Rayleigh CLAHE and 3x3 median denoising          |
+-------------------------------------------------------------------+
        |                                           |
    [Failed IQA]                               [Passed IQA]
        |                                           |
        v                                           v
[Immediate PHC Recapture Notice]        +---------------------------+
                                        | Parallel Processing Split |
                                        +---------------------------+
                                         /                         \
                                        v                           v
+-------------------------------------------------+ +-------------------------------------------------+
| STAGE 2: DETERMINISTIC BIOMARKER SEGMENTATION   | | STAGE 3: CLASS-BALANCED DEEP GRADING            |
| - Spatial resolution standardization (800 px)   | | - ResNet-50 deep residual backbone              |
| - Circular Field-of-View (FoV) boundary mask    | | - Custom 5-class fully connected output layer   |
| - 12-angle linear directional morphology sweep  | | - Inverse-frequency class penalty weighting    |
| - Optic disc localization via Gaussian cross-map| | - Calibrated 5-stage softmax probability output |
| - Morphological bottom-hat MA / hemorrhage gate | +-------------------------------------------------+
| - Top-hat hard exudate segmentation             |                         |
| - Quantitative biomarker metrics (vessel density|                         v
|   percentage, exudate burden, lesion counts)    | +-------------------------------------------------+
+-------------------------------------------------+ | STAGE 4: CLINICAL EXPLAINABILITY (GRAD-CAM)     |
                        \                           | - Target bottleneck layer: activation_49_relu   |
                         \                          | - 2D spatial gradient activation heatmap        |
                          \                         | - Thermal jet overlay on retinal fundus base    |
                           \                        +-------------------------------------------------+
                            \                                       /
                             v                                     v
+-----------------------------------------------------------------------------------------------------+
| STAGE 5: UNIFIED MASTER DIAGNOSTIC STATION & SIMULINK TELEMEDICINE DISPATCH                          |
| - Synchronized 4-quadrant multi-modal inspection matrix                                             |
| - Non-Referable (Grades 0-1): Local archival, zero uplink bandwidth, scheduled annual re-screening  |
| - Referable (Grades 2-4): 0.65 MB compressed XAI package uplinked to District Specialist Console    |
| - Discrete-event queue management: P1 Emergency (Grade 4) vs. P2 Specialist Queue (Grades 2-3)      |
+-----------------------------------------------------------------------------------------------------+
```

---

## 4. CORE TECHNICAL MODULES

### Phase 1: Ingestion and Image Quality Assessment (IQA)
*Scripts: `step1_sort_data.m`, `step2_quality_check.m`*

Raw retinal fundus captures frequently exhibit motion artifacts, out-of-focus blur, uneven corneal illumination, and optical vignetting. The IQA engine implements an automated gating mechanism executed across CPU worker threads via the MATLAB Parallel Computing Toolbox (`parfor`).

Mathematical Formulations:
1. Focus Checking via Laplacian Variance:
   The greyscale representation is convolved with a 2D discrete Laplacian operator:
   $$\nabla^2 I = \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix} * I_{gray}$$
   Focus adequacy is quantified by computing the spatial variance:
   $$\sigma^2_{blur} = \mathrm{Var}(\nabla^2 I)$$
   Acquisitions satisfying $\sigma^2_{blur} < 8.0$ are rejected as ungradeable.

2. Illumination Uniformity:
   $$\sigma_{illum} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (I_{gray}(i) - \mu_{gray})^2}$$
   Acquisitions with $\sigma_{illum} < 15.0$ (severe underexposure or illumination washout) are flagged for immediate recapture.

3. Contrast-Limited Adaptive Histogram Equalization (CLAHE):
   The green spectral channel ($G$) provides maximum photometric absorption contrast between hemoglobin-rich lesions and retinal pigment epithelium:
   $$G_{enhanced} = \mathrm{adapthisteq}(G, \text{'ClipLimit'}, 0.02, \text{'Distribution'}, \text{'rayleigh'})$$
   Followed by a $3 \times 3$ median filter for speckle artifact removal:
   $$G_{final}(x, y) = \mathrm{median}\{G_{enhanced}(x+u, y+v) \mid u, v \in [-1, 1]\}$$

---

### Phase 2: Deterministic Retinal Structure and Biomarker Segmentation
*Script: `step7_lesion_segmentation.m`*

Unlike unconstrained neural networks, clinical practice requires objective measurements of vascular architecture and lesion burden.

1. Spatial Standardization and Dynamic FoV Isolation:
   Input images are standardized to a target width of 800 pixels to preserve physical scaling of morphological kernels. The circular Field of View is isolated via binary thresholding and hole filling, followed by a dynamic 5% radial morphological erosion:
   $$\mathrm{FoV}_{inner} = \mathrm{FoV}_{mask} \ominus S_{disk}(r = 0.05 \cdot \min(H, W))$$
   This suppresses false positive lesion boundaries near camera aperture borders.

2. Multi-Angle Directional Morphology for Vascular Tree Extraction:
   Retinal blood vessels are continuous linear structures oriented across multiple angular directions. Background non-uniformities are removed via morphological opening:
   $$I_{vessels} = \max(0, (1 - G) - ((1 - G) \circ S_{disk}(r=18)))$$
   Linear directional sweeps are computed across 12 radial orientations ($\theta \in [0^\circ, 165^\circ]$ in steps of $15^\circ$) using line structuring elements of length 13:
   $$V_{resp}(x, y) = \max_{\theta} \left( I_{vessels} \circ S_{line}(L=13, \theta) \right)$$
   Adaptive thresholding at the 87th percentile isolates the continuous vascular network.

3. Optic Disc Localization and Masking:
   The optic disc appears as a high-intensity, circular yellowish structure. Candidate localization uses red-green channel cross-correlation filtered with a large Gaussian kernel ($\sigma = 10$, window $40 \times 40$):
   $$M_{OD} = (R \cdot G) * \mathcal{G}_{\sigma=10}$$
   The global maximum $(x_{OD}, y_{OD})$ defines the disc center, and a radial exclusion zone of radius $r_{OD} = 0.07 \cdot \min(H, W)$ is constructed to suppress false-positive exudate detections.

4. Microaneurysm and Hemorrhage Segmentation:
   Microaneurysms (MAs) and blot hemorrhages appear as small, dark, focal vascular anomalies. A morphological bottom-hat transform is computed on the green channel:
   $$I_{bottomhat} = (G \bullet S_{disk}(r=6)) - G$$
   Candidate lesions must satisfy strict clinical gates:
   - Must reside outside the dilated vascular tree: $\sim (V_{binary} \oplus S_{disk}(r=3))$
   - Must reside outside the optic disc exclusion mask
   - Must reside inside $\mathrm{FoV}_{inner}$
   Connected components are analyzed for area ($A$) and eccentricity ($E$):
   - Genuine Microaneurysms: $3 \le A \le 35\text{ pixels}$ and $E < 0.85$ (circular focal outpouchings)
   - Blot Hemorrhages: $36 < A \le 500\text{ pixels}$ (larger pooling of extravasated blood)

5. Hard Exudate Segmentation:
   Hard exudates represent lipid leakages with sharp, bright borders. Morphological top-hat filtering isolates bright structures:
   $$I_{tophat} = G - (G \circ S_{disk}(r=7))$$
   Filtering candidates above an intensity threshold of 0.11 outside the optic disc yields the hard exudate mask.

6. Quantitative Clinical Biomarkers:
   - Vascular Density: $\frac{\sum V_{binary}}{\sum \mathrm{FoV}_{inner}} \times 100\%$ (Standard clinical range: 10.0% to 15.0%)
   - Exudate Burden: $\frac{\sum \mathrm{Exudate}_{mask}}{\sum \mathrm{FoV}_{inner}} \times 100\%$
   - Microaneurysm Count: Total isolated focal count
   - Hemorrhage Count: Total isolated blot lesion count

---

### Phase 3: Class-Balanced Deep Convolutional Grading
*Scripts: `step3_train_classifier.m`, `step5_train_weighted_classifier.m`*

Classification assigns each scan to one of five stages according to the International Clinical Diabetic Retinopathy (ICDR) scale:
- Grade 0: No Apparent Retinopathy
- Grade 1: Mild Non-Proliferative Diabetic Retinopathy (NPDR)
- Grade 2: Moderate Non-Proliferative Diabetic Retinopathy (NPDR)
- Grade 3: Severe Non-Proliferative Diabetic Retinopathy (NPDR)
- Grade 4: Proliferative Diabetic Retinopathy (PDR)

#### Inverse-Frequency Class Weighting
Medical fundus datasets exhibit severe class imbalance, with Grade 0 and Grade 2 overrepresented, while critical sight-threatening Grades 3 and 4 constitute small fractions of training samples. Standard cross-entropy loss causes gradient starvation for minority classes.

To resolve this, inverse-frequency penalty weights are integrated directly into the cross-entropy loss:
$$W_c = \frac{N_{total}}{K \cdot N_c}$$
Where:
- $N_{total}$ is the total number of training images
- $K = 5$ is the number of clinical diagnostic classes
- $N_c$ is the image count for class $c$

During backpropagation, misclassifying a Grade 3 or Grade 4 scan incurs a proportionally higher loss penalty, compelling the model to establish robust decision boundaries for severe retinopathy.

#### Network Configuration and Hyperparameters
- Base Architecture: ResNet-50 (pretrained on ImageNet, fine-tuned on enhanced retinal datasets)
- Learnable Head: Fully connected layer with 5 outputs (WeightLearnRateFactor = 10, BiasLearnRateFactor = 10)
- Optimizer: Stochastic Gradient Descent with Momentum (SGDM, momentum = 0.9)
- Mini-Batch Size: 64
- Initial Learning Rate: $1 \times 10^{-4}$
- Data Augmentation: Real-time random horizontal and vertical reflections (`RandXReflection`, `RandYReflection`)
- Hardware Optimization: Mini-batch streaming directly on GPU memory with background dispatching

---

### Phase 4: Explainable AI via Grad-CAM Activation Mapping
*Script: `step4_explainable_report.m`*

To provide transparent decision support for reviewing ophthalmologists, Netra Rakshak computes Gradient-weighted Class Activation Maps (Grad-CAM).

The class score $y^c$ for predicted class $c$ is differentiated with respect to feature activation maps $A^k$ of the final residual bottleneck convolution layer (`activation_49_relu`):
$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$

The neuron importance weights $\alpha_k^c$ perform a weighted combination of forward activation maps, passed through a Rectified Linear Unit (ReLU) to highlight features that positively correlate with the target retinopathy grade:
$$L_{\mathrm{Grad\text{-}CAM}}^c = \mathrm{ReLU} \left( \sum_k \alpha_k^c A^k \right)$$

The resulting 2D coarse localization map is bilinearly upsampled to $224 \times 224$ pixels and alpha-blended over the fundus scan with a thermal colormap (45% transparency overlay). This allows the clinician to instantly confirm whether the network focused on genuine pathology (e.g., neovascularization loops or exudate plaques) rather than camera artifacts.

---

### Phase 5: Discrete-Event Telemedicine Simulation and Dispatch
*Scripts: `step9_telemedicine_simulation.m`, `netra_rakshak_telemedicine.slx`*

To quantify the operational and economic feasibility of deploying Netra Rakshak across public health infrastructure, a discrete-event Monte Carlo simulation was implemented in MATLAB and Simulink.

#### Operational Simulation Specifications
- Annual Screening Cohort: 100,000 patients/year
- District Deployment Scope: 20 connected Primary Health Centres (PHCs)
- Operational Calendar: 250 working days/year (District screening load = 400 patients/day; 20 patients/PHC/day)
- Raw Acquisition Size: 8.5 MB per uncompressed fundus capture
- Compressed Diagnostic Packet Size: 0.65 MB (contains compressed scan, Grad-CAM heatmap, and JSON biomarker vector)
- Rural Uplink Throughput: Nominal 1.5 Mbps cellular connection ($\approx 0.178\text{ MB/s}$)
- Specialist Review Duration: 30.0 seconds per referable case
- Dedicated Telemedicine Shift: 6.0 hours/day per reviewing ophthalmologist
- Epidemiological Distribution (Indian Rural Tele-screening Context):
  - Ungradeable (re-capture at PHC): 8.0%
  - Non-Referable (Grades 0 and 1): 74.0%
  - Referable (Grades 2, 3, and 4): 18.0%

#### Simulink Model Architecture (`netra_rakshak_telemedicine.slx`)
The programmatic Simulink model tracks four structural dynamic blocks:
1. `Patient_Arrival_Rate`: Continuous stochastic arrival source (400 arrivals/day).
2. `Edge_AI_Triage_Filter`: Transfer gain representing the 18% referable triage filter.
3. `Uplink_Bandwidth_Latency`: Transport delay block modeling transmission latency over rural cellular channels.
4. `Doctor_Review_Hours`: Service rate gain converting patient throughput into specialist review hours.
5. `District_Queue_Scope`: Discrete-event state monitor recording longitudinal backlog queue depth.

#### Comparative Operational Telemetry (100,000 Patient Longitudinal Audit)

| Operational Metric | Naive Centralized Cloud (Baseline) | Netra Rakshak (Edge-XAI Triage) | Operational Benefit |
| :--- | :--- | :--- | :--- |
| Annual Network Data Transmitted | 850.0 GB | 11.7 GB | 98.6% Bandwidth Conserved |
| Daily Average Network Upload | 3,400.0 MB / day | 46.8 MB / day | Viable on 2G/3G Cellular Networks |
| Specialist Clinical Review Time | 3.33 hours / day | 0.60 hours / day | 82.0% Reduction in Doctor Burden |
| Specialist Capacity Required | 1.0 Full-Time Equivalent (FTE) | 0.1 Full-Time Equivalent (FTE) | 1 Specialist Supports 10 Districts |
| End-of-Year Unreviewed Case Backlog | 0 cases (at continuous capacity) | 0 cases (stable, zero accumulation) | System Collapse Completely Prevented |

---

## 5. CLINICAL VALIDATION AND MULTI-COHORT BENCHMARKING

### APTOS 2019 Calibration and Training Progression
*Artifact: `untitled.png`*

Model training was conducted on the enhanced APTOS 2019 Blindness Detection dataset using an NVIDIA GPU. 
- Total Epochs: 8
- Total Iterations: 592 (74 iterations/epoch)
- Validation Frequency: Every 30 iterations
- Final Validation Accuracy: 84.47% across the full 5-stage ICDR grading task
- Total Convergence Time: 4 minutes 15 seconds

The convergence profile demonstrates smooth loss reduction from an initial value of 2.2 down to 0.43, with validation accuracy stabilizing above 82% from epoch 4 onward.

---

### Referable Diabetic Retinopathy Clinical Triage Metrics
*Artifact: `Netra Rakshak - Clinical Benchmark Report.png`*
*Script: `step6_clinical_metrics.m`*

In public screening programs, the primary clinical threshold is distinguishing between Non-Referable DR (Grades 0 and 1, managed with annual routine follow-ups) and Referable DR (Grades 2, 3, and 4, requiring specialist ophthalmic intervention to prevent blindness).

Validation was conducted on an independent 20% randomized split:

#### Referable DR Triage Matrix (Clinical Evaluation Cohort)
- True Negatives (TN, correctly identified Non-Referable): 404
- False Positives (FP, non-referable flagged as referable): 8
- False Negatives (FN, referable missed by system): 16
- True Positives (TP, correctly identified Referable): 171

#### Quantitative Metric Breakdown

| Clinical Metric | Regulatory Screening Target | Netra Rakshak Observed | Clinical Compliance Status |
| :--- | :--- | :--- | :--- |
| Sensitivity (Recall, TP / [TP + FN]) | Greater than 90.00% | 91.44% | PASSED |
| Specificity (TN / [TN + FP]) | Greater than 85.00% | 98.06% | PASSED |
| Triage Diagnostic Accuracy | - | 95.99% | Excellent |
| Positive Predictive Value (Precision) | - | 95.53% | High Clinical Reliability |
| F1-Score | - | 93.44% | Robust Diagnostic Agreement |

The observed sensitivity of 91.44% satisfies the clinical screening benchmark established by the British Diabetic Association and the American Academy of Ophthalmology, while the specificity of 98.06% prevents overwhelming tertiary eye hospitals with false-positive referrals.

---

### DRIVE Dataset Ground-Truth Benchmark
*Script: `step8_benchmark_drive.m`*

To objectively validate the deterministic vascular extraction algorithm, Netra Rakshak was benchmarked against expert manual segmentations from the international DRIVE (Digital Retinal Images for Vessel Extraction) cohort.

Evaluation was performed strictly within the circular Field of View mask:
- Metric Parameters: Dice Similarity Coefficient (F1 score), Jaccard Index (Intersection over Union), Vessel Sensitivity, Vessel Specificity, and Global Pixel Accuracy.
- Directional morphology correctly extracts main vascular arcades and secondary arteriolar branches while filtering out non-vascular background lesions.

---

### Messidor-2 Out-of-Distribution Multi-Patient Audit
*Artifact: `Netra Rakshak - Messidor-2 Multi-Patient Audit Gallery.png`*
*Data Log: `messidor2_triage_audit.csv`*
*Script: `step10_benchmark_messidor2.m`*

To evaluate generalizability to external clinical populations acquired with disparate fundus cameras, the model was audited on the Messidor-2 multi-center cohort without fine-tuning.

The automated audit script processes scans with on-the-fly green-channel CLAHE, calculates 5-stage probability distributions, generates multi-patient Grad-CAM galleries for top referable cases, and exports a standardized clinical log (`messidor2_triage_audit.csv`) tracking:
- Image Filename
- Predicted Grade (0 to 4)
- Model Confidence Percentage
- Triage Status (Non-Referable vs. Referable)

---

## 6. MASTER CLINICAL DIAGNOSTIC STATION
*Scripts: `step11_master_dashboard.m`, `NetraRakshak.mlapp`*

Netra Rakshak provides a unified Graphical User Interface designed in MATLAB App Designer and `uifigure` using a dark medical slate design system (`[0.07, 0.09, 0.15]`).

### UI Architecture and Functional Layout
1. Header Status Bar: Displays real-time GPU engine status, pipeline versioning, and project metadata.
2. Ingestion and IQA Control Panel:
   - File browser supporting PNG, JPG, JPEG, and TIFF fundus images.
   - Quick-load dropdown menu populated with clinical reference cases across all five severity tiers.
   - Automated Image Quality Assessment readout: Sharpness index, illumination balance, and automated quality decision (Passed vs. Recapture Required).
3. 4-Quadrant Visual Multi-Modal Inspection Matrix:
   - Quadrant 1 (Top-Left): Primary Optical Acquisition (normalized raw fundus capture).
   - Quadrant 2 (Top-Right): Rayleigh Green-Channel CLAHE (enhanced microvascular contrast).
   - Quadrant 3 (Bottom-Left): Grad-CAM Saliency Map (thermal jet heatmap highlighting pathological focus).
   - Quadrant 4 (Bottom-Right): Deterministic Retinal Biomarker Segmentation (composite overlay displaying cyan vessels, red microaneurysms/hemorrhages, and yellow hard exudates).
4. Severity Rating and Triage Ledger:
   - Prominent ICDR severity rating badge with color coding (Green: Level 0, Sky Blue: Level 1, Amber: Level 2, Dark Orange: Level 3, Crimson Red: Level 4).
   - Calibrated model confidence score and full 5-stage probability distribution bar chart.
5. Deterministic Biomarkers Summary Card:
   - Sub-pixel microaneurysm count
   - Blot hemorrhage count
   - Hard exudate burden percentage
   - Vascular density percentage (relative to clinical 10-15% target range)
6. Telemedicine Dispatch Telemetry Panel:
   - Transmission Action: Local PHC archive vs. District Hospital uplink dispatch.
   - Uplink Payload Size: 0.00 MB for local clearance; 0.65 MB for referable cases.
   - Estimated cellular upload time over 1.5 Mbps connection.
   - Prioritized Doctor Queue Tagging:
     - Level 4: P1 - Emergency Ophthalmic Review
     - Level 2-3: P2 - Routine Specialist Queue
     - Level 0-1: P3 - Routine Annual Re-Screen

---

## 7. REPOSITORY STRUCTURE

```
Netra_Rakshak_Model/
|
|-- step1_sort_data.m                  # Automated dataset sorting into ICDR class folders
|-- step2_quality_check.m              # Parallel IQA engine (Laplacian blur + illumination gating)
|-- step3_train_classifier.m           # Deep transfer learning classifier training (ResNet-50)
|-- step4_explainable_report.m         # Grad-CAM clinical explainability and visual report generator
|-- step5_train_weighted_classifier.m  # Inverse-frequency class-weighted training pipeline
|-- step6_clinical_metrics.m           # Clinical validation metrics, confusion charts, sensitivity
|-- step7_lesion_segmentation.m        # Deterministic morphological segmentation (vessels, lesions, OD)
|-- step8_benchmark_drive.m            # External gold-standard benchmark against DRIVE cohort
|-- step9_telemedicine_simulation.m    # Discrete-event Monte Carlo telemedicine district simulation
|-- step10_benchmark_messidor2.m       # External cohort audit and multi-patient XAI gallery generator
|-- step11_master_dashboard.m          # Programmatic Master Clinical Decision Support System station
|
|-- NetraRakshak.mlapp                 # MATLAB App Designer graphical application
|-- netra_rakshak_telemedicine.slx     # Simulink district telemedicine operational model
|-- trained_dr_classifier.mat          # Calibrated class-weighted ResNet-50 network weights (Git LFS)
|-- netra_rakshak.onnx                 # Cross-platform ONNX exported model weights (Git LFS)
|-- messidor2_triage_audit.csv         # Standalone clinical triage audit output for Messidor-2
|
|-- Netra Rakshak - Clinical Benchmark Report.png              # Confusion matrices and triage charts
|-- Netra Rakshak - Messidor-2 Multi-Patient Audit Gallery.png # External validation Grad-CAM gallery
|-- untitled.png                                               # GPU training convergence curves
|
|-- .gitignore                         # Strict exclusion rules for datasets and cache
|-- .gitattributes                    # Git LFS tracking configuration for large model files
`-- README.md                          # Technical system documentation
```

---

## 8. PREREQUISITES AND EXECUTION GUIDE

### Software Prerequisites
- MATLAB R2022b or later (R2024a recommended)
- Required MATLAB Toolboxes:
  - Deep Learning Toolbox
  - Image Processing Toolbox
  - Computer Vision Toolbox
  - Parallel Computing Toolbox
  - Simulink
- Optional Hardware Acceleration:
  - NVIDIA GPU with CUDA Compute Capability 6.0 or higher
  - Dedicated VRAM: 6 GB or higher (for mini-batch sizes of 32 to 64)

### Quick Start Execution

1. Clone the Repository:
   ```bash
   git clone https://github.com/Lost-Alien/Netra-Rakshak-Model.git
   cd Netra-Rakshak-Model
   ```

2. Pull Large Model Weights via Git LFS:
   ```bash
   git lfs pull
   ```

3. Launch the Master Diagnostic Station:
   In the MATLAB command prompt, execute:
   ```matlab
   step11_master_dashboard
   ```
   Or open and run `NetraRakshak.mlapp` within MATLAB App Designer.

4. Run Clinical Benchmark and Evaluation Scripts:
   - To inspect explainable Grad-CAM reports on test fundus samples:
     ```matlab
     step4_explainable_report
     ```
   - To evaluate clinical metrics (Sensitivity, Specificity, Confusion Matrices):
     ```matlab
     step6_clinical_metrics
     ```
   - To execute the 100,000-patient district telemedicine operational simulation:
     ```matlab
     step9_telemedicine_simulation
     ```
   - To run deterministic biomarker segmentation on an arbitrary fundus scan:
     ```matlab
     step7_lesion_segmentation
     ```

---

## 9. CLINICAL SAFETY AND GOVERNANCE

Netra Rakshak is engineered as a Clinical Decision Support System (CDSS) intended to assist qualified healthcare professionals and ophthalmologists. It is designed to augment clinical throughput in screening camps and Primary Health Centres by stratifying high-risk patients.

Key Governance Safeguards:
- Human-in-the-Loop: All referable recommendations (Grades 2 to 4) require formal sign-off by a registered ophthalmologist via the specialist console.
- Transparent Provenance: Every classification output is paired with a Grad-CAM saliency overlay and quantitative biomarker telemetry (microaneurysm count, hemorrhage count, exudate burden), enabling immediate clinical verification.
- Conservative Quality Gate: Images failing optical focus or illumination standards are flagged for immediate recapture rather than passing degraded data to the neural network.

---

## 10. AUTHORS AND ACKNOWLEDGMENTS

### Core Engineering and Research
- **Dev Kumar Sharma** (Lead AI and Systems Engineer)
  - GitHub: [@Lost-Alien](https://github.com/Lost-Alien)
  - Project Portfolio: [Lost-Alien.github.io](https://lost-alien.github.io/)

### Project Initiative
- Smart India Hackathon (SIH) 2026
- Problem Statement: 26038 (MathWorks Track)
- Focus Area: Explainable AI and Telemedicine for Diabetic Retinopathy Screening

### Scientific Citations and Reference Standards
1. International Council of Ophthalmology (ICO), "ICO Guidelines for Diabetic Eye Care," 2018.
2. Porwal, P., et al., "Indian Diabetic Retinopathy Image Dataset (IDRiD): A Database for Diabetic Retinopathy Screening Research," *Data*, 2018.
3. Selvaraju, R. R., et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization," *IEEE International Conference on Computer Vision (ICCV)*, 2017.
4. Staal, J., et al., "Ridge-Based Vessel Segmentation in Color Images of the Retina," *IEEE Transactions on Medical Imaging*, 2004 (DRIVE Dataset).
5. Decenciere, E., et al., "Feedback on a publicly distributed database: the Messidor database," *Image Analysis & Stereology*, 2014.
6. Anjana, R. M., et al., "Prevalence of diabetes and prediabetes in 15 states of India: results from the ICMR-INDIAB population-based study," *The Lancet Diabetes & Endocrinology*, 2017.

---
*Netra Rakshak: Clinical Engineering for Preventable Blindness.*
