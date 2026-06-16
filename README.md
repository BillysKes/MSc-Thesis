# Translating Chest X-Ray Explainability into Uncertainty-Aware Radiology Reports

## 1. Project Overview
Deep learning models achieve strong predictive performance in chest X-ray interpretation, but their clinical utility is heavily limited by opaque decision-making, unreliable spatial localization, and the tendency of Generative AI (LLMs) to hallucinate unsupported medical claims. 

This repository contains the code and methodology for an end-to-end, explainability-driven framework that elevates post-hoc explainability (XAI) from a qualitative visualization into a calibrated, decision-aware engine for safe radiology report generation. 

By combining a **Vision Transformer (ViT)** classifier with a multi-method saliency fusion strategy, this system strictly maps pixel-level evidence to a 6-zone anatomical grid. It then stratifies model confidence into calibrated clinical decision zones, which act as strict guardrails for a Large Language Model (**gpt-4o-mini**), enforcing silence on negative findings and dictating appropriately hedged vocabulary for uncertain cases.

## 2. Architecture & Pipeline

### Phase 1: Multi-Label Classification
* **Model:** A Vision Transformer (`vit_large_patch14_clip_336`) initialized with pretrained weights and adapted for multi-label classification of thoracic pathologies.
* **Training:** Trained on the **CheXpert** dataset using Asymmetric Loss (ASL) to handle severe class imbalance, optimized via AdamW with Automatic Mixed Precision (AMP). 

### Phase 2: Explainability Fusion & Anatomical Grounding
* **Multi-Method Saliency:** Generates heatmaps using Grad-CAM++, GradientSHAP, and Occlusion sensitivity.
* **Anatomical Grid:** Saliency signals are aggregated into a 6-zone anatomical representation (Left/Right $\times$ Upper/Mid/Lower lung).
* **Fusion Strategy ($Mean Top-1^{2}$):** Introduces a novel, pathology-aware squared fusion strategy that exponentially rewards the most reliable saliency methods, suppressing noisy artifacts.

### Phase 3: Decision-Zone Modeling
Model probabilities are calibrated via temperature scaling and stratified into four clinically meaningful decision zones validated via population-level likelihood ratios ($LR^+$):
1. **Zone 0 (Rule-out):** Model classifies the observation as negative; the report remains silent.
2. **Zone 1a (Indeterminate):** Triggers safely hedged uncertainty language (e.g., "cannot exclude").
3. **Zone 1b (Suggestive):** Uses cautious reporting language (e.g., "probable").
4. **Zone 2 (Rule-in):** Permits conclusive, assertive phrasing.

### Phase 4: Constrained LLM Report Generation
The extracted anatomical zones and decision-zone confidence bands are fed into a strictly constrained prompt template. The LLM (`gpt-4o-mini`) is forced to rely *only* on the provided structured evidence, effectively mitigating hallucinated findings and overconfident language.

## 3. Evaluation & Key Results

The framework was rigorously evaluated using the **CheXlocalize** dataset (radiologist-annotated polygons) for spatial localization and **CheXbert** for report-level NLP fidelity.

### Localization Performance
* The $Mean Top-1^{2}$ fusion strategy successfully retrieved **~80% of the absolute maximum achievable ground-truth abnormality mass** using only the Top-1 predicted anatomical zone.
* Expanding to the Top-2 predicted zones captured **~85%** of the achievable limit, proving that pathology-aware fusion yields highly robust spatial grounding.

### Report Generation & Clinical Safety
* **Reclaiming Sensitivity Through Safety:** Strict NLP evaluators (like CheXbert) often penalize safely hedged language (Zone 1a) as false positives. However, treating these indeterminate predictions as diagnostic safety-nets recovers high effective sensitivity.
* **Mitigating Automation Bias:** By preventing the LLM from outputting hallucinated certainties on borderline radiographs, the system prioritizes clinical safety over narrative completeness, effectively bridging visual evidence with trustworthy text generation.


## 4. Read the Full Thesis
For a comprehensive breakdown of the threshold derivations, likelihood-ratio calibrations, and per-pathology performance metrics, please refer to the [MSc_Thesis.pdf](MSc_Thesis.pdf) included in this repository.
https://repository.ihu.gr/items/969cb98a-ca0c-448a-8568-ffe8cdb17726
