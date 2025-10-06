# Supervised_Unsupervised_Learning_Example

## Overview

This project analyzes a vertebral column dataset with multiple machine learning techniques for dimensionality reduction, supervised classification, and unsupervised clustering. The goal is to classify vertebral conditions (Normal vs. Abnormal) and explore underlying data patterns.

The workflow includes:

Data preprocessing and scaling

Dimensionality reduction using PCA and Factor Analysis (FA)

Supervised classification using:

K-Nearest Neighbors (KNN)

Random Forest (RF)

Support Vector Machine (SVM)

Linear Discriminant Analysis (LDA)

Unsupervised clustering using:

Gaussian Mixture Models (GMM)

K-Means clustering

## 1. Dataset

The dataset contains 6 numerical features:

pelvic_incidence, pelvic_tilt, lumbar_lordosis_angle, sacral_slope, pelvic_radius, grade_of_spondylolisthesis

Target variable:

class (0: Normal, 1: Abnormal)

Data size: 310 samples

## 2. Preprocessing

Standardization: All features are scaled to mean ≈ 0 and standard deviation ≈ 1 using StandardScaler.

SMOTE: Applied to training data to address class imbalance in supervised learning.

## 3. Dimensionality Reduction
### 3.1 Principal Component Analysis (PCA)

PCA reduces data dimensions while preserving variance.

Scree plot and cumulative variance plot help determine optimal components.

Threshold: 90% explained variance → 4 components retained.

### 3.2 Factor Analysis (FA)

Extracts latent factors to represent correlations between features.

Varimax rotation applied to enhance interpretability.

Elbow method suggested using 3 factors.

## 4. Supervised Classification
### 4.1 K-Nearest Neighbors (KNN)

Weighted by distance (weights='distance') to handle imbalanced classes.

Optimal k determined using 5-fold cross-validation.

Models trained with:

Original features

PCA-reduced features

FA-reduced features

Decision boundaries visualized for both feature pairs and PCA/FA components.

### 4.2 Random Forest (RF)

Hyperparameters tuned via GridSearchCV.

Class imbalance handled via class_weight='balanced'.

Feature importance plotted to identify key contributors.

### 4.3 Support Vector Machine (SVM)

Hyperparameter tuning using GridSearchCV.

Class imbalance addressed using SMOTE.

Applied on original, PCA, and FA features.

### 4.4 Linear Discriminant Analysis (LDA)

Best hyperparameters selected with cross-validation and class weight adjustment.

Decision boundaries visualized in PCA and FA spaces.

Linear boundaries due to the nature of LDA.

### 4.5 Performance Summary
Model	Original Accuracy	PCA Accuracy	FA Accuracy
KNN	0.839	0.823	0.726
RF	0.790	0.790	0.677
SVM	0.823	0.839	0.661
LDA	0.855	0.839	0.758

KNN and LDA generally outperform other models.

Dimensionality reduction (PCA/FA) reduces complexity but may slightly affect accuracy.

## 5. Unsupervised Clustering
### 5.1 Gaussian Mixture Model (GMM)

Tested 2 and 3 clusters.

3 clusters best reflect underlying vertebral classes (Normal, Disk Hernia, Spondylolisthesis).

Cluster visualization in PCA space.

### 5.2 K-Means Clustering

Applied with 2 and 3 clusters.

Results visualized in PCA-reduced feature space.

### 5.3 Model Selection

BIC scores used to select optimal number of components for GMM.

## 6. Visualizations

PCA scatter plots for dimensionality reduction.

Scree and cumulative variance plots for PCA and FA.

Decision boundaries for KNN, LDA in PCA/FA spaces.

Feature importance bar plots for Random Forest.

Cluster plots for GMM and K-Means.

## 7. Key Libraries

pandas, numpy, matplotlib, seaborn

scikit-learn: PCA, FA, KNN, RF, SVM, LDA, StandardScaler

imblearn: SMOTE

factor_analyzer: FactorAnalysis

scikit-learn.mixture: GaussianMixture

## 8. Conclusions

KNN and LDA achieve the highest classification performance.

PCA preserves most of the variance and reduces dimensionality effectively.

FA captures latent factors but may reduce predictive accuracy slightly.

GMM with 3 clusters aligns well with true class distribution in unsupervised learning.

Supervised models benefit from resampling and careful hyperparameter tuning to address class imbalance.

