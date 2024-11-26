# Clustering Countries Using Socio-Economic and Health Indicators

This repository contains the analysis and code for the **2nd Homework Assignment** of the "Clustering Algorithms" graduate course of the MSc Data Science & Information Technologies Master's programme (Bioinformatics - Biomedical Data Science Specialization) of the Department of Informatics and Telecommunications department of the National and Kapodistrian University of Athens (NKUA), under the supervision of professor Konstantinos Koutroumbas, in the academic year 2023-2024.

The assignment includes an implementation of clustering algorithms applied to socio-economic and health data from 167 countries. The analysis uses features such as GDP per capita, life expectancy, child mortality, and more, to identify meaningful clusters of countries based on their development levels. The project is implemented in MATLAB and focuses on understanding how various socio-economic and health indicators contribute to global disparities.

---

## Overview

The main goal of this project is to apply clustering techniques to categorize countries into distinct groups based on their socio-economic and health indicators. The workflow includes data preprocessing, feature selection, normalization, and the application of clustering algorithms such as K-Means and K-Medians. Additionally, the project evaluates the quality of clustering using metrics like the Silhouette Score and Calinski-Harabasz Index, and provides visualization of the results.

---

## Main Workflow and Tools

1. **Data Exploration ("Feeling the Data")**
   - Analysis of individual features for range, distribution, and correlation.
   - Identification and handling of missing values.
   - Normalization using Z-score and Min-Max scaling.
2. **Feature Selection and Transformation**
   - Principal Component Analysis (PCA) and correlation analysis to select significant features.
   - Two experiments conducted:
      - Experiment A: GDPP, Inflation, Total Fertility, and Life Expectancy (PCA-based).
      - Experiment B: GDPP, Inflation, Total Fertility, Health, and Life Expectancy (Correlation-based).
3. **Clustering Algorithm Selection**
   - K-Means and K-Medians algorithms are used due to their efficiency in grouping compact clusters.
4. **Execution of Clustering Algorithms**
   - Determining the optimal number of clusters using methods like the Elbow Method and Silhouette Analysis.
   - Clustering experiments for 3-5 clusters and evaluating results.
5. **Cluster Characterization**
   - Analysis of cluster properties using descriptive statistics and visualizations.
   - Identification of the most contributing features to the clustering process.
6. **Visualization**
   - Heatmaps for correlation coefficients.
   - Histograms and boxplots for feature distributions.
   - Silhouette plots and elbow curves for cluster quality.

---

## Results Overview
- Clustering revealed meaningful groupings of countries based on socio-economic and health metrics.
- Experiment A showed more compact clusters using fewer features, while Experiment B provided deeper insights with additional features.
- The optimal number of clusters was determined to be 3, aligning with common economic groupings such as developed, developing, and least developed countries.

---

## Cloning the Repository
To clone this repository, run the following command in your terminal:
```bash
git clone https://github.com/GiatrasKon/Clustering-Countries-Socioeconomic-Health-Analysis.git
```

## Prerequisites

- MATLAB: Core programming environment used for all computations and visualizations.
   - MATLAB Toolboxes:
      - Statistics and Machine Learning Toolbox
      - Optimization Toolbox
- Dataset: [Kaggle - Unsupervised Learning on Country Data](https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data).

## Running the Code

1. Open MATLAB and navigate to the repository folder.
2. Ensure that the `data_country.mat`, `Country-data.csv` and `data-dictionary.csv` are in the `data/` directory before running the script.
3. Run the `cluster_countries.m` script to process the dataset and generate output.
4. Follow the inline comments in the scripts to customize parameters like the number of clusters and seed numbers.

## Documentation

Refer to the `documents` directory for the assignment description and report.

---