# Optimization of Concrete Mix Design Using AI-Driven Technology

An end-to-end Machine Learning and Optimization framework designed to predict the compressive strength of concrete mixes and discover eco-friendly, cost-effective mix proportions using Artificial Neural Networks (ANN), Support Vector Regression (SVR), Gradient Boosting (GBR), and Evolutionary Algorithms.

---

## 📌 Project Overview

Concrete mix design traditionally relies on empirical trials and extensive laboratory testing to achieve target compressive strengths. This project leverages AI and optimization algorithms to:
1. **Predict Compressive Strength (CS)** based on constituent materials and curing age.
2. **Optimize Mix Proportions** to minimize cement consumption while satisfying structural strength requirements and physical constraints (e.g., water-cement ratio).

---

## 📊 Dataset Description

The dataset (`experimental_dataset.csv`) consists of experimental concrete batch samples with the following parameters:

| Column Name | Description | Unit |
| :--- | :--- | :--- |
| `cement` | Mass of Cement | $\text{kg/m}^3$ |
| `sand` | Fine Aggregate Content | $\text{kg/m}^3$ |
| `granite` | Coarse Aggregate Content | $\text{kg/m}^3$ |
| `water` | Water Content | $\text{kg/m}^3$ |
| `wc` | Water-Cement Ratio ($w/c$) | Dimensionless |
| `age` | Curing Age | Days ($7, 14, 21, 28$) |
| `cs` | **Target:** Compressive Strength | MPa |

---

## 🤖 Predictive Models & Performance

Three machine learning regression models were trained and evaluated on an $80/20$ train-test split:
- **Artificial Neural Network (ANN):** Multi-Layer Perceptron (`MLPRegressor`) with feature scaling.
- **Support Vector Regressor (SVR):** Radial Basis Function (RBF) kernel.
- **Gradient Boosting Regressor (GBR):** Ensemble decision trees.

### Benchmark Results

| Model | $R^2$ Score | RMSE (MPa) | MAE (MPa) |
| :--- | :---: | :---: | :---: |
| **Artificial Neural Network (ANN)** | **0.9632** | **1.3103** | **1.0062** |
| **Support Vector Regressor (SVR)** | **0.9001** | **2.1589** | **1.4768** |
| **Gradient Boosting Regressor (GBR)** | **0.8681** | **2.4807** | **1.9467** |

---

## ⚙️ Mix Optimization Engine

To optimize mix design for sustainability and economy, the project integrates **Differential Evolution (Genetic Optimization)**. The optimization objective minimizes cement mass subject to non-linear physical constraints:

$$\text{Minimize } f(\vec{x}) = \text{Cement Mass} + \text{Penalty}(\text{Strength Deficit}) + \text{Penalty}(w/c \text{ Inconsistency})$$

### Sample Optimization Output (Target: $30\text{ MPa}$ at $28\text{ Days}$)
- **Predicted Compressive Strength:** $30.00\text{ MPa}$
- **Cement:** $320.00\text{ kg/m}^3$
- **Fine Aggregate (Sand):** $680.00\text{ kg/m}^3$
- **Coarse Aggregate (Granite):** $1180.00\text{ kg/m}^3$
- **Water:** $160.00\text{ kg/m}^3$
- **$w/c$ Ratio:** 0.50

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed along with the following packages:

```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn
