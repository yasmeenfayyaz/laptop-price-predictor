# 💻 Laptop Price Predictor — End-to-End ML Project

An interactive Machine Learning web application that predicts laptop prices based on hardware specifications such as RAM, CPU brand, GPU, Storage (SSD/HDD), Display resolution, and Weight. Built using **Python, Scikit-Learn, and Streamlit**.

---

## 📌 Project Overview
Buying a laptop can be challenging due to varying price points across different brands and technical configurations. This project addresses the issue by utilizing a trained Machine Learning model on real-world laptop hardware specs to deliver dynamic, accurate price estimations in real time.

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Data Manipulation & Preprocessing:** Pandas, NumPy
* **Data Visualization:** Seaborn, Matplotlib
* **Machine Learning Models:** Scikit-Learn (Linear Regression, Random Forest, Decision Tree), XGBoost
* **Web Framework:** Streamlit
* **Deployment:** Streamlit Community Cloud

---

## 📊 Machine Learning Workflow

1. **Data Cleaning & Preprocessing:**
   * Parsed raw specifications to extract numerical values (e.g., RAM in GB, Weight in kg).
   * Separated storage types into distinct `SSD` and `HDD` features.
   * Handled CPU and GPU categories by grouping them into primary vendor classes.

2. **Feature Engineering:**
   * Extracted display metrics: Created `Touchscreen` and `Ips` binary features.
   * Computed **PPI (Pixels Per Inch)** using screen resolution and screen size to capture screen quality accurately.

3. **Model Training & Evaluation:**
   * Trained multiple algorithms, including Linear Regression, Random Forest, Gradient Boosting, and XGBoost.
   * Applied log transformation on target prices to handle right-skewed price distributions.
   * **Selected Model:** Picked the best-performing pipeline based on $R^2$ Score and Root Mean Squared Error (RMSE).

---

## 🚀 Live Demo & Repository Files

* 🌐 **Live Web App:** https://laptop-price-predictor-application.streamlit.app/

### Repository Structure:
```text
.
├── app.py           # Streamlit Web Application Interface
├── df.pkl           # Processed DataFrame for UI dropdown options
├── pipe.pkl         # Trained Machine Learning Pipeline
├── requirements.txt # Project Dependencies
└── README.md        # Project Documentation
