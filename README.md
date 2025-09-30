# 🫀 Heart Disease Prediction & Analysis Project

This project implements a full **machine learning pipeline** on the [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease).  
It covers **data preprocessing, dimensionality reduction, feature selection, supervised & unsupervised learning, hyperparameter tuning, and deployment** with a Streamlit web app.

---

---

##  Features

### 🔹 Data Preprocessing & Cleaning
- Handle missing values  
- Encode categorical features (OneHotEncoder)  
- Scale numerical features (MinMaxScaler)  
- Exploratory Data Analysis (EDA)  

### 🔹 PCA (Dimensionality Reduction)
- Reduce feature dimensionality while retaining variance  
- Determine optimal number of components  
- Visualize cumulative variance & PCA scatter plots  

### 🔹 Feature Selection
- Recursive Feature Elimination (RFE)  
- Chi-Square Test  
- Feature importance ranking (XGBoost / Random Forest)  

### 🔹 Supervised Learning
- Models: Logistic Regression, Decision Tree, Random Forest, SVM  
- Metrics: Accuracy, Precision, Recall, F1-score  
- ROC Curve & AUC comparison  

### 🔹 Unsupervised Learning
- K-Means Clustering (Elbow Method)  
- Hierarchical Clustering (Dendrograms)  

### 🔹 Hyperparameter Tuning
- GridSearchCV & RandomizedSearchCV  
- Compare baseline vs optimized models  


---

## ⚙️ Installation

Clone this repo:
```bash
git clone https://github.com/yourusername/Heart_Disease_Project.git
cd Heart_Disease_Project
```
Install dependencies:
```bash
pip install -r requirements.txt
```

Deliverables
✔️ Cleaned dataset
✔️ PCA-transformed dataset
✔️ Selected key features
✔️ Trained & tuned models
✔️ Evaluation metrics & visualizations
✔️ Exported model (.pkl)
✔️ Streamlit UI + Ngrok deployment
