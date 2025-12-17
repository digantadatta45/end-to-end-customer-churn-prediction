# 📊 Customer Churn Prediction – End‑to‑End Streamlit App

A complete **end‑to‑end machine learning web application** that predicts whether a customer is likely to churn. The app covers the full ML lifecycle — data loading, preprocessing, model training, evaluation, and real‑time prediction — all deployed with **Streamlit Cloud**.

🔗 **Live App**: [https://end-to-end-customer-churn-prediction-hxxtveotkuepfrvnucfayn.streamlit.app/](https://end-to-end-customer-churn-prediction-hxxtveotkuepfrvnucfayn.streamlit.app/)

---

## 🚀 Overview

Customer churn is a critical business problem in subscription‑based industries. This project demonstrates how machine learning can be used to identify customers who are at risk of leaving, enabling businesses to take proactive retention actions.

The application is designed as a **single‑file, production‑ready Streamlit app**, making it easy to deploy, demo, and maintain.

---

## ✨ Features

* End‑to‑end ML pipeline (data → model → prediction)
* Interactive Streamlit dashboard
* Automatic model training on first run
* Cached models for fast future loading
* Real‑time churn probability prediction
* Business‑friendly insights and recommendations

---

## 🧠 Machine Learning Models

The app trains and evaluates multiple algorithms:

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* Support Vector Machine (SVM)
* K‑Nearest Neighbors (KNN)
* Naive Bayes

The best‑performing models are used for prediction.

---

## 📊 Dataset

**Telco Customer Churn Dataset** (IBM)

* 7,043 customers
* 21 input features (demographics, services, billing)
* Target variable: **Churn (Yes / No)**

Dataset source: [Telco Customer Churn Dataset (Raw GitHub CSV)](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)


---

## 🖥️ Application Pages

* **Home** – Project overview
* **Dataset Overview** – Data preview and statistics
* **EDA** – Churn distribution and feature analysis
* **Model Training** – ML algorithms and setup
* **Model Evaluation** – Performance comparison
* **Prediction** – Real‑time churn prediction
* **Business Insights** – Key takeaways and retention strategies

---

## 🗂️ Project Structure

```
customer-churn-prediction/
│
├── app.py              # Main Streamlit application
├── requirements.txt    # Project dependencies
├── README.md           # Project documentation
└── assets/             # Project images inside this folder

```

---

## ⚙️ Tech Stack

* **Python**
* **Streamlit**
* **Pandas & NumPy**
* **Scikit‑learn**
* **Plotly**

---

## ▶️ Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

⏳ On the first run, models are trained automatically. Subsequent runs load instantly from cache.

---

## ☁️ Deployment

This app is deployed using **Streamlit Cloud**:

1. Push the code to GitHub
2. Go to [https://share.streamlit.io/](https://share.streamlit.io/)
3. Select the repository
4. Set `app.py` as the main file
5. Deploy

---

## 📄 License

This project is open‑source and free to use for learning, portfolio projects, and demos.

---

## 🙌 Author

**Diganta Datta**

---

⭐ If you like this project, consider giving the repository a star!

