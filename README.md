# House Price Prediction Web App

### Project Overview

This project is a **House Price Prediction Web Application** built using **Flask**, **XGBoost**, and **Docker**. It predicts house prices based on various input features like area, location, number of rooms, and other housing attributes. The app uses a trained machine learning model and provides an intuitive web interface for users to interact with.

🔗 **Live Demo:** [House Price Predictor (Render Deployment)](https://house-price-prediction-aoj4.onrender.com)

---

## Key Features

* **Machine Learning Model:** Trained using **XGBoost (via scikit-learn API)** for high prediction accuracy.
* **Metrics Achieved:**

  * RMSE: **26,948.76**
  * R² Score: **0.9053**
* **Interactive Web Interface:** Simple and clean interface built with **HTML, CSS, and Flask templates**.
* **Containerized Deployment:** Fully containerized using **Docker**, making it portable and easy to deploy on any platform.

---

## 🧠 Model Development Workflow

1. **Data Preprocessing:**

   * Cleaned and encoded categorical features.
   * Split the dataset into **train (80%)** and **test (20%)** sets.

2. **Model Training:**

   * Used **XGBoost Regressor** integrated with the **scikit-learn API**.
   * Tuned hyperparameters using grid search and achieved optimal performance.

3. **Evaluation:**

   * Achieved **R² = 0.9053** and **RMSE = 26,948.76**.

4. **Model Export:**

   * Saved trained model as `house_price_model.pkl`.
   * Encoders saved as `label_encoders.pkl`.

5. **Deployment:**

   * Created a **Flask app (`app.py`)** to handle user input and display predictions.
   * **Dockerized** the entire application for easy deployment.

---

## 🧩 Project Structure

```
House_Price_Predictor_ver2/
│
├── app.py                  # Flask backend
├── requirements.txt        # Project dependencies
├── Dockerfile              # Docker container setup
├── .dockerignore           # Ignored files for Docker build
├── house_price_model.pkl   # Trained ML model
├── label_encoders.pkl      # Encoders for categorical variables
│
├── templates/              # HTML templates
│   └── index.html          # Main UI page
│
├── static/                 # CSS and JS files
│   └── style.css           # Frontend styling
│
├── data/                   # Raw data (ignored in deployment)
└── notebooks/              # Development notebooks
```

---

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/House_Price_Predictor_ver2.git
cd House_Price_Predictor_ver2
```

### 2. Build the Docker Image

```bash
docker build -t house-price-app .
```

### 3. Run the Container

```bash
docker run -p 5000:5000 house-price-app
```

The application will be available at **[http://localhost:5000/](http://localhost:5000/)**

---

## 🧾 Tech Stack

* **Frontend:** HTML, CSS (Flask templates)
* **Backend:** Flask (Python)
* **Model:** XGBoost via scikit-learn API
* **Containerization:** Docker
* **Deployment:** Render (via Docker container)

---

## 📊 Model Performance

| Metric   | Value     |
| -------- | --------- |
| RMSE     | 26,948.76 |
| R² Score | 0.9053    |

---

## 👨‍💻 Author

**Amrit Mohapatra**
Engineering Student at NIT Rourkela
[GitHub Profile](https://github.com/AmriMohapatra)

---

## 🧩 License

This project is released under the **MIT License**.
