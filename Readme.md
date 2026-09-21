# 🏠 House Price Prediction

A Machine Learning project for predicting California housing prices using regression models.

## 📌 Project Overview

This project uses the California Housing dataset to predict the median house value of a geographic area based on different features such as location, income, rooms, bedrooms, population, and households.

The project includes:

- Data cleaning
- Exploratory Data Analysis (EDA)
- Feature engineering
- Linear Regression
- Random Forest Regression
- Hyperparameter tuning with GridSearchCV
- Model evaluation
- Streamlit web application

## 🤖 Machine Learning Models

Two regression models were explored:

### Linear Regression

A baseline regression model used to establish an initial performance level.

### Random Forest Regressor

An ensemble learning model used to improve prediction performance.

Hyperparameter tuning was performed using:

```text
18 configurations × 5 folds = 90 training runs
```

with GridSearchCV.

## 🛠️ Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib

## 📊 Dataset

The project uses the California Housing dataset.

The dataset contains information about geographic areas in California, including:

- Longitude
- Latitude
- Housing median age
- Total rooms
- Total bedrooms
- Population
- Households
- Median income
- Ocean proximity

The target variable is:

```text
median_house_value
```

## 🌐 Streamlit Application

A Streamlit web application was created to allow users to enter housing-area information and obtain a predicted median house value.

To run the application locally:

```bash
streamlit run app.py
```

## 📁 Project Structure

```text
House_Price_Prediction/
│
├── models/
│   ├── house_price_model.pkl
│   └── scaler.pkl
│
├── app.py
├── house_price_prediction.ipynb
├── housing.csv
├── requirements.txt
└── README.md
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/House_Price_Prediction.git
```

Navigate to the project:

```bash
cd House_Price_Prediction
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## 🚀 Future Improvements

- Improve the Streamlit interface
- Add prediction visualizations
- Compare additional regression models
- Improve model preprocessing using Scikit-learn pipelines
- Deploy the Streamlit application