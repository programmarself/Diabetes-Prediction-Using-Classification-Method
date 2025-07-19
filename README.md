# 🩺 Diabetes Prediction Using Classification Method 🩺 

### *Machine Learning & Data-Science Project ( Final Year Project )*  
> Built with `Python 3.x` • `Anaconda` • `Jupyter-Lab` • `Scikit-learn` • `TensorFlow/Keras`

---
![FYP](https://github.com/user-attachments/assets/ddbec609-008c-4c7c-bf72-cdd00dd808cd)

## 🎯 Project Overview
| Item | Details |
|------|---------|
| **Goal** | Predict whether a patient has **diabetes** or not. |
| **Approach** | Supervised Classification using Neural Network + Classical ML models. |
| **Dataset** | Pima Indians Diabetes Dataset (768 rows × 9 columns). |
| **Tools** | <img src="https://img.icons8.com/color/48/000000/python.png"/> <img src="https://img.icons8.com/color/48/000000/anaconda.png"/> <img src="https://img.icons8.com/color/48/000000/tensorflow.png"/> |

---

## 📊 1. Exploratory Data Analysis (EDA)

### 1.1 Quick Peek 👀
```python
import pandas as pd
df = pd.read_csv('diabetes.csv')
df.head()
```

| Pregnancies | Glucose | BloodPressure | SkinThickness | Insulin | BMI | DiabetesPedigreeFunction | Age | Outcome |
|-------------|---------|---------------|---------------|---------|-----|---------------------------|-----|---------|
| 6 | 148 | 72 | 35 | 0 | 33.6 | 0.627 | 50 | 1 |
| 1 | 85 | 66 | 29 | 0 | 26.6 | 0.351 | 31 | 0 |

### 1.2 Summary Statistics 📈
```python
df.describe().T.style.bar(subset=['mean'], color='#5fba7d')
```

| | count | mean | std | min | 25% | 50% | 75% | max |
|---|------|------|-----|-----|-----|-----|-----|-----|
| Glucose | 768 | 120.89 | 31.97 | 0 | 99 | 117 | 140.25 | 199 |
| BMI | 768 | 31.99 | 7.88 | 0 | 27.3 | 32 | 36.6 | 67.1 |

### 1.3 Visual Insights 📉
![pairplot](https://i.imgur.com/BmVj6bP.png)
*Pair-plot showing correlations among features; red points are diabetic (Outcome=1).*

- **Strongest predictor**: Glucose levels  
- **Missing values**: 0 in Insulin & SkinThickness → impute with median.

---

## 🧹 2. Data Pre-processing

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Impute 0's → median
cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
X[cols] = X[cols].replace(0, X[cols].median())

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split 70/30
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.30, random_state=42, stratify=y)
```

---

## 🧠 3. Neural Network (Keras)

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])
history = model.fit(X_train, y_train,
                    validation_split=0.15,
                    epochs=100, batch_size=16, verbose=0)
```

![loss](https://i.imgur.com/3kLxK7D.png)

---

## 🧪 4. Classical ML Models

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Logistic Regression** | 0.794 | 0.75 | 0.64 | 0.69 |
| **Decision Tree** | 0.739 | 0.70 | 0.60 | 0.65 |
| **SVM (RBF)** | 0.792 | 0.76 | 0.62 | 0.68 |

![confusion_matrix](https://i.imgur.com/0fQzXbL.png)

---

## 🏆 5. Model Comparison & Best Pick

| Metric | Neural Net | Logistic | Decision Tree | SVM |
|--------|------------|----------|---------------|-----|
| **Accuracy** | **0.844** 👑 | 0.794 | 0.739 | 0.792 |
| **Precision** | 0.81 | 0.75 | 0.70 | 0.76 |
| **Recall** | **0.73** | 0.64 | 0.60 | 0.62 |
| **F1-Score** | **0.77** | 0.69 | 0.65 | 0.68 |

> 🥇 **Neural Network wins** with 84.4 % accuracy.

---

## 💾 6. Save Models for Production

```python
# Keras model
model.save('diabetes_nn.h5')

# Sci-kit models
import joblib
joblib.dump(lr, 'diabetes_lr.pkl')
joblib.dump(dt, 'diabetes_dt.pkl')
joblib.dump(svm, 'diabetes_svm.pkl')
```

---

## 🚀 7. Quick Usage Demo

```python
# Load & predict
from tensorflow.keras.models import load_model
model = load_model('diabetes_nn.h5')

patient = [[6, 148, 72, 35, 0, 33.6, 0.627, 50]]
patient_scaled = scaler.transform(patient)
pred = model.predict(patient_scaled)[0][0]
print("Risk of diabetes: {:.1%}".format(pred))
# → Risk of diabetes: 91.4%
```

---

## 📁 Project Tree

```
📦 Diabetes-Prediction/
 ├─ 📁 data/
 │   └─ diabetes.csv
 ├─ 📁 notebooks/
 │   └─ EDA.ipynb
 ├─ 📁 models/
 │   ├─ diabetes_nn.h5
 │   └─ *.pkl
 ├─ 📁 src/
 │   ├─ train.py
 │   └─ predict.py
 ├─ 📄 requirements.txt
 └─ 📄 README.md
```

---

## 📚 Requirements (`requirements.txt`)
```
pandas==2.2.2
numpy==1.26.4
matplotlib==3.9.0
seaborn==0.13.2
scikit-learn==1.5.0
tensorflow==2.17.0
joblib==1.4.2
```

---

## 🤝 Contributing
Feel free to open issues or PRs to improve the model or add new features (e.g., SHAP explainability, Streamlit GUI).

---

## 📄 License
MIT © 2025 Diabetes-Prediction-Team

---

<div align="center">
  <img src="https://img.icons8.com/color/96/000000/health-checkup.png"/>
  <p><em>“Early diagnosis saves lives.”</em></p>
</div>
-------------------------------------------------------------------------------------------------------------------


<h2 style="font-family: 'poppins'; font-weight: bold; color: Green;">👨💻 By: Irfan Ullah Khan</h2>


[![GitHub](https://img.shields.io/badge/GitHub-Profile-blue?style=for-the-badge&logo=github)](https://github.com/programmarself) 
[![Kaggle](https://img.shields.io/badge/Kaggle-Profile-blue?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/programmarself) 
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/irfan-ullah-khan-4a2871208/)  

[![YouTube](https://img.shields.io/badge/YouTube-Profile-red?style=for-the-badge&logo=youtube)](https://www.youtube.com/@irfanullahkhan7748) 
[![Email](https://img.shields.io/badge/Email-Contact%20Me-red?style=for-the-badge&logo=email)](mailto:programmarself@gmail.com)
[![Website](https://img.shields.io/badge/Website-Contact%20Me-red?style=for-the-badge&logo=website)]([https://flowcv.me/ikm](https://programmarself.github.io/My_Portfolio/))
