import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, roc_auc_score
from xgboost import XGBClassifier

# Пути к файлам
MODEL_PATH = "xgboost_model.pkl"
DATA_PATH = "Main_Data.csv"

# Загрузка данных
@st.cache_data
def load_data(data_path):
    data = pd.read_csv(data_path)
    X = data[["T", "E", "C", "FM", "Xfm", "AFM", "Xafm"]]
    y = data["h"]
    return X, y

# Загрузка модели
@st.cache_data
def load_model(model_path):
    with open(model_path, "rb") as file:
        return pickle.load(file)

# Инициализация
X, y = load_data(DATA_PATH)
model = load_model(MODEL_PATH)

st.title("AI_phase_diagram_API")

# Настройки кросс-валидации
st.sidebar.header("Настройки кросс-валидации")
cv_folds = st.sidebar.slider("Количество фолдов (k)", 2, 10, 5)
selected_metric = st.sidebar.selectbox("Метрика для оценки", ["roc_auc", "accuracy", "f1", "precision", "recall"])

# Кросс-валидация
st.header("Кросс-валидация")
if st.button("Запустить кросс-валидацию"):
    with st.spinner("Выполняется кросс-валидация..."):
        start_time = time.time()
        kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        st.subheader("Результаты кросс-валидации")
        scores = cross_val_score(model, X, y, cv=kfold, scoring=selected_metric)

        for i, score in enumerate(scores, start=1):
            st.write(f"Fold {i}: {selected_metric} = {score:.4f}")

        st.write(f"\nСредний {selected_metric}: {np.mean(scores):.4f}")
        elapsed_time = time.time() - start_time
        st.write(f"Время выполнения: {elapsed_time:.2f} секунд")

# Построение графика ROC-AUC
st.header("ROC-AUC для кросс-валидации")
if st.button("Построить график ROC-AUC"):
    with st.spinner("Генерация графика ROC-AUC..."):
        start_time = time.time()
        kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        roc_auc_scores = []

        for train_idx, test_idx in kfold.split(X, y):
            X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
            y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train_fold, y_train_fold)
            y_pred_prob_fold = model.predict_proba(X_test_fold)[:, 1]
            roc_auc_scores.append(roc_auc_score(y_test_fold, y_pred_prob_fold))

        # Построение графика
        plt.figure(figsize=(10, 6))
        folds = np.arange(1, len(roc_auc_scores) + 1)
        plt.plot(folds, roc_auc_scores, marker='o', linestyle='-', color='royalblue', linewidth=2, label='ROC-AUC Score')
        plt.title("5-fold Cross-Validation ROC-AUC", fontsize=16)
        plt.xlabel("Fold Number", fontsize=14)
        plt.ylabel("ROC-AUC Score", fontsize=14)
        plt.xticks(folds)
        plt.grid(True)
        plt.legend(fontsize=12)
        st.pyplot(plt)
        elapsed_time = time.time() - start_time
        st.write(f"Время выполнения: {elapsed_time:.2f} секунд")
