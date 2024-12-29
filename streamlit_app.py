import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Главный интерфейс
st.title("AI_phase_diagram_API")

# Метрики модели
st.header("Метрики модели")
if st.button("Рассчитать метрики"):
    with st.spinner("Расчет метрик модели..."):
        start_time = time.time()
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision (Class 0)": precision_score(y_test, y_pred, pos_label=0),
            "Precision (Class 1)": precision_score(y_test, y_pred, pos_label=1),
            "Recall (Class 0)": recall_score(y_test, y_pred, pos_label=0),
            "Recall (Class 1)": recall_score(y_test, y_pred, pos_label=1),
            "F1-score (Class 0)": f1_score(y_test, y_pred, pos_label=0),
            "F1-score (Class 1)": f1_score(y_test, y_pred, pos_label=1),
            "ROC-AUC": roc_auc_score(y_test, y_pred_prob),
        }
        st.json(metrics)
        elapsed_time = time.time() - start_time
        st.write(f"Время выполнения: {elapsed_time:.2f} секунд")

# ROC-кривая
st.header("ROC-кривая")
if st.button("Построить ROC-кривую"):
    with st.spinner("Построение ROC-кривой..."):
        start_time = time.time()
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = roc_auc_score(y_test, y_pred_prob)

        plt.figure()
        plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        st.pyplot(plt)
        elapsed_time = time.time() - start_time
        st.write(f"Время выполнения: {elapsed_time:.2f} секунд")

# Кросс-валидация
st.header("Кросс-валидация")
if st.button("Запустить кросс-валидацию"):
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    progress = st.progress(0)

    results = []
    mean_roc_auc = 0
    st.subheader("Результаты по каждому фолду")
    start_time = time.time()

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y), start=1):
        with st.spinner(f"Обновление модели на фолде {fold}..."):
            X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
            y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train_fold, y_train_fold)
            y_pred_fold = model.predict(X_test_fold)
            y_pred_prob_fold = model.predict_proba(X_test_fold)[:, 1]

            # Вычисление метрик
            cm = confusion_matrix(y_test_fold, y_pred_fold)
            roc_auc = roc_auc_score(y_test_fold, y_pred_prob_fold)
            mean_roc_auc += roc_auc

            # Отображение результатов для текущего фолда
            st.write(f"Fold {fold}:")
            st.write(f"ROC-AUC: {roc_auc:.4f}")
            cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
            st.table(cm_df)

            # Обновление прогресса
            progress.progress(fold / 5)

    mean_roc_auc /= kfold.get_n_splits()
    st.subheader(f"Средний ROC-AUC: {mean_roc_auc:.4f}")
    elapsed_time = time.time() - start_time
    st.write(f"Время выполнения: {elapsed_time:.2f} секунд")

# Построение графика ROC-AUC
st.header("ROC-AUC для кросс-валидации")
if st.button("Построить график ROC-AUC"):
    with st.spinner("Обновление модели для графика ROC-AUC..."):
        start_time = time.time()
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        roc_auc_scores = []
        progress = st.progress(0)

        for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y), start=1):
            X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
            y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train_fold, y_train_fold)
            y_pred_prob_fold = model.predict_proba(X_test_fold)[:, 1]
            roc_auc_scores.append(roc_auc_score(y_test_fold, y_pred_prob_fold))

            # Обновление прогресса
            progress.progress(fold / 5)

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

# Важность признаков
st.header("Важность признаков")
if st.button("Показать важность признаков"):
    with st.spinner("Обновление модели для анализа важности признаков..."):
        start_time = time.time()
        model.fit(X, y)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        st.pyplot(plt)
        elapsed_time = time.time() - start_time
        st.write(f"Время выполнения: {elapsed_time:.2f} секунд")

# SHAP-анализ
st.header("SHAP-анализ")
if st.button("Построить SHAP-анализ"):
    with st.spinner("Обновление модели для SHAP-анализа..."):
        start_time = time.time()
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)

        shap.summary_plot(shap_values, X_train, show=False)
        st.pyplot(plt)
        elapsed_time = time.time() - start_time
        st.write(f"Время выполнения: {elapsed_time:.2f} секунд")

# Предсказания
st.header("Предсказания")
uploaded_file = st.file_uploader("Загрузите файл CSV", type="csv")
if uploaded_file:
    try:
        # Загрузка данных
        data = pd.read_csv(uploaded_file)
        REQUIRED_COLUMNS = ["L", "sample", "T", "E", "C", "FM", "Xfm", "AFM", "Xafm"]
        if not all(col in data.columns for col in REQUIRED_COLUMNS):
            st.error(f"Отсутствуют необходимые колонки: {', '.join(REQUIRED_COLUMNS)}")
        else:
            # Получение предсказаний
            features = ["T", "E", "C", "FM", "Xfm", "AFM", "Xafm"]
            data["y_predict"] = model.predict(data[features])
            data["predict_proba"] = model.predict_proba(data[features])[:, 1]

            # Определение sample_predict
            sample_predictions = (
                data.groupby("sample")["y_predict"]
                .apply(lambda x: 1 if x.sum() > len(x) / 2 else 0)
                .rename("sample_predict")
            )
            data = data.merge(sample_predictions, on="sample")

            # Проверка совпадений
            data["y_predict_matches_sample"] = data["y_predict"] == data["sample_predict"]

            # Вывод данных
            st.write(data)

            # Сохранение результатов
            csv = data.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Скачать результаты",
                data=csv,
                file_name="processed_results.csv",
                mime="text/csv",
            )
    except Exception as e:
        st.error(f"Ошибка обработки файла: {str(e)}")
