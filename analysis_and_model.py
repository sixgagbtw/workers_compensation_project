import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess():
    data = fetch_openml(data_id=42876, as_frame=True, parser='auto')
    df = data.frame.copy()
    
    # Признаки
    df['DateTimeOfAccident'] = pd.to_datetime(df['DateTimeOfAccident'])
    df['DateReported'] = pd.to_datetime(df['DateReported'])
    df['AccidentMonth'] = df['DateTimeOfAccident'].dt.month
    df['AccidentDayOfWeek'] = df['DateTimeOfAccident'].dt.dayofweek
    df['ReportingDelay'] = (df['DateReported'] - df['DateTimeOfAccident']).dt.days
    df.drop(columns=['DateTimeOfAccident', 'DateReported'], inplace=True)
    
    # Кодирование категорий
    categorical_cols = ['Gender', 'MaritalStatus', 'PartTimeFullTime', 'ClaimDescription']
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    X = df.drop(columns=['UltimateIncurredClaimCost'])
    y = df['UltimateIncurredClaimCost']
    
    # Масштабирование
    numerical_features = ['Age', 'DependentChildren', 'DependentsOther', 'WeeklyPay',
                          'HoursWorkedPerWeek', 'DaysWorkedPerWeek', 'InitialCaseEstimate',
                          'AccidentMonth', 'AccidentDayOfWeek', 'ReportingDelay']
    scaler = StandardScaler()
    X[numerical_features] = scaler.fit_transform(X[numerical_features])
    
    return X, y, label_encoders, scaler

def train_models(X_train, y_train):
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbosity=0)
    }
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
    return trained

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return y_pred, mae, rmse, r2

def analysis_and_model_page():
    st.title("Прогнозирование стоимости страховых выплат")
    st.markdown("Загрузка данных, обучение моделей и предсказание итоговой стоимости страхового возмещения.")
    
    if 'df_loaded' not in st.session_state:
        st.session_state.df_loaded = False
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False

    with st.expander("1. Загрузка и предобработка данных", expanded=not st.session_state.df_loaded):
        if st.button("Загрузить и подготовить данные"):
            with st.spinner("Идёт загрузка с OpenML..."):
                X, y, label_encoders, scaler = load_and_preprocess()
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.label_encoders = label_encoders
                st.session_state.scaler = scaler
                st.session_state.feature_names = X.columns.tolist()
                st.session_state.df_loaded = True
                st.session_state.models_trained = False
                st.success("Данные загружены и предобработаны!")
        
        if st.session_state.df_loaded:
            st.write(f"**Размер обучающей выборки:** {st.session_state.X_train.shape[0]} записей")
            st.write(f"**Размер тестовой выборки:** {st.session_state.X_test.shape[0]} записей")
            st.subheader("Пример данных после предобработки")
            st.dataframe(st.session_state.X_train.head())
            
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            ax[0].hist(st.session_state.y_train, bins=50, color='skyblue', edgecolor='black')
            ax[0].set_title("Распределение в обучающей выборке")
            ax[0].set_xlabel("Стоимость")
            ax[1].hist(st.session_state.y_test, bins=50, color='salmon', edgecolor='black')
            ax[1].set_title("Распределение в тестовой выборке")
            st.pyplot(fig)

    if st.session_state.df_loaded:
        with st.expander("2. Обучение и сравнение моделей", expanded=not st.session_state.models_trained):
            if st.button("Обучить модели"):
                with st.spinner("Обучение моделей..."):
                    models = train_models(st.session_state.X_train, st.session_state.y_train)
                    st.session_state.models = models
                    results = {}
                    for name, model in models.items():
                        y_pred, mae, rmse, r2 = evaluate_model(model, st.session_state.X_test, st.session_state.y_test)
                        results[name] = {"MAE": mae, "RMSE": rmse, "R²": r2, "y_pred": y_pred}
                    st.session_state.results = results
                    st.session_state.models_trained = True
                    st.success("Модели обучены!")
            
            if st.session_state.models_trained:
                st.subheader("Сравнение метрик")
                metrics_df = pd.DataFrame({
                    name: [res['MAE'], res['RMSE'], res['R²']] 
                    for name, res in st.session_state.results.items()
                }, index=["MAE", "RMSE", "R²"]).T
                st.dataframe(metrics_df.style.highlight_min(subset=['MAE', 'RMSE'], color='lightgreen')
                             .highlight_max(subset=['R²'], color='lightgreen'))
                
                best_model_name = st.selectbox(
                    "Выберите модель для визуализации", 
                    list(st.session_state.results.keys())
                )
                best_res = st.session_state.results[best_model_name]
                y_test = st.session_state.y_test
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(y_test, best_res['y_pred'], alpha=0.3)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                ax.set_xlabel("Реальные значения")
                ax.set_ylabel("Предсказания")
                ax.set_title(f"{best_model_name}: Предсказания vs Реальные")
                st.pyplot(fig)
                
                if best_model_name in ["Random Forest", "XGBoost"]:
                    model = st.session_state.models[best_model_name]
                    importances = model.feature_importances_
                    feat_imp = pd.DataFrame({
                        'feature': st.session_state.feature_names,
                        'importance': importances
                    }).sort_values('importance', ascending=False)
                    
                    st.subheader(f"Топ-10 важных признаков ({best_model_name})")
                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    ax2.barh(feat_imp['feature'].head(10), feat_imp['importance'].head(10))
                    ax2.invert_yaxis()
                    ax2.set_xlabel("Важность")
                    st.pyplot(fig2)

    if st.session_state.models_trained:
        st.header("3. Предсказание для нового случая")
        best_model_key = max(st.session_state.results, key=lambda k: st.session_state.results[k]['R²'])
        model_choice = st.selectbox(
            "Модель для предсказания", 
            list(st.session_state.models.keys()),
            index=list(st.session_state.models.keys()).index(best_model_key)
        )
        model = st.session_state.models[model_choice]
        
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Возраст", min_value=13, max_value=76, value=35)
                gender = st.selectbox("Пол", ["M", "F"])
                weekly_pay = st.number_input("Еженедельная зарплата ($)", min_value=0, value=500)
                initial_estimate = st.number_input("Начальная оценка ($)", min_value=0.0, value=5000.0, step=100.0)
                accident_month = st.slider("Месяц происшествия", 1, 12, 6)
                accident_dayofweek = st.slider("День недели (0=Пн)", 0, 6, 2)
            with col2:
                marital_status = st.selectbox("Семейное положение", 
                                              ["Single", "Married", "Separated", "Widowed", "Divorced"])
                part_time = st.selectbox("Тип занятости", ["Full Time", "Part Time"])
                dep_children = st.number_input("Дети на иждивении", 0, 10, 0)
                dep_other = st.number_input("Другие иждивенцы", 0, 10, 0)
                hours_week = st.number_input("Часов в неделю", 0, 80, 40)
                days_week = st.number_input("Дней в неделю", 1, 7, 5)
                reporting_delay = st.number_input("Задержка отчёта (дни)", 0, 365, 10)
                claim_desc = st.selectbox("Описание заявки", 
                                          ["LACERATION", "FRACTURE", "CONTUSION", "SPRAIN", "STRAIN", 
                                           "BURN", "DISLOCATION", "OTHER"])
            
            submitted = st.form_submit_button("Предсказать стоимость")
            
            if submitted:
                input_data = pd.DataFrame([{
                    'Age': age,
                    'Gender': gender,
                    'MaritalStatus': marital_status,
                    'DependentChildren': dep_children,
                    'DependentsOther': dep_other,
                    'WeeklyPay': weekly_pay,
                    'PartTimeFullTime': part_time,
                    'HoursWorkedPerWeek': hours_week,
                    'DaysWorkedPerWeek': days_week,
                    'ClaimDescription': claim_desc,
                    'InitialCaseEstimate': initial_estimate,
                    'AccidentMonth': accident_month,
                    'AccidentDayOfWeek': accident_dayofweek,
                    'ReportingDelay': reporting_delay
                }])
                
                for col, le in st.session_state.label_encoders.items():
                    input_data[col] = input_data[col].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
                
                num_features = ['Age', 'DependentChildren', 'DependentsOther', 'WeeklyPay',
                                'HoursWorkedPerWeek', 'DaysWorkedPerWeek', 'InitialCaseEstimate',
                                'AccidentMonth', 'AccidentDayOfWeek', 'ReportingDelay']
                input_data[num_features] = st.session_state.scaler.transform(input_data[num_features])
                input_data = input_data[st.session_state.feature_names]
                
                prediction = model.predict(input_data)[0]
                st.metric("Предсказанная итоговая стоимость", f"${prediction:,.2f}")

# Для тестирования страницы напрямую
analysis_and_model_page()
