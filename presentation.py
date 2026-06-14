import streamlit as st
import reveal_slides as rs

slides_md = """
# Прогнозирование стоимости страховых выплат

Note:
Анализ данных о компенсациях работникам с помощью машинного обучения.

---

## Введение

- Датасет Workers Compensation (100 000 записей)
- Задача: предсказать итоговую стоимость страхового возмещения
- Инструменты: Python, scikit-learn, Streamlit, XGBoost

---

## Бизнес-задача

- Страховые компании нуждаются в точном прогнозе выплат
- Начальная оценка (InitialCaseEstimate) часто неточна
- Точный прогноз позволяет правильно формировать резервы и тарифы

---

## Данные

| Признак | Тип | Описание |
|---------|-----|----------|
| Age | Числовой | Возраст работника |
| Gender | Категориальный | Пол |
| WeeklyPay | Числовой | Еженедельная зарплата |
| InitialCaseEstimate | Числовой | Начальная оценка случая |
| ClaimDescription | Категориальный | Тип травмы |
| ... | ... | ... |

Целевая переменная: **UltimateIncurredClaimCost**

---

## Предобработка данных

1. Преобразование дат → месяц, день недели, задержка отчёта
2. Label Encoding категориальных признаков
3. Удаление неинформативных столбцов
4. Масштабирование (StandardScaler)
5. Разделение 80/20 (train/test)

---

## Модели машинного обучения

Сравнивались четыре регрессора:
- **Linear Regression** – базовая модель
- **Ridge Regression** – линейная с L2-регуляризацией
- **Random Forest Regressor** – ансамбль деревьев
- **XGBoost Regressor** – градиентный бустинг

---

## Результаты (образец)

| Модель | MAE | RMSE | R² |
|--------|-----|------|----|
| Linear Regr. | 8500 | 14500 | 0.72 |
| Ridge | 8400 | 14300 | 0.73 |
| Random Forest | 3200 | 5800 | 0.92 |
| XGBoost | 3100 | 5600 | 0.93 |

Лучшая модель – **XGBoost** (R² > 0.9)

---

## Важность признаков (Random Forest)

1. InitialCaseEstimate
2. WeeklyPay
3. ReportingDelay
4. Age
5. ClaimDescription
6. HoursWorkedPerWeek
...

Ключевой фактор – начальная оценка, но модель корректирует её на основе других параметров.

---

## Streamlit-приложение

- Интерактивная загрузка и предобработка
- Обучение нескольких моделей в реальном времени
- Сравнение метрик и графики
- Анализ важности признаков
- Форма для предсказания нового случая

---

## Заключение

- Разработана модель с высокой точностью (R² > 0.9)
- Выявлены ключевые факторы стоимости страховых выплат
- Приложение позволяет быстро оценить новый случай
- Возможные улучшения: добавление внешних данных, более глубокая настройка гиперпараметров, ансамблирование

---

## Спасибо за внимание
"""

def presentation_page():
    st.title("Презентация проекта")
    
    with st.sidebar:
        st.header("Настройки презентации")
        theme = st.selectbox("Тема", ["black", "white", "league", "beige", "sky", "night"], index=0)
        height = st.number_input("Высота слайдов", value=500, min_value=300)
        transition = st.selectbox("Переход", ["slide", "convex", "concave", "zoom"])
        plugins = st.multiselect("Плагины", ["highlight", "notes", "search", "zoom"], default=["highlight"])
    
    rs.slides(
        slides_md,
        height=height,
        theme=theme,
        config={
            "transition": transition,
            "plugins": plugins,
        },
        markdown_props={"data-separator-vertical": "^--$"},
    )

# Для тестирования страницы напрямую 
presentation_page()
