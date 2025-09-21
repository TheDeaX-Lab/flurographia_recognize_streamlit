import numpy as np
import cv2
from tensorflow.keras.models import load_model
import streamlit as st
import pandas as pd
import time

# =============================
# ЗАГРУЗКА МОДЕЛЕЙ (с кэшированием)
# =============================

MODEL_PATHS = {
    "ResNet50 (276MB) 94%": "resnet50-Covid-19-94.33.h5",
    "DenseNet121 (83MB) 95%": "densenet121-Covid-19-95.09.h5",
    "NASNet (52MB) 95%": "NASNet-Covid-19-95.42.h5",
}

@st.cache_resource
def load_model_by_name(model_name: str):
    """Загружает модель по имени из словаря MODEL_PATHS."""
    start_time = time.time()
    path = MODEL_PATHS[model_name]
    model = load_model(path, compile=False)
    return model, time.time() - start_time

# Загружаем все модели при первом запуске (опционально — можно загружать по требованию)
# Но лучше загружать по требованию, чтобы не тратить память
# models = {name: load_model_by_name(name) for name in MODEL_PATHS.keys()}  # если хочешь все сразу

# Определяем метки классов — ПОРЯДОК ВАЖЕН!
class_labels = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']

# =============================
# ФУНКЦИЯ ПРЕДСКАЗАНИЯ (с передачей модели)
# =============================

def predict_image(image_array, model) -> tuple:
    """
    Принимает numpy-массив изображения и модель, возвращает (метку, уверенность, все вероятности).
    """
    # Ресайзим до 224x224
    img = cv2.resize(image_array, (224, 224))

    # Нормализация (если модель обучалась с нормализацией — раскомментируй)
    # img = img.astype(np.float32) / 255.0

    # Добавляем batch-ось: (224,224,3) → (1,224,224,3)
    img = np.expand_dims(img, axis=0)

    # Предсказание
    predictions = model.predict(img, verbose=0)

    # Получаем индекс и уверенность
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    return class_labels[predicted_class_index], confidence, predictions[0]

# =============================
# STREAMLIT ИНТЕРФЕЙС
# =============================

st.title("🩺 Классификация рентгеновских снимков лёгких")
st.write("Загрузите рентгеновский снимок грудной клетки, чтобы определить диагноз.")

# Выбор модели
selected_model_name = st.selectbox(
    "Выберите модель для классификации:",
    list(MODEL_PATHS.keys()),
    index=2  # по умолчанию — NASNet, как самая точная
)

# Загружаем выбранную модель (ленивая загрузка + кэш)
model, finished_time = load_model_by_name(selected_model_name)
# st.info(f"Время загрузки модели {finished_time:.2f} секунд")

# Загрузка изображения
uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Преобразуем в массив с помощью OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # BGR формат

    if image is None:
        st.error("Не удалось загрузить изображение. Попробуйте другой файл.")
    else:
        # Показываем изображение
        st.image(image, channels="BGR", caption="Загруженное изображение", use_column_width=True)

        # Делаем предсказание
        with st.spinner(f"Анализируем изображение с помощью {selected_model_name}..."):
            start_time = time.time()
            label, confidence, probs = predict_image(image, model)
            finished_time = time.time() - start_time

        # Выводим результат
        st.info(f"Время отведенное на прогноз {finished_time:.2f} секунд")
        st.success(f"**Диагноз:** {label}")
        st.info(f"**Уверенность модели:** {confidence:.4f} (максимум 1.0)")

        # Дополнительно: вывод всех вероятностей
        st.write("### Распределение вероятностей по классам:")
        prob_df = pd.DataFrame({
            'Класс': class_labels,
            'Вероятность': probs
        }).sort_values('Вероятность', ascending=False).reset_index(drop=True)
        st.bar_chart(prob_df.set_index('Класс'))

        # Опционально: показать, какая модель использовалась
        st.caption(f"Использована модель: *{selected_model_name}*")