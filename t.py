import numpy as np
import cv2
from tensorflow.keras.models import load_model
import streamlit as st
import pandas as pd

# =============================
# ЗАГРУЗКА ДАННЫХ И МОДЕЛИ
# =============================
# Загружаем модель
@st.cache_resource  # кэшируем модель, чтобы не грузить каждый раз
def load_my_model():
    model = load_model('resnet50-Covid-19-94.33.h5', compile=False)
    # Если веса отдельно — раскомментируй:
    # model.load_weights("resnet50-Covid-19-weights.h5")
    return model

model = load_my_model()

# Определяем метки классов — ПОРЯДОК ВАЖЕН!
# Должен совпадать с порядком, который использовался при обучении (обычно по алфавиту или как в генераторе)
class_labels = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']

# =============================
# ФУНКЦИЯ ПРЕДСКАЗАНИЯ
# =============================

def predict_image(image_array) -> tuple:
    """
    Принимает numpy-массив изображения, возвращает (метку, уверенность).
    """
    # 1. Конвертируем BGR → RGB (если нужно — Streamlit обычно отдаёт RGB)
    # Но OpenCV работает в BGR, а мы конвертируем в RGB — если изображение уже RGB, можно пропустить
    # img = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)  # раскомментируй, если нужно

    # 2. Ресайзим до 224x224
    img = cv2.resize(image_array, (224, 224))

    # 3. Нормализуем пиксели до [0, 1]
    # img = img.astype(np.float32) / 255.0

    # 4. Добавляем batch-ось: (224,224,3) → (1,224,224,3)
    img = np.expand_dims(img, axis=0)

    # 5. Предсказание
    predictions = model.predict(img, verbose=0)

    # 6. Получаем индекс и уверенность
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    # 7. Возвращаем метку и уверенность
    return class_labels[predicted_class_index], confidence

# =============================
# STREAMLIT ИНТЕРФЕЙС
# =============================

st.title("🩺 Классификация рентгеновских снимков лёгких")
st.write("Загрузите рентгеновский снимок грудной клетки, чтобы определить диагноз.")

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
        st.image(image, channels="BGR", caption="Загруженное изображение", use_container_width=True)

        # Делаем предсказание
        with st.spinner("Анализируем изображение..."):
            label, confidence = predict_image(image)

        # Выводим результат
        st.success(f"**Диагноз:** {label}")
        st.info(f"**Уверенность модели:** {confidence:.4f} (максимум 1.0)")

        # Дополнительно: вывод всех вероятностей
        st.write("### Распределение вероятностей по классам:")
        probs = model.predict(np.expand_dims(cv2.resize(image, (224,224)).astype(np.float32), axis=0), verbose=0)[0]
        prob_df = pd.DataFrame({
            'Класс': class_labels,
            'Вероятность': probs
        }).sort_values('Вероятность', ascending=False).reset_index(drop=True)
        st.bar_chart(prob_df.set_index('Класс'))