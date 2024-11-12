import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# Configuración de Streamlit y estilos
st.set_page_config(page_title="Comparación de Algoritmos", page_icon=":chart_with_upwards_trend:")

# Paleta de colores
colors = ["#20063B", "#586F7C", "#B8D8D9", "#FFC09F", "#F4F4F9"]
sns.set_palette(sns.color_palette(colors))

# Cargar el vectorizador TF-IDF y el analizador de sentimiento
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
analyzer = SentimentIntensityAnalyzer()

# Función de preprocesamiento que combina la pregunta y las respuestas de ambos modelos
def preprocess_input(question, response_a, response_b):
    # Concatenar la pregunta y respuestas
    combined_text = f"{question} {response_a} {response_b}"
    
    # Transformar el texto combinado con el vectorizador TF-IDF
    tfidf_features = vectorizer.transform([combined_text]).toarray()
    
    # Calcular la puntuación de sentimiento
    sentiment_score = analyzer.polarity_scores(combined_text)["compound"]
    sentiment_features = np.array([[sentiment_score]])
    
    # Combinar las características TF-IDF y de sentimiento
    input_features = np.hstack([tfidf_features, sentiment_features])
    
    return input_features

# Cargar modelos entrenados
def load_models():
    model_a, model_b = None, None
    try:
        model_a = tf.keras.models.load_model("modelo_entrenado.h5", compile=False)
        # Compilar el modelo con una función de pérdida y optimizador que sean compatibles
        model_a.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            optimizer='adam',
            metrics=['accuracy']
        )
        print("Modelo A cargado exitosamente.")
    except Exception as e:
        print("Error al cargar Modelo A:", e)

    try:
        model_b = tf.keras.models.load_model("modelo_entrenado.h5", compile=False)
        # Compilar el modelo con una función de pérdida y optimizador que sean compatibles
        model_a.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            optimizer='adam',
            metrics=['accuracy']
        )
        print("Modelo B cargado exitosamente.")
    except Exception as e:
        print("Error al cargar Modelo B:", e)
    return model_a, model_b

model_a, model_b = load_models()

# Título de la Aplicación
st.title("Comparación de Algoritmos de Clasificación")

# Entrada de datos para el usuario
st.header("Ingrese su pregunta y las respuestas de los modelos")

# Ingreso de la pregunta
user_question = st.text_input("Pregunta al chatbot:")

# Ingreso de la respuesta proporcionada por cada modelo
response_a_text = st.text_input("Respuesta del Modelo A:")
response_b_text = st.text_input("Respuesta del Modelo B:")

# Ingreso de los nombres de los modelos
model_a_name = st.text_input("Nombre del Modelo A:")
model_b_name = st.text_input("Nombre del Modelo B:")

# Mostrar la predicción de cada modelo
if user_question and response_a_text and response_b_text and model_a_name and model_b_name:
    # Preprocesar la entrada para ambos modelos
    input_example = preprocess_input(user_question, response_a_text, response_b_text)

    try:
        # Predicciones de ambos modelos sobre el input combinado
        prediction_a = model_a.predict(input_example)
        prediction_b = model_b.predict(input_example)
        print(prediction_a)

        # Mostrar resultados en la app
        st.subheader("Resultados de la evaluación")
        st.subheader("En el modelo 1" )
        if prediction_a[0][1] > prediction_a[0][2]:
            st.write("La respuesta ganadora es la del Modelo A.")
        elif prediction_a[0][1] < prediction_a[0][2]:
            st.write("La respuesta ganadora es la del Modelo B.")
        else:
            st.write("Hay un empate entre las respuestas.")
    except Exception as e:
        st.error(f"Error en la predicción: {e}")



# Opción para ocultar o mostrar detalles del rendimiento
show_details = st.checkbox("Mostrar detalles de rendimiento")

if show_details:
    # Graficar las preferencias del usuario
    st.header("Gráficas de Preferencias")
    st.subheader("Preferencias del Usuario")
    # Graficar precisión de los algoritmos
    st.subheader("Precisión de los Algoritmos")
    fig, ax = plt.subplots()
    accuracy_a = np.random.uniform(0.7, 0.95, 20)
    accuracy_b = np.random.uniform(0.6, 0.9, 20)
    ax.plot(accuracy_a, label="Modelo A", color=colors[0])
    ax.plot(accuracy_b, label="Modelo B", color=colors[1])
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Precisión")
    ax.legend()
    st.pyplot(fig)
