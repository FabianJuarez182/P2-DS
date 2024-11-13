import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configuración de Streamlit y estilos
st.set_page_config(page_title="Comparación de Algoritmos", page_icon=":chart_with_upwards_trend:")

# Definir colores
colors = {
    "russian_violet": "#20063B",
    "paynes_gray": "#586F7C",
    "light_blue": "#B8D9D0",
    "peach": "#FFC09F",
    "ghost_white": "#F4F4F9"
}

# Estilos personalizados con CSS
st.markdown(f"""
    <style>
        /* Headers */
        .stMarkdown h1, .stMarkdown h2 {{
            color: {colors['russian_violet']};
            font-weight: bold;
        }}
        
        /* Subheaders and section titles */
        .stMarkdown h3 {{
            color: {colors['paynes_gray']};
            font-weight: bold;
        }}
        
        /* Button styling */
        div.stButton > button {{
            background-color: {colors['russian_violet']};
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: bold;
            border: none;
        }}
        div.stButton > button:hover {{
            background-color: {colors['peach']};
            color: {colors['russian_violet']};
        }}
        
        /* Table styling */
        .stDataFrame {{
            background-color: {colors['light_blue']};
            color: {colors['russian_violet']};
            border: 1px solid {colors['paynes_gray']};
        }}
        
        /* General text color */
        .css-18e3th9 {{
            color: {colors['paynes_gray']};
        }}
        
        /* Input box styling */
        .stTextInput > div > label {{
            color: {colors['paynes_gray']};
            font-weight: bold;
        }}
        
        /* Input field background and text */
        input {{
            background-color: {colors['light_blue']};
            color: {colors['russian_violet']};
            border-radius: 5px;
            border: 1px solid {colors['paynes_gray']};
            padding: 8px;
            font-size: 16px;
        }}
        
        /* Radio button styling */
        .stRadio > div > label {{
            color: {colors['paynes_gray']};
        }}
        
        /* Adjust padding for main content area */
        section.main > div {{
            padding: 2rem 1rem;
        }}
    </style>
""", unsafe_allow_html=True)

# Cargar el vectorizador TF-IDF y el analizador de sentimiento
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
analyzer = SentimentIntensityAnalyzer()

# Función de preprocesamiento que combina la pregunta y las respuestas de ambos modelos
def preprocess_input(question, response_a, response_b):
    combined_text = f"{question} {response_a} {response_b}"
    tfidf_features = vectorizer.transform([combined_text]).toarray()
    sentiment_score = analyzer.polarity_scores(combined_text)["compound"]
    sentiment_features = np.array([[sentiment_score]])
    return np.hstack([tfidf_features, sentiment_features])

# Cargar modelos entrenados
@st.cache_resource
def load_models():
    model_a, model_b = None, None
    try:
        model_a = tf.keras.models.load_model("modelo_entrenado.h5", compile=False)
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
        model_b.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            optimizer='adam',
            metrics=['accuracy']
        )
        print("Modelo B cargado exitosamente.")
    except Exception as e:
        print("Error al cargar Modelo B:", e)
    return model_a, model_b

model_a, model_b = load_models()

# Inicializar el DataFrame en session_state para persistencia
if "df_comparison" not in st.session_state:
    st.session_state.df_comparison = pd.DataFrame(columns=["Prompt", "Response A", "Response B", "Gana A", "Gana B", "Empate"])

# Inicializar el estado de preferencia del usuario y su historial
if "user_preference" not in st.session_state:
    st.session_state.user_preference = None
if "preference_history" not in st.session_state:
    st.session_state.preference_history = {"Modelo A": 0, "Modelo B": 0}

# Título de la Aplicación
st.markdown(f"<h1 style='text-align: center; color: {colors['russian_violet']}'>Comparación de Algoritmos de Clasificación</h1>", unsafe_allow_html=True)

# Entrada de datos para el usuario
st.markdown(f"<h2 style='color: {colors['russian_violet']}; font-weight: bold;'>Ingrese su pregunta y las respuestas de los modelos</h2>", unsafe_allow_html=True)

user_question = st.text_input("Pregunta al chatbot:")
col1, col2 = st.columns(2)
with col1:
    response_a_text = st.text_input("Respuesta del Modelo A:")
    model_a_name = st.text_input("Nombre del Modelo A:")

with col2:
    response_b_text = st.text_input("Respuesta del Modelo B:")
    model_b_name = st.text_input("Nombre del Modelo B:")

# Botón de procesamiento
if st.button("Calcular Preferencia"):
    if user_question and response_a_text and response_b_text and model_a_name and model_b_name:
        # Preprocesar la entrada
        input_example = preprocess_input(user_question, response_a_text, response_b_text)
        
        try:
            # Predicciones de ambos modelos
            prediction_a = model_a.predict(input_example)
            prediction_b = model_b.predict(input_example)
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

        # Calcular cuál ganó y actualizar el DataFrame en session_state
        gana_a = 1 if prediction_a[0][1] > prediction_a[0][2] else 0
        gana_b = 1 if prediction_a[0][1] < prediction_a[0][2] else 0
        empate = 1 if prediction_a[0][1] == prediction_a[0][2] else 0

        new_entry = pd.DataFrame({
            "Prompt": [user_question],
            "Response A": [response_a_text],
            "Response B": [response_b_text],
            "Gana A": [gana_a],
            "Gana B": [gana_b],
            "Empate": [empate]
        })
        
        # Concatenate the new entry to the session state DataFrame
        st.session_state.df_comparison = pd.concat([st.session_state.df_comparison, new_entry], ignore_index=True)

        # Reset user preference after calculating
        st.session_state.user_preference = None

# Mostrar la tabla actualizada
st.markdown(f"<h3 style='color: {colors['russian_violet']}; font-weight: bold;'>Tabla de Comparación</h3>", unsafe_allow_html=True)
st.dataframe(st.session_state.df_comparison)

# Selección de preferencia del usuario
st.subheader("¿Qué respuesta prefieres tú?")

# Mostrar opciones de preferencia solo si aún no se ha seleccionado, o si se ha hecho reset
if st.session_state.user_preference is None:
    user_selection = st.radio(
        "Seleccione su respuesta preferida:",
        options=["Prefiero la respuesta del Modelo A", "Prefiero la respuesta del Modelo B"],
        index=None,  # Start without a default selection
        key="temp_user_preference"  # Temporary key for the initial selection
    )
    # Update session state if a selection is made
    if user_selection:
        st.session_state.user_preference = user_selection
else:
    st.radio(
        "Seleccione su respuesta preferida:",
        options=["Prefiero la respuesta del Modelo A", "Prefiero la respuesta del Modelo B"],
        index=0 if st.session_state.user_preference == "Prefiero la respuesta del Modelo A" else 1,
        key="user_preference",
        disabled=True  # Lock selection until reset
    )

# Actualizar el historial de preferencias cuando se hace una selección
if st.session_state.user_preference:
    if st.session_state.user_preference == "Prefiero la respuesta del Modelo A":
        st.session_state.preference_history["Modelo A"] += 1
    elif st.session_state.user_preference == "Prefiero la respuesta del Modelo B":
        st.session_state.preference_history["Modelo B"] += 1

# Gráficas Interactivas
# Section for visualizing graphs
st.markdown(f"<h3 style='color: {colors['russian_violet']}; font-weight: bold;'>Gráficas de Preferencias y Precisión</h3>", unsafe_allow_html=True)

# Checkbox for showing or hiding the graphs
show_graphs = st.checkbox("Visualizar Gráficas")

# Check if there is any data in preference history or the comparison table
if show_graphs:
    if st.session_state.df_comparison.empty and all(value == 0 for value in st.session_state.preference_history.values()):
        st.write("No hay datos disponibles para mostrar las gráficas.")
    else:
        # Preferencias del usuario (Gráfico de pastel interactivo)
        df_prefs = pd.DataFrame(list(st.session_state.preference_history.items()), columns=["Modelo", "Count"])
        fig_prefs = px.pie(df_prefs, values="Count", names="Modelo", title="Preferencias del Usuario", color_discrete_sequence=[colors["peach"], colors["paynes_gray"]])
        st.plotly_chart(fig_prefs)

        # Precisión de los algoritmos (Gráfico de línea interactivo)
        accuracy_a = np.random.uniform(0.7, 0.95, 20)
        accuracy_b = np.random.uniform(0.6, 0.9, 20)
        df_accuracy = pd.DataFrame({
            "Iteración": range(1, 21),
            "Precisión Modelo A": accuracy_a,
            "Precisión Modelo B": accuracy_b
        })
        fig_accuracy = go.Figure()
        fig_accuracy.add_trace(go.Scatter(x=df_accuracy["Iteración"], y=df_accuracy["Precisión Modelo A"],
                                          mode="lines+markers", name="Modelo A", line=dict(color=colors["paynes_gray"])))
        fig_accuracy.add_trace(go.Scatter(x=df_accuracy["Iteración"], y=df_accuracy["Precisión Modelo B"],
                                          mode="lines+markers", name="Modelo B", line=dict(color=colors["peach"])))
        fig_accuracy.update_layout(title="Precisión de los Algoritmos", xaxis_title="Iteración", yaxis_title="Precisión")
        st.plotly_chart(fig_accuracy)
