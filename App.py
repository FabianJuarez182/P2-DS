import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, f1_score

# Ruta del archivo CSV
csv_path = "historial_comparacion.csv"

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

# Función para cargar el historial desde el CSV con manejo de errores
@st.cache_data
def load_data():
    try:
        return pd.read_csv(csv_path)
    except pd.errors.ParserError:
        st.warning("El archivo CSV tiene un formato inconsistente. Algunas filas serán omitidas.")
        return pd.read_csv(csv_path, on_bad_lines='skip')

# Función para guardar el historial actualizado en el CSV
def save_data(df):
    df.to_csv(csv_path, index=False)

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
    except Exception as e:
        st.warning(f"Error al cargar Modelo A: {e}")

    try:
        model_b = tf.keras.models.load_model("modelo_entrenado.h5", compile=False)
        model_b.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            optimizer='adam',
            metrics=['accuracy']
        )
    except Exception as e:
        st.warning(f"Error al cargar Modelo B: {e}")
    return model_a, model_b

# Función para calcular métricas acumuladas basadas en la comparación entre modelo y preferencia del usuario
def calculate_model_user_based_metrics(df):
    accuracy_values = []
    f1_values = []
    for i in range(1, len(df) + 1):
        subset = df.iloc[:i]
        y_pred = subset.apply(lambda row: 1 if row["Gana A"] == 1 else (0 if row["Gana B"] == 1 else -1), axis=1)
        y_true = subset["User Preference"]
        accuracy_values.append(accuracy_score(y_true, y_pred))
        f1_values.append(f1_score(y_true, y_pred, average='macro'))
    return accuracy_values, f1_values

# Cargar modelos y datos
model_a, model_b = load_models()
df_comparison = load_data()

# Inicializar el estado de comparación si no existen
if "df_comparison" not in st.session_state:
    st.session_state.df_comparison = df_comparison

# Título de la Aplicación
st.markdown(f"<h1 style='text-align: center; color: {colors['russian_violet']}'>Comparación de Algoritmos de Clasificación</h1>", unsafe_allow_html=True)

# Entrada de datos para el usuario
st.markdown(f"<h2 style='color: {colors['russian_violet']}; font-weight: bold;'>Ingrese su pregunta y las respuestas de los modelos</h2>", unsafe_allow_html=True)

user_question = st.text_input("Pregunta al chatbot:")
col1, col2 = st.columns(2)
with col1:
    response_a_text = st.text_input("Respuesta del Modelo A:")
    model_a_name = st.text_input("Nombre del Modelo A:")
    show_model_a = st.checkbox("Mostrar respuesta del Modelo A")

with col2:
    response_b_text = st.text_input("Respuesta del Modelo B:")
    model_b_name = st.text_input("Nombre del Modelo B:")
    show_model_b = st.checkbox("Mostrar respuesta del Modelo B")

# Verificar que al menos un modelo esté seleccionado para habilitar el botón de "Calcular Preferencia"
calculate_button_disabled = not (show_model_a or show_model_b)

# Selección de preferencia del usuario (siempre está activo)
st.subheader("¿Qué respuesta prefieres tú?")
user_selection = st.radio(
    "Seleccione su respuesta preferida:",
    options=["Prefiero la respuesta del Modelo A", "Prefiero la respuesta del Modelo B", "Empate"],
    index=0
)

# Botón de procesamiento
if st.button("Calcular Preferencia", disabled=calculate_button_disabled):
    # Verificar si todos los campos están llenos antes de proceder
    if not user_question or not response_a_text or not response_b_text or not model_a_name or not model_b_name:
        st.warning("Por favor, complete todos los campos antes de calcular la preferencia.")
    else:
        # Preprocesar la entrada
        input_example = preprocess_input(user_question, response_a_text, response_b_text)
        
        try:
            # Realizar predicciones según los modelos seleccionados
            prediction_a = model_a.predict(input_example) if show_model_a else None
            prediction_b = model_b.predict(input_example) if show_model_b else None
            
            # Mostrar el resultado de la evaluación para cada modelo
            st.subheader("Resultados de la evaluación")
            if prediction_a is not None:
                if prediction_a[0][1] > prediction_a[0][2]:
                    st.write("La respuesta ganadora en el Modelo A es: **Respuesta del Modelo A**")
                    gana_a, gana_b, empate = 1, 0, 0
                else:
                    st.write("La respuesta ganadora en el Modelo A es: **Respuesta del Modelo B**")
                    gana_a, gana_b, empate = 0, 1, 0
            if prediction_b is not None:
                if prediction_b[0][1] > prediction_b[0][2]:
                    st.write("La respuesta ganadora en el Modelo B es: **Respuesta del Modelo A**")
                    gana_a, gana_b, empate = 1, 0, 0
                else:
                    st.write("La respuesta ganadora en el Modelo B es: **Respuesta del Modelo B**")
                    gana_a, gana_b, empate = 0, 1, 0
            if prediction_a is None and prediction_b is None:
                st.write("Ningún modelo fue seleccionado para evaluación.")
                gana_a, gana_b, empate = 0, 0, 1

        except Exception as e:
            st.error(f"Error en la predicción: {e}")
            gana_a, gana_b, empate = 0, 0, 0

        # Procesar la preferencia del usuario como valores numéricos
        user_preference = 1 if user_selection == "Prefiero la respuesta del Modelo A" else (0 if user_selection == "Prefiero la respuesta del Modelo B" else -1)

        new_entry = pd.DataFrame({
            "Prompt": [user_question],
            "Response A": [response_a_text],
            "Response B": [response_b_text],
            "Gana A": [gana_a],
            "Gana B": [gana_b],
            "Empate": [empate],
            "User Preference": [user_preference]
        })

        # Concatenar la nueva entrada al DataFrame en session_state
        st.session_state.df_comparison = pd.concat([st.session_state.df_comparison, new_entry], ignore_index=True)
        
        # Guardar el DataFrame actualizado en el CSV
        save_data(st.session_state.df_comparison)

# Cálculo de precisión y F1 acumulativo
accuracy_values, f1_values = calculate_model_user_based_metrics(st.session_state.df_comparison)

# Mostrar la tabla actualizada
st.markdown(f"<h3 style='color: {colors['russian_violet']}; font-weight: bold;'>Tabla de Comparación</h3>", unsafe_allow_html=True)
st.dataframe(st.session_state.df_comparison)

# Gráficas Interactivas
show_graphs = st.checkbox("Visualizar Gráficas")
if show_graphs and not st.session_state.df_comparison.empty:
    # Preferencias del modelo (Gráfico de pastel interactivo)
    df_model_prefs = pd.DataFrame([
        {"Modelo": "Modelo A", "Count": st.session_state.df_comparison["Gana A"].sum()},
        {"Modelo": "Modelo B", "Count": st.session_state.df_comparison["Gana B"].sum()},
        {"Modelo": "Empate", "Count": st.session_state.df_comparison["Empate"].sum()}
    ])
    fig_model_prefs = px.pie(df_model_prefs, values="Count", names="Modelo", title="Preferencias del Modelo",
                             color_discrete_sequence=[colors["peach"], colors["paynes_gray"], colors["light_blue"]])
    st.plotly_chart(fig_model_prefs)

    # Nueva gráfica de preferencias del usuario (Gráfico de pastel interactivo)
    user_pref_counts = st.session_state.df_comparison["User Preference"].value_counts().rename(index={1: "Modelo A", 0: "Modelo B", -1: "Empate"})
    df_user_prefs = pd.DataFrame({
        "Preferencia": user_pref_counts.index,
        "Count": user_pref_counts.values
    })
    fig_user_prefs = px.pie(df_user_prefs, values="Count", names="Preferencia", title="Preferencias del Usuario",
                            color_discrete_sequence=[colors["peach"], colors["paynes_gray"], colors["light_blue"]])
    st.plotly_chart(fig_user_prefs)

    # Gráfico de línea para precisión y F1-score con límites de 0 a 1
    df_accuracy = pd.DataFrame({
        "Iteración": range(1, len(accuracy_values) + 1),
        "Precisión": accuracy_values,
        "F1": f1_values
    })
    fig_accuracy = go.Figure()
    fig_accuracy.add_trace(go.Scatter(x=df_accuracy["Iteración"], y=df_accuracy["Precisión"],
                                      mode="lines+markers", name="Precisión", line=dict(color=colors["paynes_gray"])))
    fig_accuracy.add_trace(go.Scatter(x=df_accuracy["Iteración"], y=df_accuracy["F1"],
                                      mode="lines+markers", name="F1-Score", line=dict(color=colors["peach"])))
    fig_accuracy.update_layout(
        title="Precisión y F1-Score Basados en Comparación con Preferencia del Usuario",
        xaxis_title="Iteración",
        yaxis_title="Métricas",
        yaxis=dict(range=[0, 1.1])  # Establecer el rango del eje y de 0 a 1
    )
    st.plotly_chart(fig_accuracy)
