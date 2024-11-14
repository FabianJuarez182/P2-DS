import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score

# Ruta del archivo CSV
csv_path_a = "historial_comparacion_a.csv"
csv_path_b = "historial_comparacion_b.csv"

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
def preprocess_input_a(question, response_a, response_b):
    combined_text = f"{question} {response_a} {response_b}"
    tfidf_features = vectorizer.transform([combined_text]).toarray()
    sentiment_score = analyzer.polarity_scores(combined_text)["compound"]
    sentiment_features = np.array([[sentiment_score]])
    return np.hstack([tfidf_features, sentiment_features])
    
def preprocess_input_b(question, response_a, response_b):
    combined_text = f"{question} {response_a} {response_b}"
    tfidf_features = vectorizer.transform([combined_text]).toarray()
    if tfidf_features.shape[1] < 1537:
        padded_features = np.pad(tfidf_features, ((0, 0), (0, 1537 - tfidf_features.shape[1])), 'constant')
    else:
        padded_features = tfidf_features[:, :1537]
    sentiment_score = analyzer.polarity_scores(combined_text)["compound"]
    sentiment_features = np.array([[sentiment_score]])
    return np.hstack([padded_features, sentiment_features])
# Función para cargar el historial desde el CSV con manejo de errores
@st.cache_data
def load_data(path):
    try:
        return pd.read_csv(path)
    except pd.errors.ParserError:
        st.warning("El archivo CSV tiene un formato inconsistente. Algunas filas serán omitidas.")
        return pd.read_csv(path, on_bad_lines='skip')

# Función para guardar el historial actualizado en el CSV
def save_data(df, path):
    df.to_csv(path, index=False)

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
        model_b = tf.keras.models.load_model("modelo_final.h5", compile=False)
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
    for i in range(1, len(df) + 1):
        subset = df.iloc[:i]
        y_pred = subset.apply(lambda row: 1 if row["Gana A"] == 1 else (0 if row["Gana B"] == 1 else -1), axis=1)
        y_true = subset["User Preference"]
        accuracy_values.append(accuracy_score(y_true, y_pred))
    return accuracy_values

# Cargar modelos y datos
model_a, model_b = load_models()
df_comparison_a = load_data(csv_path_a)
df_comparison_b = load_data(csv_path_b)

# Inicializar el estado de comparación si no existen
if "df_comparison_a" not in st.session_state:
    st.session_state.df_comparison_a = df_comparison_a
if "df_comparison_b" not in st.session_state:
    st.session_state.df_comparison_b = df_comparison_b

# Título de la Aplicación
st.markdown(f"<h1 style='text-align: center; color: {colors['russian_violet']}'>Comparación de Algoritmos de Clasificación</h1>", unsafe_allow_html=True)

# Entrada de datos para el usuario
st.markdown(f"<h2 style='color: {colors['russian_violet']}; font-weight: bold;'>Ingrese su pregunta y las respuestas de los modelos</h2>", unsafe_allow_html=True)

user_question = st.text_input("Pregunta al chatbot:")
col1, col2 = st.columns(2)
with col1:
    response_a_text = st.text_input("Respuesta del Modelo A:")
    model_a_name = st.text_input("Nombre del Modelo A:")
    show_model_a = st.checkbox("Mostrar respuesta del Modelo TF-IDF ")

with col2:
    response_b_text = st.text_input("Respuesta del Modelo B:")
    model_b_name = st.text_input("Nombre del Modelo B:")
    show_model_b = st.checkbox("Mostrar respuesta del Modelo DistilBERT")

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
        input_example_a = preprocess_input_a(user_question, response_a_text, response_b_text)
        input_example_b = preprocess_input_b(user_question, response_a_text, response_b_text)

        try:
            # Realizar predicciones según los modelos seleccionados
            prediction_a = model_a.predict(input_example_a) if show_model_a else None
            prediction_b = model_b.predict(input_example_b) if show_model_b else None
            # Mostrar el resultado de la evaluación para cada modelo
            st.subheader("Resultados de la evaluación")
            if prediction_a is not None:
                if prediction_a[0][1] > prediction_a[0][2]:
                    st.write("La respuesta ganadora en el Modelo TF-IDF es: **Respuesta del Modelo A**")
                    gana_a, gana_b, empate = 1, 0, 0
                elif prediction_a[0][1] < prediction_a[0][2]:
                    st.write("La respuesta ganadora en el Modelo TF-IDF es: **Respuesta del Modelo B**")
                    gana_a, gana_b, empate = 0, 1, 0
                else:
                    st.write("La respuesta ganadora en el Modelo TF-IDF es: **Empate**")
                    gana_a, gana_b, empate = 0, 0, 1
            if prediction_b is not None:
                if prediction_b[0][1] > prediction_b[0][2]:
                    st.write("La respuesta ganadora en el DistilBERT es: **Respuesta del Modelo A**")
                    gana_a_2, gana_b_2, empate_2 = 1, 0, 0
                elif prediction_b[0][1] < prediction_b[0][2]:
                    st.write("La respuesta ganadora en el DistilBERT es: **Respuesta del Modelo B**")
                    gana_a_2, gana_b_2, empate_2 = 0, 1, 0
                else:
                    st.write("La respuesta ganadora en el DistilBERT es: **Empate**")
                    gana_a_2, gana_b_2, empate_2 = 0, 0, 1
        except Exception as e:
            st.error(f"Error en la predicción: {e}")
            gana_a, gana_b, empate = 0, 0, 0

        # Procesar la preferencia del usuario como valores numéricos
        user_preference = 1 if user_selection == "Prefiero la respuesta del Modelo A" else (0 if user_selection == "Prefiero la respuesta del Modelo B" else -1)
        if prediction_a is not None:
            new_entry = pd.DataFrame({
                "Prompt": [user_question],
                "Response A": [response_a_text],
                "Response B": [response_b_text],
                "Gana A": [gana_a],
                "Gana B": [gana_b],
                "Empate": [empate],
                "User Preference": [user_preference]
            })
        else:
            new_entry = pd.DataFrame({
                "Prompt": [user_question],
                "Response A": [response_a_text],
                "Response B": [response_b_text],
                "Gana A": [0],
                "Gana B": [0],
                "Empate": [1],
                "User Preference": [user_preference]
            })
        if prediction_b is not None:
            new_entry_2 = pd.DataFrame({
                "Prompt": [user_question],
                "Response A": [response_a_text],
                "Response B": [response_b_text],
                "Gana A": [gana_a_2],
                "Gana B": [gana_b_2],
                "Empate": [empate_2],
                "User Preference": [user_preference]
            })
        else:
            new_entry_2 = pd.DataFrame({
                "Prompt": [user_question],
                "Response A": [response_a_text],
                "Response B": [response_b_text],
                "Gana A": [0],
                "Gana B": [0],
                "Empate": [1],
                "User Preference": [user_preference]
            })

        # Guardado de nuevo historial en ambos DataFrames
        if show_model_a:
            st.session_state.df_comparison_a = pd.concat([st.session_state.df_comparison_a, new_entry], ignore_index=True)
            save_data(st.session_state.df_comparison_a, csv_path_a)
        if show_model_b:
            st.session_state.df_comparison_b = pd.concat([st.session_state.df_comparison_b, new_entry_2], ignore_index=True)
            save_data(st.session_state.df_comparison_b, csv_path_b)

# Cálculo de precisión
accuracy_values= calculate_model_user_based_metrics(st.session_state.df_comparison_a)
accuracy_values_2= calculate_model_user_based_metrics(st.session_state.df_comparison_b)

# Mostrar la tabla actualizada
st.markdown(f"<h3 style='color: {colors['russian_violet']}; font-weight: bold;'>Tabla de Comparación TD-IDF</h3>", unsafe_allow_html=True)
st.dataframe(st.session_state.df_comparison_a)

# Mostrar la tabla actualizada
st.markdown(f"<h3 style='color: {colors['russian_violet']}; font-weight: bold;'>Tabla de Comparación DistilBERT</h3>", unsafe_allow_html=True)
st.dataframe(st.session_state.df_comparison_b)

# Gráficas Interactivas
show_graphs = st.checkbox("Visualizar Gráficas")
if show_graphs and not st.session_state.df_comparison_a.empty or st.session_state.df_comparison_b.empty:
    # Preferencias del modelo TF-IDF (Gráfico de pastel)
    df_model_prefs_a = pd.DataFrame([
        {"Modelo": "Modelo A", "Count": st.session_state.df_comparison_a["Gana A"].sum()},
        {"Modelo": "Modelo B", "Count": st.session_state.df_comparison_a["Gana B"].sum()},
        {"Modelo": "Empate", "Count": st.session_state.df_comparison_a["Empate"].sum()}
    ])
    fig_model_prefs_a = px.pie(df_model_prefs_a, values="Count", names="Modelo", title="Preferencias del Modelo TF-IDF",
                               color_discrete_sequence=[colors["peach"], colors["paynes_gray"], colors["light_blue"]])
    st.plotly_chart(fig_model_prefs_a)

    # Preferencias del modelo DistilBERT (Gráfico de pastel)
    df_model_prefs_b = pd.DataFrame([
        {"Modelo": "Modelo A", "Count": st.session_state.df_comparison_b["Gana A"].sum()},
        {"Modelo": "Modelo B", "Count": st.session_state.df_comparison_b["Gana B"].sum()},
        {"Modelo": "Empate", "Count": st.session_state.df_comparison_b["Empate"].sum()}
    ])
    fig_model_prefs_b = px.pie(df_model_prefs_b, values="Count", names="Modelo", title="Preferencias del Modelo DistilBERT",
                               color_discrete_sequence=[colors["peach"], colors["paynes_gray"], colors["light_blue"]])
    st.plotly_chart(fig_model_prefs_b)

    # Nueva gráfica de preferencias del usuario (Gráfico de pastel interactivo)
    user_pref_counts = st.session_state.df_comparison_a["User Preference"].value_counts().rename(index={1: "Modelo A", 0: "Modelo B", -1: "Empate"})
    df_user_prefs = pd.DataFrame({
        "Preferencia": user_pref_counts.index,
        "Count": user_pref_counts.values
    })
    fig_user_prefs = px.pie(df_user_prefs, values="Count", names="Preferencia", title="Preferencias del Usuario",
                            color_discrete_sequence=[colors["peach"], colors["paynes_gray"], colors["light_blue"]])
    st.plotly_chart(fig_user_prefs)

    # Gráfico de línea para precisión de ambos modelos con límites de 0 a 1
    df_accuracy_comparison = pd.DataFrame({
        "Iteración": range(1, len(accuracy_values) + 1),
        "Precisión Modelo TF-IDF": accuracy_values,
        "Precisión Modelo DistilBERT": accuracy_values_2
    })

    fig_accuracy_comparison = go.Figure()
    fig_accuracy_comparison.add_trace(go.Scatter(
        x=df_accuracy_comparison["Iteración"], y=df_accuracy_comparison["Precisión Modelo TF-IDF"],
        mode="lines+markers", name="Precisión Modelo TF-IDF", line=dict(color=colors["paynes_gray"])
    ))
    fig_accuracy_comparison.add_trace(go.Scatter(
        x=df_accuracy_comparison["Iteración"], y=df_accuracy_comparison["Precisión Modelo DistilBERT"],
        mode="lines+markers", name="Precisión Modelo DistilBERT", line=dict(color=colors["peach"])
    ))
    fig_accuracy_comparison.update_layout(
        title={
            "text": "Comparación de Precisión entre Modelo TF-IDF y Modelo DistilBERT",
            "x": 0.5,
            "y": 0.85,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 16}
        },
        xaxis_title="Iteración",
        yaxis_title="Precisión",
        yaxis=dict(range=[0, 1.1])
    )
    st.plotly_chart(fig_accuracy_comparison)