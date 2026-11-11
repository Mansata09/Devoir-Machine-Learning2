
import streamlit as st

st.set_page_config(
    page_title="Accueil - Machine Learning",
    page_icon="",
    layout="wide"
)

# CSS
st.markdown("""
<style>

/* Fond général */
.stApp {
    background-color: #f0f4ff;
    color: #263453;
}

/* Contenu */
.block-container {
    padding-top: 3rem;
}

/* Titres */
h1 {
    color: #4338ca;
    text-align: center;
    font-size: 42px;
    font-weight: bold;
}

h2, h3 {
    color: #4f46e5;
}

/* Paragraphes */
p {
    color: #475569;
    font-size: 17px;
}

/* Barre latérale */
section[data-testid="stSidebar"] {
    background-color: #e0e7ff;
}

/* Métriques */
div[data-testid="stMetric"] {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #c7d2fe;
    box-shadow: 0 4px 12px rgba(79,70,229,0.08);
}

/* Boutons */
.stButton button {
    background-color: #6366f1;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
}

.stButton button:hover {
    background-color: #4338ca;
    color: white;
}

/* Séparateurs */
hr {
    border-color: #c7d2fe;
}

</style>
""", unsafe_allow_html=True)


# ACCUEIL
st.title("Bienvenue sur notre plateforme")

st.write(
    "Découvrez une application intelligente basée sur "
    "le Machine Learning, conçue pour analyser les données, "
    "effectuer des prédictions et faciliter la prise de décision."
)

st.divider()

st.header("Nos modèles de Machine Learning")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Régression automobile")
    st.write(
        "Estimez le prix de vente d'une voiture "
        "selon ses caractéristiques."
    )

    st.subheader("Classification des téléphones")
    st.write(
        "Prédisez l'état d'un téléphone à partir "
        "de son prix, sa marque et ses caractéristiques."
    )

with col2:
    st.subheader("Régression des taxis")
    st.write(
        "Estimez le prix d'une course en fonction "
        "de la distance, de la durée et du trafic."
    )

    st.subheader("Classification des véhicules")
    st.write(
        "Déterminez si un véhicule est venant "
        "ou d'occasion."
    )

st.divider()

st.header("Comment utiliser cette application ?")

st.info(
    "Sélectionnez un modèle dans le menu situé à gauche. "
    "Vous pourrez effectuer une prédiction individuelle "
    "ou importer un fichier CSV pour réaliser plusieurs "
    "prédictions simultanément."
)

st.success(
    "Notre objectif : rendre le Machine Learning "
    "simple, accessible et utile à tous !"
)

st.divider()

st.caption(
    "© 2026 - Application de Machine Learning | "
    "Développée avec Python, Scikit-learn et Streamlit."
)
