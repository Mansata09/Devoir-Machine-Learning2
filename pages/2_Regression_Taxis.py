
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Chemin des modèles
DOSSIER = Path(__file__).resolve().parent.parent / "models"

NOM_MODELE = "rf_model_reg1"


# Chargement du modèle et des encodeurs
@st.cache_resource
def charger():

    modele = joblib.load(
        DOSSIER / f"{NOM_MODELE}.joblib"
    )

    encoders = joblib.load(
        DOSSIER / "encoders_r1.joblib"
    )

    return modele, encoders


modele, encoders = charger()


# Colonnes catégorielles
COLONNES_ENCODEES = [
    "Time_of_Day",
    "Day_of_Week",
    "Traffic_Conditions",
    "Weather"
]


# Fonction de préparation des données
def preparer_donnees(df):

    df = df.copy()

    correspondances = {
        "Time_of_Day": {
            "Morning": 2,
            "Afternoon": 0,
            "Evening": 1,
            "Night": 3
        },

        "Day_of_Week": {
            "Weekday": 0,
            "Weekend": 1
        },

        "Traffic_Conditions": {
            "Low": 0,
            "Medium": 1,
            "High": 2
        },

        "Weather": {
            "Clear": 0,
            "Rain": 1,
            "Snow": 2
        }
    }

    for colonne, mapping in correspondances.items():

        if colonne in df.columns:

            df[colonne] = (
                df[colonne]
                .astype(str)
                .str.strip()
                .map(mapping)
            )

            if df[colonne].isna().any():
                raise ValueError(
                    f"Valeur inconnue dans {colonne}"
                )

    if hasattr(modele, "feature_names_in_"):
        df = df[list(modele.feature_names_in_)]

    return df


# Interface Streamlit
st.title("Prix d'une course de taxi ")

tab_simple, tab_multiple = st.tabs(
    ["Prédiction simple", "Prédiction multiple"]
)


# =====================================
# 1. PREDICTION SIMPLE
# =====================================

with tab_simple:

    st.subheader("Informations sur la course")

    distance = st.number_input(
        "Distance (km)",
        min_value=0.5,
        value=25.0
    )

    moment = st.selectbox(
        "Moment de la journée",
        ["Morning", "Afternoon", "Evening", "Night"]
    )

    jour = st.selectbox(
        "Jour de la semaine",
        ["Weekday", "Weekend"]
    )

    passagers = st.number_input(
        "Nombre de passagers",
        min_value=1,
        max_value=4,
        value=1
    )

    trafic = st.selectbox(
        "Conditions de circulation",
        ["Low", "Medium", "High"]
    )

    meteo = st.selectbox(
        "Météo",
        ["Clear", "Rain", "Snow"]
    )

    base = st.number_input(
        "Tarif de base",
        min_value=2.0,
        max_value=5.0,
        value=3.5
    )

    par_km = st.number_input(
        "Tarif par kilomètre",
        min_value=0.5,
        max_value=2.0,
        value=1.2
    )

    par_min = st.number_input(
        "Tarif par minute",
        min_value=0.1,
        max_value=0.5,
        value=0.3
    )

    duree = st.number_input(
        "Durée du trajet (minutes)",
        min_value=5.0,
        max_value=120.0,
        value=60.0
    )

    if st.button("Prédire le prix"):

        ligne = pd.DataFrame([{

            "Trip_Distance_km": distance,
            "Time_of_Day": moment,
            "Day_of_Week": jour,
            "Passenger_Count": passagers,
            "Traffic_Conditions": trafic,
            "Weather": meteo,
            "Base_Fare": base,
            "Per_Km_Rate": par_km,
            "Per_Minute_Rate": par_min,
            "Trip_Duration_Minutes": duree

        }])

        try:

            ligne_preparee = preparer_donnees(ligne)

            prix = modele.predict(ligne_preparee)[0]

            st.success(
                f"Prix estimé de la course : {prix:.2f}"
            )

        except Exception as erreur:

            st.error(f"Erreur : {erreur}")


# =====================================
# 2. PREDICTION MULTIPLE
# =====================================

with tab_multiple:

    st.subheader("Prédiction à partir d'un fichier CSV")

    COLONNES = [
        "Trip_Distance_km",
        "Time_of_Day",
        "Day_of_Week",
        "Passenger_Count",
        "Traffic_Conditions",
        "Weather",
        "Base_Fare",
        "Per_Km_Rate",
        "Per_Minute_Rate",
        "Trip_Duration_Minutes"
    ]

    fichier = st.file_uploader(
        "Importer un fichier CSV",
        type=["csv"]
    )

    if fichier is not None:

        df = pd.read_csv(
            fichier,
            sep=None,
            engine="python"
        )

        manquantes = [
            c for c in COLONNES if c not in df.columns
        ]

        if manquantes:

            st.error(
                f"Colonnes manquantes : {manquantes}"
            )

        elif df[COLONNES].isna().any().any():

            st.error(
                "Le fichier contient des valeurs vides."
            )

        else:

            try:

                donnees = preparer_donnees(
                    df[COLONNES]
                )

                df["Prix_predit"] = modele.predict(
                    donnees
                )

                st.success(
                    "Prédictions réalisées avec succès !"
                )

                st.dataframe(df)

                st.download_button(
                    label="Télécharger les résultats",
                    data=df.to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name="predictions_taxis.csv",
                    mime="text/csv"
                )

            except Exception as erreur:

                st.error(f"Erreur : {erreur}")
