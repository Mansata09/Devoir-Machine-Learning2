
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Chemin des modèles
DOSSIER = Path(__file__).resolve().parent.parent / "models"


# Chargement des modèles
@st.cache_resource
def charger():
    modele = joblib.load(DOSSIER / "rf_model_reg2.joblib")
    encoders = joblib.load(DOSSIER / "encoders_r2.joblib")
    return modele, encoders


modele, encoders = charger()


# Colonnes catégorielles
COLONNES_ENCODEES = [
    "Fuel_Type",
    "Seller_Type",
    "Transmission"
]


# Fonction de préparation des données
def preparer_donnees(df):

    df = df.copy()

    # Correspondances des variables catégorielles
    correspondances = {
        "Fuel_Type": {
            "Petrol": 2,
            "Diesel": 1,
            "CNG": 0
        },
        "Seller_Type": {
            "Dealer": 0,
            "Individual": 1
        },
        "Transmission": {
            "Manual": 1,
            "Automatic": 0
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

    # Respecter l'ordre des colonnes du modèle
    if hasattr(modele, "feature_names_in_"):
        df = df[list(modele.feature_names_in_)]

    return df

# Interface Streamlit
st.title("Prix de vente d'une voiture")

tab_simple, tab_multiple = st.tabs(
    ["Prédiction simple", "Prédiction multiple"]
)


# ==================================
# 1. PREDICTION SIMPLE
# ==================================

with tab_simple:

    st.subheader("Informations sur la voiture")

    kms = st.number_input(
        "Kilométrage",
        min_value=0,
        value=30000
    )

    present = st.number_input(
        "Prix actuel",
        min_value=0.0,
        value=5.0
    )

    age = st.number_input(
        "Age de la voiture",
        min_value=0,
        max_value=40,
        value=5
    )

    fuel = st.selectbox(
        "Type de carburant",
        ["Petrol", "Diesel", "CNG"]
    )

    seller = st.selectbox(
        "Type de vendeur",
        ["Dealer", "Individual"]
    )

    trans = st.selectbox(
        "Transmission",
        ["Manual", "Automatic"]
    )

    if st.button("Prédire le prix"):

        ligne = pd.DataFrame([{
            "Kms_Driven": kms,
            "Present_Price": present,
            "Fuel_Type": fuel,
            "Seller_Type": seller,
            "Transmission": trans,
            "Age": age
        }])

        try:

            ligne_preparee = preparer_donnees(ligne)

            prix = modele.predict(ligne_preparee)[0]

            st.success(
                f"Prix estimé : {prix:.2f}"
            )

        except Exception as erreur:

            st.error(f"Erreur : {erreur}")


# ==================================
# 2. PREDICTION MULTIPLE
# ==================================

with tab_multiple:

    st.subheader("Prédiction à partir d'un fichier CSV")

    COLONNES = [
        "Kms_Driven",
        "Present_Price",
        "Fuel_Type",
        "Seller_Type",
        "Transmission",
        "Age"
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

                df["Prix_predit"] = modele.predict(donnees)

                st.success(
                    "Prédictions réalisées avec succès !"
                )

                st.dataframe(df)

                st.download_button(
                    label="Télécharger les résultats",
                    data=df.to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name="predictions_voitures.csv",
                    mime="text/csv"
                )

            except Exception as erreur:

                st.error(f"Erreur : {erreur}")
