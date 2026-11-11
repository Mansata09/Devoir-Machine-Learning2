
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

DOSSIER = Path(__file__).resolve().parent.parent / "models"


# ==================================
# CHARGEMENT DES FICHIERS
# ==================================

@st.cache_resource
def charger():

    modele = joblib.load(
        DOSSIER / "rf_model_clas2.joblib"
    )

    encoders = joblib.load(
        DOSSIER / "encoders_clas2.joblib"
    )

    scaler = joblib.load(
        DOSSIER / "scaler_clas2.joblib"
    )

    correspondances = joblib.load(
        DOSSIER / "correspondances_clas2.joblib"
    )

    return modele, encoders, scaler, correspondances


modele, encoders, scaler, correspondances = charger()


# ==================================
# LISTES DES VALEURS
# ==================================

MARQUES = sorted(
    correspondances["Marque"].keys()
)

BOITES = sorted(
    correspondances["Boite_Vitesse"].keys()
)

ADRESSES = sorted(
    correspondances["Adresse"].keys()
)


# ==================================
# COLONNES DU MODELE
# ==================================

COLONNES = [
    "Marque",
    "Année",
    "Boite_Vitesse",
    "Adresse",
    "Prix"
]


# ==================================
# PREPARATION DES DONNEES
# ==================================

def preparer_donnees(df):

    df = df.copy()

    # Encodage des variables catégorielles
    for colonne in [
        "Marque",
        "Boite_Vitesse",
        "Adresse"
    ]:

        mapping = correspondances[colonne]

        df[colonne] = (
            df[colonne]
            .astype(str)
            .str.strip()
            .map(mapping)
        )

        if df[colonne].isna().any():

            valeurs = df.loc[
                df[colonne].isna()
            ]

            raise ValueError(
                f"Valeurs inconnues dans {colonne}. "
                "Vérifie les catégories."
            )

    # Conversion des variables numériques
    for colonne in [
        "Année",
        "Prix"
    ]:

        df[colonne] = pd.to_numeric(
            df[colonne],
            errors="raise"
        )

    # Respecter l'ordre des colonnes
    df = df[COLONNES]

    # Normalisation
    donnees = scaler.transform(df)

    return donnees


# ==================================
# INTERFACE STREAMLIT
# ==================================

st.title("Classification des véhicules")

st.write(
    "Prédire si un véhicule est neuf ou d'occasion."
)

tab_simple, tab_multiple = st.tabs([
    "Prédiction simple",
    "Prédiction multiple"
])


# ==================================
# PREDICTION SIMPLE
# ==================================

with tab_simple:

    st.subheader("Informations du véhicule")

    marque = st.selectbox(
        "Marque",
        MARQUES
    )

    annee = st.number_input(
        "Année",
        min_value=1990,
        max_value=2024,
        value=2015
    )

    boite = st.selectbox(
        "Boîte de vitesse",
        BOITES
    )

    adresse = st.selectbox(
        "Adresse",
        ADRESSES
    )

    prix = st.number_input(
        "Prix (FCFA)",
        min_value=850000,
        value=5000000,
        step=100000
    )

    if st.button("Prédire"):

        ligne = pd.DataFrame([{

            "Marque": marque,
            "Année": annee,
            "Boite_Vitesse": boite,
            "Adresse": adresse,
            "Prix": prix

        }])

        try:

            ligne_preparee = preparer_donnees(
                ligne
            )

            prediction = modele.predict(
                ligne_preparee
            )[0]

            # Décodage de la classe
            etat = encoders[3].inverse_transform(
                [prediction]
            )[0]

            st.success(
                f"État du véhicule : {etat}"
            )

            # Probabilités
            if hasattr(modele, "predict_proba"):

                probabilites = modele.predict_proba(
                    ligne_preparee
                )

                classes = encoders[3].inverse_transform(
                    modele.classes_.astype(int)
                )

                probas = pd.DataFrame(
                    probabilites,
                    columns=classes
                ).round(3)

                st.write(
                    "Probabilité de chaque classe :"
                )

                st.dataframe(probas)

        except Exception as erreur:

            st.error(f"Erreur : {erreur}")


# ==================================
# PREDICTION MULTIPLE
# ==================================

with tab_multiple:

    st.subheader(
        "Classification de plusieurs véhicules"
    )

    fichier = st.file_uploader(
        "Charger un fichier CSV",
        type="csv"
    )

    if fichier is not None:

        df = pd.read_csv(
            fichier,
            sep=None,
            engine="python"
        )

        manquantes = [
            c for c in COLONNES
            if c not in df.columns
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

                predictions = modele.predict(
                    donnees
                )

                # Décodage des prédictions
                df["Prediction"] = encoders[3].inverse_transform(
                    predictions.astype(int)
                )

                st.success(
                    "Classification terminée avec succès !"
                )

                st.dataframe(df)

                st.download_button(
                    "Télécharger les résultats",

                    df.to_csv(
                        index=False
                    ).encode("utf-8"),

                    "predictions_vehicules.csv",

                    "text/csv"
                )

            except Exception as erreur:

                st.error(
                    f"Erreur : {erreur}"
                )
