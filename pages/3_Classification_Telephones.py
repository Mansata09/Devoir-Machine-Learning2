
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

DOSSIER = Path(__file__).resolve().parent.parent / "models"

# Chargement des fichiers
@st.cache_resource
def charger():

    modele = joblib.load(DOSSIER / "rf_model_clas1.joblib")
    encoders = joblib.load(DOSSIER / "encoders_clas1.joblib")
    scaler = joblib.load(DOSSIER / "scaler_clas1.joblib")

    return modele, encoders, scaler


@st.cache_data
def listes():

    d = pd.read_csv(
        DOSSIER / "Telephone_data.csv",
        sep=None,
        engine="python"
    )

    adresses = sorted(d["adresse"].dropna().astype(str).unique())
    marques = sorted(d["marque"].dropna().astype(str).unique())

    return adresses, marques


modele, encoders, scaler = charger()
ADRESSES, MARQUES = listes()


COLONNES = [
    "prix",
    "adresse",
    "marque",
    "dim_ecr",
    "ram",
    "stockage"
]


# Préparation des données

def preparer_donnees(df):

    df = df.copy()

    # Correspondances exactes retrouvées dans le notebook
    adresses = {
        "Almadies": 0,
        "Almadies 2": 1,
        "Biscuiterie": 2,
        "Castor": 3,
        "Cité Damel": 4,
        "Colobane": 5,
        "Dalifort": 6,
        "Diamaguene": 7,
        "Dieuppeul": 8,
        "Fann": 9,
        "Fass": 10,
        "Fenêtre mermoz": 11,
        "Grand-Yoff": 12,
        "Guediawaye": 13,
        "Gueule-Tapée": 14,
        "Hann maristes": 15,
        "Hlm": 16,
        "Hlm grand-yoff": 17,
        "Keur ndiaye lô": 18,
        "Liberte 5": 19,
        "Liberte 6": 20,
        "Liberte 6 extension": 21,
        "Mermoz": 22,
        "Médina": 23,
        "Ngaparou": 24,
        "Ngor": 25,
        "Nord foire": 26,
        "Ouakam": 27,
        "Ouest foire": 28,
        "Parcelles Assainies": 29,
        "Patte d‘oie": 30,
        "Pikine": 31,
        "Plateau": 32,
        "Point-e": 33,
        "Sacré-cœur": 34,
        "Saly": 35,
        "Sicap Liberté": 36,
        "Sicap Sacré-cœur": 37,
        "Sicap foire": 38,
        "VDN": 39,
        "Virage": 40,
        "Yoff": 41,
        "Zone industrielle": 42
    }

    marques = {
        "Alcatel": 0,
        "Asus": 1,
        "Atouch": 2,
        "Autres": 3,
        "BlackBerry": 4,
        "Crosscall": 5,
        "Google": 6,
        "Honor": 7,
        "Huawei": 8,
        "Ice-Phone": 9,
        "Infinix": 10,
        "Itel": 11,
        "LG": 12,
        "Motorola": 13,
        "Nodis": 14,
        "Nokia": 15,
        "OnePlus": 16,
        "Oppo": 17,
        "Panasonic": 18,
        "Samsung": 19,
        "Sharp": 20,
        "Sony": 21,
        "Tecno": 22,
        "Viwa": 23,
        "Xiaomi": 24,
        "ZTE": 25,
        "iPad": 26,
        "iPhone": 27
    }

    df["adresse"] = (
        df["adresse"].astype(str).str.strip().map(adresses)
    )

    df["marque"] = (
        df["marque"].astype(str).str.strip().map(marques)
    )

    if df[["adresse", "marque"]].isna().any().any():
        raise ValueError(
            "Adresse ou marque inconnue. Vérifie les noms dans le CSV."
        )

    df = df[COLONNES]

    # Normalisation des données
    df = scaler.transform(df)

    return df



st.title("Classification de l'état d'un téléphone")

tab_simple, tab_multiple = st.tabs([
    "Prédiction simple",
    "Prédiction multiple"
])


# ==================================
# PRÉDICTION SIMPLE
# ==================================

with tab_simple:

    st.subheader("Informations du téléphone")

    prix = st.number_input(
        "Prix (FCFA)",
        min_value=0,
        value=300000,
        step=10000
    )

    adresse = st.selectbox("Adresse", ADRESSES)

    marque = st.selectbox("Marque", MARQUES)

    dim_ecr = st.number_input(
        "Dimension de l'écran (pouces)",
        min_value=0.0,
        value=6.0
    )

    ram = st.number_input(
        "RAM (Go)",
        min_value=1,
        value=4
    )

    stockage = st.number_input(
        "Stockage (Go)",
        min_value=1,
        value=128
    )

    if st.button("Prédire"):

        ligne = pd.DataFrame([{
            "prix": prix,
            "adresse": adresse,
            "marque": marque,
            "dim_ecr": dim_ecr,
            "ram": ram,
            "stockage": stockage
        }])

        try:

            ligne_preparee = preparer_donnees(ligne)

            prediction = modele.predict(ligne_preparee)[0]

            st.success(f"État prédit : {prediction}")

            if hasattr(modele, "predict_proba"):

                probabilites = modele.predict_proba(
                    ligne_preparee
                )

                probas = pd.DataFrame(
                    probabilites,
                    columns=modele.classes_
                ).round(3)

                st.write("Probabilité de chaque classe :")
                st.dataframe(probas)

        except Exception as erreur:

            st.error(f"Erreur : {erreur}")


# ==================================
# PRÉDICTION MULTIPLE
# ==================================

with tab_multiple:

    st.subheader("Classification de plusieurs téléphones")

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
            c for c in COLONNES if c not in df.columns
        ]

        if manquantes:

            st.error(f"Colonnes manquantes : {manquantes}")

        elif df[COLONNES].isna().any().any():

            st.error("Le fichier contient des valeurs vides.")

        else:

            try:

                donnees = preparer_donnees(df[COLONNES])

                df["Etat_predit"] = modele.predict(donnees)

                st.success("Classification terminée !")

                st.dataframe(df)

                st.download_button(
                    "Télécharger les résultats",
                    df.to_csv(index=False).encode("utf-8"),
                    "predictions_telephones.csv",
                    "text/csv"
                )

            except Exception as erreur:

                st.error(f"Erreur : {erreur}")
