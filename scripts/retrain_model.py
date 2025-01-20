import os
import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# Define the paths for your reference and production data
REF_DATA_PATH = "../data/ref_data.csv"
PROD_DATA_PATH = "../data/prod_data.csv"

# Define the path for your model
MODEL_PATH = "../artifacts/model_xgb.json"

# Variable pour surveiller la taille précédente du fichier prod_data.csv
last_row_count = 0
RETRAIN_THRESHOLD = 5  # Nombre de nouvelles lignes avant réentraînement

def retrain_with_prod_data(ref_data_csv, prod_data_csv, model_path, save_new_model_path=None):
    """
    Lit les données de ref_data.csv et prod_data.csv, évalue le modèle actuel,
    réentraîne le modèle avec les données combinées, et remplace l'ancien modèle
    si les performances s'améliorent.

    Paramètres :
        ref_data_csv : chemin du CSV de référence (ref_data.csv)
        prod_data_csv : chemin du CSV de production (prod_data.csv)
        model_path : chemin du modèle XGBoost actuel (model_xgb.json)
        save_new_model_path : chemin où sauvegarder le nouveau modèle
                              (si None, on remplace directement model_path)

    Retourne :
        (old_accuracy, new_accuracy) : tuple de précision avant/après réentraînement
    """
    # Charger le modèle XGBoost existant
    model_xgb = xgb.XGBClassifier()
    model_xgb.load_model(model_path)  # Charger le modèle au format JSON

    # Lire les données
    df_ref = pd.read_csv(ref_data_csv)
    df_prod = pd.read_csv(prod_data_csv)

    # Vérifier la présence des colonnes attendues
    expected_ref_cols = [str(i) for i in range(100)] + ["label"]
    expected_prod_cols = [str(i) for i in range(100)] + ["label", "prediction"]

    if not all(col in df_ref.columns for col in expected_ref_cols):
        raise ValueError("Les colonnes de ref_data.csv ne correspondent pas au format attendu.")
    if not all(col in df_prod.columns for col in expected_prod_cols):
        raise ValueError("Les colonnes de prod_data.csv ne correspondent pas au format attendu.")

    # Séparer les features et le label
    X_ref = df_ref[[str(i) for i in range(100)]].values
    y_ref = df_ref["label"].values

    X_prod = df_prod[[str(i) for i in range(100)]].values
    y_prod = df_prod["label"].values

    # Évaluer les performances de l'ancien modèle sur la prod
    y_pred_old = model_xgb.predict(X_prod)
    old_accuracy = accuracy_score(y_prod, y_pred_old)
    print(f"\n[INFO] Ancien modèle - Accuracy sur prod_data.csv : {old_accuracy*100:.2f}%")

    # Concaténer ref_data et prod_data pour le nouvel entraînement
    X_combined = np.vstack((X_ref, X_prod))
    y_combined = np.concatenate((y_ref, y_prod))

    # Créer un nouveau modèle basé sur les mêmes hyperparamètres
    new_model_xgb = xgb.XGBClassifier(**model_xgb.get_params())
    if 'device' in new_model_xgb.get_params():
        new_model_xgb.set_params(device='cpu')  # ou 'gpu' si GPU disponible

    # Entraîner le nouveau modèle
    print("[INFO] Entraînement du nouveau modèle sur données combinées...")
    new_model_xgb.fit(X_combined, y_combined)

    # Évaluer le nouveau modèle sur les mêmes données de production
    y_pred_new = new_model_xgb.predict(X_prod)
    new_accuracy = accuracy_score(y_prod, y_pred_new)
    print(f"[INFO] Nouveau modèle - Accuracy sur prod_data.csv : {new_accuracy*100:.2f}%")
    print("[INFO] Nouveau modèle - Rapport de classification :")
    print(classification_report(y_prod, y_pred_new))

    # Comparer et remplacer si nécessaire
    if new_accuracy > old_accuracy:
        print("[INFO] Les performances se sont améliorées, on remplace l'ancien modèle.")
        final_path = save_new_model_path if save_new_model_path else model_path
        new_model_xgb.save_model(final_path)  # Sauvegarder au format JSON
    else:
        print("[INFO] Les performances n'ont pas augmenté, l'ancien modèle est conservé.")

    return old_accuracy, new_accuracy

def monitor_and_retrain():
    global last_row_count

    while True:
        try:
            # Vérifier si prod_data.csv existe
            if not os.path.exists(PROD_DATA_PATH):
                print(f"[INFO] {PROD_DATA_PATH} n'existe pas encore. Attente...")
                time.sleep(60)
                continue

            # Lire prod_data.csv
            df_prod = pd.read_csv(PROD_DATA_PATH)
            current_row_count = df_prod.shape[0]

            # Vérifier si le nombre de nouvelles lignes dépasse le seuil
            if current_row_count - last_row_count >= RETRAIN_THRESHOLD:
                print(f"\n[INFO] {current_row_count - last_row_count} nouvelles lignes détectées. Déclenchement du réentraînement...\n")
                old_acc, new_acc = retrain_with_prod_data(
                    ref_data_csv=REF_DATA_PATH,
                    prod_data_csv=PROD_DATA_PATH,
                    model_path=MODEL_PATH
                )
                print(f"[RESULT] Ancien modèle = {old_acc*100:.2f}%, Nouveau modèle = {new_acc*100:.2f}%\n")
                last_row_count = current_row_count  # Mettre à jour la dernière taille connue

            else:
                print(f"[INFO] Aucun changement significatif détecté. {current_row_count - last_row_count} lignes ajoutées.")
            
        except Exception as e:
            print(f"[ERROR] Une erreur s'est produite : {e}")

        # Attendre 5 min avant de vérifier à nouveau
        time.sleep(300)

if __name__ == "__main__":
    monitor_and_retrain()
