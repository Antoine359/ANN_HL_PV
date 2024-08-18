#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 17:58:00 2024

@author: sene
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Charger le modèle
model = tf.keras.models.load_model('/Users/sene/Documents/These_IA/DataBase public/NHANES/NHANES 2017-2018/ANN/auditory_loss_model.h5')

# Initialiser l'application Flask
app = Flask(__name__)

# Charger le scaler
scaler = StandardScaler()
# Charger les données de train pour le scaler
train_df = pd.read_excel('/Users/sene/Documents/These_IA/DataBase public/NHANES/NHANES 2017-2018/ANN/train_data.xlsx')
X_train = train_df.drop(columns=['diagnostic'])
scaler.fit(X_train)

# Afficher les noms des caractéristiques utilisées pour l'entraînement
print("Features used for training:", X_train.columns.tolist())

# Endpoint pour la prédiction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame(data, index=[0])
    
    # Vérifier que toutes les caractéristiques nécessaires sont présentes
    missing_features = set(X_train.columns) - set(df.columns)
    
    # Ajouter des caractéristiques manquantes avec une valeur par défaut (par exemple, 0)
    for feature in missing_features:
        df[feature] = 0
    
    # Réordonner les colonnes pour correspondre à l'ordre d'entraînement
    df = df[X_train.columns]
    
    # Prétraiter les données
    X = scaler.transform(df)
    
    # Faire la prédiction
    prediction = model.predict(X)
    result = (prediction > 0.5).astype(int)
    
    return jsonify({'prediction': int(result[0][0])})

if __name__ == '__main__':
    app.run(debug=True, port=5002, use_reloader=False)
    
