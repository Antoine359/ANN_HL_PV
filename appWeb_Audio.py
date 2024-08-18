#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 17:58:00 2024

@author: sene
"""

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Charger le modèle
model = tf.keras.models.load_model('auditory_loss_model.h5')

# Charger les données de train pour le scaler
train_df = pd.read_excel('train_data.xlsx')
X_train = train_df.drop(columns=['diagnostic'])
scaler = StandardScaler().fit(X_train)

# Interface utilisateur Streamlit
st.title("Prédiction de la Perte Auditive")

# Entrée utilisateur
input_data = {
    "feature1": st.number_input("Feature 1"),
    "feature2": st.number_input("Feature 2"),
    # Ajoutez toutes les autres features nécessaires ici
}

df = pd.DataFrame([input_data])
X = scaler.transform(df)

# Prédiction
if st.button("Prédire"):
    prediction = model.predict(X)
    result = (prediction > 0.5).astype(int)
    st.write(f"Résultat de la prédiction : {'Perte auditive' if result[0][0] == 1 else 'Pas de perte auditive'}")

