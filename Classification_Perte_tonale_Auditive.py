#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 17:40:00 2024

@author: sene
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Charger les ensembles d'entraînement et de test
train_df = pd.read_excel('/Users/sene/Documents/These_IA/DataBase public/NHANES/NHANES 2017-2018/ANN/train_data.xlsx')
test_df = pd.read_excel('/Users/sene/Documents/These_IA/DataBase public/NHANES/NHANES 2017-2018/ANN/test_data.xlsx')

# Séparer les caractéristiques (X) et les étiquettes (y)
X_train = train_df.drop(columns=['diagnostic'])
y_train = train_df['diagnostic']
X_test = test_df.drop(columns=['diagnostic'])
y_test = test_df['diagnostic']

# Normaliser les caractéristiques
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Créer le modèle ANN
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compiler le modèle
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Définir le callback pour l'early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Entraîner le modèle
history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, callbacks=[early_stopping])

# Évaluer le modèle sur l'ensemble de test
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Perte sur l'ensemble de test : {loss}")
print(f"Exactitude sur l'ensemble de test : {accuracy}")

# Sauvegarder le modèle
model.save('/Users/sene/Documents/These_IA/DataBase public/NHANES/NHANES 2017-2018/ANN/auditory_loss_model.h5')

# Afficher les courbes de perte et d'exactitude
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Perte d\'entraînement')
plt.plot(history.history['val_loss'], label='Perte de validation')
plt.xlabel('Épochs')
plt.ylabel('Perte')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Exactitude d\'entraînement')
plt.plot(history.history['val_accuracy'], label='Exactitude de validation')
plt.xlabel('Épochs')
plt.ylabel('Exactitude')
plt.legend()

plt.show()
