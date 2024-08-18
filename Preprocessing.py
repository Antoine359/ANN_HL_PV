#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 17:25:39 2024

@author: sene
"""


import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

# Chemin vers le fichier de données combiné
combined_data_path = '/Users/sene/Documents/ChatBot/Dépot_ChatBot/ANN_HL_PV/NHANES 2017-2018final_combined_data.xlsx'

# Charger les données
df = pd.read_excel(combined_data_path)

# Colonnes de perte auditive
auditory_vars = [
    'AUXU1K1L', 'AUXU1K1R', 'AUXU1K2L', 'AUXU1K2R', 'AUXU2KL', 'AUXU2KR',
    'AUXU3KL', 'AUXU3KR', 'AUXU4KL', 'AUXU4KR', 'AUXU500L', 'AUXU500R',
    'AUXU6KL', 'AUXU6KR', 'AUXU8KL', 'AUXU8KR'
]

# Calculer la moyenne YL et YR
df['YL'] = df[['AUXU1K1L', 'AUXU1K2L', 'AUXU2KL', 'AUXU3KL', 'AUXU4KL', 'AUXU500L', 'AUXU6KL', 'AUXU8KL']].mean(axis=1)
df['YR'] = df[['AUXU1K1R', 'AUXU1K2R', 'AUXU2KR', 'AUXU3KR', 'AUXU4KR', 'AUXU500R', 'AUXU6KR', 'AUXU8KR']].mean(axis=1)

# Ajouter la colonne de diagnostic (0 = absence de perte d'audition, 1 = présence de perte d'audition)
df['diagnostic'] = ((df['YL'] > 30) | (df['YR'] > 30)).astype(int)

# Supprimer les colonnes de perte auditive et les colonnes YL et YR
df.drop(columns=auditory_vars + ['YL', 'YR'], inplace=True)

# A. Supprimer les caractéristiques avec plus de 30 % de valeurs manquantes
missing_percentage = df.isnull().mean() * 100
columns_to_drop = missing_percentage[missing_percentage > 30].index
df.drop(columns=columns_to_drop, inplace=True)

# B. Imputer les valeurs manquantes pour les caractéristiques restantes
columns_with_na = df.columns[df.isnull().any()]
imputer = SimpleImputer(strategy='mean')
df[columns_with_na] = imputer.fit_transform(df[columns_with_na])

# C. Supprimer les caractéristiques non pertinentes ou redondantes

# a. Supprimer les caractéristiques à faible variance
selector = VarianceThreshold(threshold=0.01)  # Seuil de variance très faible
selector.fit(df.drop(columns=['diagnostic']))  # Exclure la colonne 'diagnostic' lors de l'ajustement
columns_to_keep = df.drop(columns=['diagnostic']).columns[selector.get_support()]
df = df[columns_to_keep.union(['diagnostic'])]

# b. Supprimer les caractéristiques fortement corrélées
correlation_matrix = df.corr().abs()
upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95) and column != 'diagnostic']
df.drop(columns=to_drop, inplace=True)

# D. Diviser les données en ensembles d'entraînement (70 %) et de test (30 %)
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['diagnostic'])

# Sauvegarder les ensembles d'entraînement et de test dans des fichiers Excel
train_df.to_excel('/Users/sene/Documents/ChatBot/Dépot_ChatBot/ANN_HL_PV/train_data.xlsx', index=False)
test_df.to_excel('/Users/sene/Documents/ChatBot/Dépot_ChatBot/ANN_HL_PV/test_data.xlsx', index=False)

# Afficher les premières lignes du jeu de données d'entraînement et de test pour vérification
print("Ensemble d'entraînement:")
print(train_df.head())
print("\nEnsemble de test:")
print(test_df.head())
