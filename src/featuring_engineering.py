import pandas as pd
import numpy as np
from datetime import datetime

def criar_features_climaticas(df):
    """
    Cria features temporais e defasagens para modelagem preditiva
    """
    df = df.copy()
    
    # Features temporais básicas
    df['data'] = pd.to_datetime(df['data'])
    df['dia_ano'] = df['data'].dt.dayofyear
    df['mes_seno'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cosseno'] = np.cos(2 * np.pi * df['mes'] / 12)
    df['dia_semana'] = df['data'].dt.dayofweek
    df['fim_semana'] = (df['dia_semana'] >= 5).astype(int)
    
    # Features climáticas defasadas (últimos 7 dias)
    for lag in [1, 2, 3, 7]:
        df[f'precip_lag_{lag}'] = df['precipitacao_mm'].shift(lag)
        df[f'temp_lag_{lag}'] = df['temperatura_c'].shift(lag)
        df[f'umidade_lag_{lag}'] = df['umidade_percent'].shift(lag)
    
    # Médias móveis (janelas de 3, 7 e 14 dias)
    for window in [3, 7, 14]:
        df[f'precip_media_{window}d'] = df['precipitacao_mm'].rolling(window=window, min_periods=1).mean()
        df[f'temp_media_{window}d'] = df['temperatura_c'].rolling(window=window, min_periods=1).mean()
        df[f'umidade_media_{window}d'] = df['umidade_percent'].rolling(window=window, min_periods=1).mean()
    
    # Tendência (diferença entre dias consecutivos)
    df['temp_tendencia'] = df['temperatura_c'].diff(1)
    df['precip_tendencia'] = df['precipitacao_mm'].diff(1)
    
    # Features de fenômeno climático (one-hot encoding)
    df = pd.get_dummies(df, columns=['fenomeno_climatico'], prefix='fen')
    
    # Estação do ano (codificada)
    estacao_map = {'Chuvosa': 2, 'Transição': 1, 'Seca': 0}
    df['estacao_cod'] = df['estacao'].map(estacao_map)
    
    # Target para classificação (chuva forte)
    df['chuva_forte'] = (df['precipitacao_mm'] >= 5).astype(int)
    
    # Remover linhas com NaN (causados pelos shifts e rolling)
    df = df.dropna()
    
    return df

# Exemplo de uso
if __name__ == "__main__":
    # Carregar dados originais
    df = pd.read_csv('clima_fortaleza_2016_2026.csv', parse_dates=['data'])
    
    # Criar features
    df_features = criar_features_climaticas(df)
    
    # Salvar dataset processado
    df_features.to_csv('clima_features.csv', index=False)
    print(f"Features criadas: {df_features.shape}")
    print(f"Colunas disponíveis: {df_features.columns.tolist()}")