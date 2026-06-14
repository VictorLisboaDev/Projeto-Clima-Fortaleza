# src/gerar_dados_climaticos.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configuração dos fenômenos climáticos por ano
fenomenos = {
    2016: 'El Niño Forte',
    2017: 'Neutro',
    2018: 'El Niño Fraco',
    2019: 'Neutro',
    2020: 'La Niña Moderada',
    2021: 'La Niña Forte',
    2022: 'La Niña Fraca',
    2023: 'El Niño Moderado',
    2024: 'El Niño Forte',
    2025: 'Neutro',
    2026: 'La Niña Fraca'  # até junho/2026
}

# Parâmetros base por fenômeno
parametros_climaticos = {
    'El Niño Forte': {'precip_media': 85, 'precip_std': 35, 'temp_media': 28.5, 'temp_std': 1.8, 'umidade': 68},
    'El Niño Moderado': {'precip_media': 95, 'precip_std': 40, 'temp_media': 28.2, 'temp_std': 1.7, 'umidade': 70},
    'El Niño Fraco': {'precip_media': 105, 'precip_std': 45, 'temp_media': 27.9, 'temp_std': 1.6, 'umidade': 72},
    'Neutro': {'precip_media': 120, 'precip_std': 50, 'temp_media': 27.5, 'temp_std': 1.5, 'umidade': 74},
    'La Niña Fraca': {'precip_media': 140, 'precip_std': 55, 'temp_media': 27.2, 'temp_std': 1.4, 'umidade': 76},
    'La Niña Moderada': {'precip_media': 155, 'precip_std': 60, 'temp_media': 27.0, 'temp_std': 1.3, 'umidade': 78},
    'La Niña Forte': {'precip_media': 170, 'precip_std': 65, 'temp_media': 26.8, 'temp_std': 1.2, 'umidade': 80}
}

# Fator sazonal mensal (chuva e temperatura)
fator_sazonal_precip = {
    1: 0.6, 2: 1.4, 3: 1.8, 4: 1.6, 5: 1.2, 6: 0.8,
    7: 0.4, 8: 0.2, 9: 0.1, 10: 0.2, 11: 0.3, 12: 0.5
}

fator_sazonal_temp = {
    1: -0.8, 2: -0.5, 3: -0.2, 4: 0.0, 5: 0.3, 6: 0.5,
    7: 0.7, 8: 0.8, 9: 0.6, 10: 0.4, 11: 0.0, 12: -0.3
}

# Gerar dados
np.random.seed(42)  # Para reprodutibilidade

data_inicio = datetime(2016, 1, 1)
data_fim = datetime(2026, 6, 30)  # até junho de 2026
datas = pd.date_range(start=data_inicio, end=data_fim, freq='D')

dados = []

for data in datas:
    ano = data.year
    mes = data.month
    fen = fenomenos[ano]
    params = parametros_climaticos[fen]
    
    # Precipitação diária (mm)
    precip_base = np.random.gamma(shape=2, scale=params['precip_media']/2)
    precip_sazonal = precip_base * fator_sazonal_precip[mes]
    precipitacao = max(0, np.random.normal(precip_sazonal, params['precip_std']/3))
    
    # Temperatura (°C)
    temp_base = np.random.normal(params['temp_media'], params['temp_std']/2)
    temperatura = temp_base + fator_sazonal_temp[mes]
    temperatura = max(22, min(34, temperatura))  # limites realistas
    
    # Umidade relativa (%)
    umidade_base = np.random.normal(params['umidade'], 5)
    umidade = max(40, min(95, umidade_base))
    
    # Vento (km/h) - dados médios de Fortaleza
    vento = np.random.gamma(shape=3, scale=3) + 8
    
    # Pressão atmosférica (hPa)
    pressao = np.random.normal(1012, 3)
    
    dados.append({
        'data': data,
        'ano': ano,
        'mes': mes,
        'dia': data.day,
        'fenomeno_climatico': fen,
        'precipitacao_mm': round(precipitacao, 1),
        'temperatura_c': round(temperatura, 1),
        'umidade_percent': round(umidade, 1),
        'vento_kmh': round(vento, 1),
        'pressao_hpa': round(pressao, 1),
        'estacao': 'Chuvosa' if mes in [2,3,4,5] else 'Seca' if mes in [7,8,9,10,11] else 'Transição'
    })

df = pd.DataFrame(dados)
df.to_csv('clima_fortaleza_2016_2026.csv', index=False)
print(f"Dados gerados: {len(df)} registros de {df['data'].min()} a {df['data'].max()}")
print(f"\nResumo por fenômeno:")
print(df.groupby('fenomeno_climatico').agg({
    'precipitacao_mm': 'mean',
    'temperatura_c': 'mean'
}).round(1))