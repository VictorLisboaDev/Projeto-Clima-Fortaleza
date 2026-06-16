import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib

def prever_proximos_dias(dias=7, data_inicio=None):
    """
    Faz previsões para os próximos dias
    """
    # Carregar modelos
    modelo_precip = joblib.load('models/modelo_precipitacao.pkl')
    modelo_temp = joblib.load('models/modelo_temperatura.pkl')
    modelo_class = joblib.load('models/modelo_classificacao.pkl')
    scaler = joblib.load('models/scaler.pkl')
    
    # Carregar dados históricos para referência
    df_historico = pd.read_csv('data/processed/clima_features.csv', parse_dates=['data'])
    
    if data_inicio is None:
        data_inicio = df_historico['data'].max() + timedelta(days=1)
    
    # Criar DataFrame para previsão
    datas_futuras = [data_inicio + timedelta(days=i) for i in range(dias)]
    
    # DataFrame base para feature engineering
    df_futuro = pd.DataFrame({
        'data': datas_futuras,
        'ano': [d.year for d in datas_futuras],
        'mes': [d.month for d in datas_futuras],
        'dia': [d.day for d in datas_futuras]
    })
    
    # Adicionar features básicas
    df_futuro['dia_ano'] = df_futuro['data'].dt.dayofyear
    df_futuro['mes_seno'] = np.sin(2 * np.pi * df_futuro['mes'] / 12)
    df_futuro['mes_cosseno'] = np.cos(2 * np.pi * df_futuro['mes'] / 12)
    df_futuro['dia_semana'] = df_futuro['data'].dt.dayofweek
    df_futuro['fim_semana'] = (df_futuro['dia_semana'] >= 5).astype(int)
    
    # Para features defasadas, usar últimos valores históricos
    ultimos_dados = df_historico.tail(30).copy()
    
    # Preencher lags com média dos últimos 30 dias
    for lag in [1, 2, 3, 7]:
        df_futuro[f'precip_lag_{lag}'] = ultimos_dados['precipitacao_mm'].tail(lag).mean()
        df_futuro[f'temp_lag_{lag}'] = ultimos_dados['temperatura_c'].tail(lag).mean()
        df_futuro[f'umidade_lag_{lag}'] = ultimos_dados['umidade_percent'].tail(lag).mean()
    
    # Médias móveis
    for window in [3, 7, 14]:
        df_futuro[f'precip_media_{window}d'] = ultimos_dados['precipitacao_mm'].tail(window).mean()
        df_futuro[f'temp_media_{window}d'] = ultimos_dados['temperatura_c'].tail(window).mean()
        df_futuro[f'umidade_media_{window}d'] = ultimos_dados['umidade_percent'].tail(window).mean()
    
    # Tendência (aproximada)
    df_futuro['temp_tendencia'] = 0
    df_futuro['precip_tendencia'] = 0
    
    # One-hot encoding para fenômeno (usar mais comum)
    fen_columns = [col for col in df_historico.columns if col.startswith('fenomeno_climatico_')]
    for col in fen_columns:
        df_futuro[col] = 0
    # Usar 'Neutro' como padrão
    neutro_col = 'fenomeno_climatico_Neutro'
    if neutro_col in fen_columns:
        df_futuro[neutro_col] = 1
    
    df_futuro['estacao_cod'] = 1  # Transição como padrão
    
    # Selecionar apenas features usadas no treino
    feature_cols = [col for col in df_historico.columns 
                    if col not in ['data', 'precipitacao_mm', 'temperatura_c', 'chuva_forte', 'estacao']]
    
    # Garantir que todas as features existem
    for col in feature_cols:
        if col not in df_futuro.columns:
            df_futuro[col] = 0
    
    X_futuro = df_futuro[feature_cols].values
    X_futuro_scaled = scaler.transform(X_futuro)
    
    # Fazer previsões
    precipitacao_pred = modelo_precip.predict(X_futuro_scaled)
    temperatura_pred = modelo_temp.predict(X_futuro_scaled)
    chuva_pred = modelo_class.predict(X_futuro_scaled)
    chuva_proba = modelo_class.predict_proba(X_futuro_scaled)[:, 1]
    
    # Montar resultados
    resultados = pd.DataFrame({
        'data': datas_futuras,
        'precipitacao_prevista_mm': np.maximum(0, precipitacao_pred.round(1)),
        'temperatura_prevista_c': temperatura_pred.round(1),
        'probabilidade_chuva': (chuva_proba * 100).round(1),
        'previsao_chuva': ['Sim' if c == 1 else 'Não' for c in chuva_pred]
    })
    
    return resultados

# Executar previsão para os próximos 7 dias
print("="*60)
print("PREVISÃO CLIMÁTICA PARA FORTALEZA - PRÓXIMOS DIAS")
print("="*60)

previsoes = prever_proximos_dias(dias=7)
print(previsoes.to_string(index=False))

# Salvar previsões
previsoes.to_csv('reports/previsoes_proximos_dias.csv', index=False)
print("\n✅ Previsões salvas em 'reports/previsoes_proximos_dias.csv'")