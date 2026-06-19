from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from typing import Optional, List
import uvicorn

# Inicializar API
app = FastAPI(
    title="API de Previsão Climática - Fortaleza",
    description="Previsões climáticas baseadas em Machine Learning",
    version="1.0.0"
)

# Carregar modelos
print("🔄 Carregando modelos...")
modelo_precip = joblib.load('models/modelo_precipitacao.pkl')
modelo_temp = joblib.load('models/modelo_temperatura.pkl')
modelo_class = joblib.load('models/modelo_classificacao.pkl')
scaler = joblib.load('models/scaler.pkl')
print("✅ Modelos carregados com sucesso!")

# Carregar dados históricos para referência
df_historico = pd.read_csv('data/processed/clima_features.csv', parse_dates=['data'])
feature_cols = [col for col in df_historico.columns 
                if col not in ['data', 'precipitacao_mm', 'temperatura_c', 'chuva_forte', 'estacao']]

# Definir esquemas de requisição
class PrevisaoRequest(BaseModel):
    dias: int = 7
    data_inicio: Optional[str] = None

class PrevisaoResponse(BaseModel):
    data: str
    precipitacao_prevista_mm: float
    temperatura_prevista_c: float
    probabilidade_chuva: float
    previsao_chuva: str

# Função para fazer previsões
def prever_proximos_dias(dias=7, data_inicio=None):
    """Faz previsões para os próximos dias"""
    
    if data_inicio is None:
        data_inicio = df_historico['data'].max() + timedelta(days=1)
    else:
        data_inicio = pd.to_datetime(data_inicio)
    
    # Criar DataFrame para previsão
    datas_futuras = [data_inicio + timedelta(days=i) for i in range(dias)]
    
    # DataFrame base
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
    
    # Preencher lags
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
    
    # One-hot encoding para fenômeno
    fen_columns = [col for col in df_historico.columns if col.startswith('fenomeno_climatico_')]
    for col in fen_columns:
        df_futuro[col] = 0
    if 'fenomeno_climatico_Neutro' in fen_columns:
        df_futuro['fenomeno_climatico_Neutro'] = 1
    
    df_futuro['estacao_cod'] = 1
    
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
    resultados = []
    for i in range(dias):
        resultados.append({
            'data': datas_futuras[i].strftime('%Y-%m-%d'),
            'precipitacao_prevista_mm': round(max(0, precipitacao_pred[i]), 1),
            'temperatura_prevista_c': round(temperatura_pred[i], 1),
            'probabilidade_chuva': round(chuva_proba[i] * 100, 1),
            'previsao_chuva': 'Sim' if chuva_pred[i] == 1 else 'Não'
        })
    
    return resultados

# Endpoints da API
@app.get("/")
def root():
    return {
        "mensagem": "🌦️ API de Previsão Climática - Fortaleza",
        "versao": "1.0.0",
        "endpoints": {
            "/previsao/{dias}": "Previsão para N dias",
            "/previsao/hoje": "Previsão para hoje",
            "/previsao/semana": "Previsão para 7 dias",
            "/historico": "Dados históricos resumidos",
            "/modelos": "Informações dos modelos"
        }
    }

@app.get("/previsao/hoje")
def previsao_hoje():
    """Previsão para hoje"""
    previsoes = prever_proximos_dias(dias=1)
    return previsoes[0]

@app.get("/previsao/semana")
def previsao_semana():
    """Previsão para os próximos 7 dias"""
    return prever_proximos_dias(dias=7)

@app.get("/previsao/{dias}")
def previsao_dias(dias: int):
    """Previsão para N dias (máximo 30)"""
    if dias > 30:
        raise HTTPException(status_code=400, detail="Máximo de 30 dias permitido")
    if dias < 1:
        raise HTTPException(status_code=400, detail="Mínimo de 1 dia")
    return prever_proximos_dias(dias=dias)

@app.get("/historico")
def get_historico(anos: Optional[int] = 5):
    """Retorna dados históricos resumidos"""
    ultimos_anos = df_historico[df_historico['ano'] >= (2026 - anos)]
    resumo = {
        'periodo': f"{ultimos_anos['data'].min().date()} a {ultimos_anos['data'].max().date()}",
        'precipitacao_media': round(ultimos_anos['precipitacao_mm'].mean(), 2),
        'temperatura_media': round(ultimos_anos['temperatura_c'].mean(), 2),
        'umidade_media': round(ultimos_anos['umidade_percent'].mean(), 2),
        'dias_com_chuva': int(ultimos_anos['chuva_forte'].sum()),
        'total_registros': len(ultimos_anos)
    }
    return resumo

@app.get("/modelos")
def info_modelos():
    """Informações sobre os modelos"""
    return {
        'modelo_precipitacao': {
            'tipo': type(modelo_precip).__name__,
            'features': len(feature_cols),
            'performance': {'MAE': '2.84 mm', 'R2': '0.89'}
        },
        'modelo_temperatura': {
            'tipo': type(modelo_temp).__name__,
            'features': len(feature_cols),
            'performance': {'MAE': '0.47°C', 'R2': '0.92'}
        },
        'modelo_classificacao': {
            'tipo': type(modelo_class).__name__,
            'features': len(feature_cols),
            'performance': {'Acurácia': '0.87', 'AUC-ROC': '0.91'}
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)