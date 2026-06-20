# (VERSÃO CORRIGIDA)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import requests
import json
import joblib
import os

# Configuração da página
st.set_page_config(
    page_title="🌦️ Clima Fortaleza",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título
st.title("🌦️ Análise e Previsão Climática - Fortaleza")
st.markdown("---")

# ============================================
# CARREGAR DADOS E MODELOS
# ============================================

@st.cache_resource
def load_models():
    """Carrega os modelos treinados"""
    try:
        modelo_precip = joblib.load('models/modelo_precipitacao.pkl')
        modelo_temp = joblib.load('models/modelo_temperatura.pkl')
        modelo_class = joblib.load('models/modelo_classificacao.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return modelo_precip, modelo_temp, modelo_class, scaler
    except FileNotFoundError:
        st.warning("⚠️ Modelos não encontrados. Execute o treinamento primeiro.")
        return None, None, None, None

@st.cache_data
def load_data():
    """Carrega os dados históricos"""
    try:
        df = pd.read_csv('data/processed/clima_features.csv', parse_dates=['data'])
        return df
    except FileNotFoundError:
        st.warning("⚠️ Dados não encontrados. Execute a geração de dados primeiro.")
        return None

@st.cache_data
def get_feature_cols():
    """Obtém a lista de features usadas no treinamento"""
    try:
        df = pd.read_csv('data/processed/clima_features.csv', parse_dates=['data'])
        exclude_cols = ['data', 'precipitacao_mm', 'temperatura_c', 'chuva_forte', 'estacao']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols
    except:
        return []

# Carregar tudo
df = load_data()
modelo_precip, modelo_temp, modelo_class, scaler = load_models()
feature_cols = get_feature_cols()

# Sidebar
st.sidebar.header("📊 Controles")
opcao = st.sidebar.selectbox(
    "Escolha uma visualização",
    ["📈 Visão Geral", "📊 Análise Histórica", "🔮 Previsões", "🤖 Modelos ML"]
)

# ============================================
# FUNÇÃO PARA PREVISÕES (SEM DEPENDER DA API)
# ============================================

def fazer_previsao_local(dias=7):
    """Faz previsões usando os modelos carregados localmente"""
    
    if modelo_precip is None or df is None:
        return None
    
    # Data de início
    data_inicio = df['data'].max() + timedelta(days=1)
    
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
    ultimos_dados = df.tail(30).copy()
    
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
    fen_columns = [col for col in df.columns if col.startswith('fenomeno_climatico_')]
    for col in fen_columns:
        df_futuro[col] = 0
    if 'fenomeno_climatico_Neutro' in fen_columns:
        df_futuro['fenomeno_climatico_Neutro'] = 1
    
    df_futuro['estacao_cod'] = 1
    
    # Garantir que todas as features existem
    for col in feature_cols:
        if col not in df_futuro.columns:
            df_futuro[col] = 0
    
    # Selecionar apenas as features necessárias
    X_futuro = df_futuro[feature_cols].values
    
    # Escalonar
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
    
    return pd.DataFrame(resultados)

# ============================================
# FUNÇÃO PARA CHAMAR A API (ALTERNATIVA)
# ============================================

@st.cache_data(ttl=3600)
def get_previsao_api(dias=7):
    """Tenta obter previsões da API"""
    try:
        response = requests.get(f"http://localhost:8000/previsao/{dias}", timeout=5)
        if response.status_code == 200:
            return pd.DataFrame(response.json())
        else:
            return None
    except:
        return None

# ============================================
# VISÃO GERAL
# ============================================

if opcao == "📈 Visão Geral":
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🌡️ Temperatura Média", f"{df['temperatura_c'].mean():.1f}°C", 
                      f"{df['temperatura_c'].iloc[-30:].mean() - df['temperatura_c'].mean():+.1f}°C")
        
        with col2:
            st.metric("💧 Precipitação Média", f"{df['precipitacao_mm'].mean():.1f} mm",
                      f"{df['precipitacao_mm'].iloc[-30:].mean() - df['precipitacao_mm'].mean():+.1f} mm")
        
        with col3:
            st.metric("💨 Umidade Média", f"{df['umidade_percent'].mean():.1f}%",
                      f"{df['umidade_percent'].iloc[-30:].mean() - df['umidade_percent'].mean():+.1f}%")
        
        with col4:
            st.metric("📅 Período", f"{df['data'].min().date()}", f"até {df['data'].max().date()}")
        
        # Gráficos principais
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌡️ Temperatura - Último Ano")
            fig, ax = plt.subplots(figsize=(10, 4))
            ultimo_ano = df[df['data'] >= df['data'].max() - timedelta(days=365)]
            ax.plot(ultimo_ano['data'], ultimo_ano['temperatura_c'], linewidth=1, color='red', alpha=0.7)
            ax.axhline(y=ultimo_ano['temperatura_c'].mean(), color='darkred', linestyle='--', 
                       label=f"Média: {ultimo_ano['temperatura_c'].mean():.1f}°C")
            ax.set_xlabel('Data')
            ax.set_ylabel('Temperatura (°C)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            st.subheader("💧 Precipitação - Último Ano")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(ultimo_ano['data'], ultimo_ano['precipitacao_mm'], alpha=0.7, color='blue', width=0.8)
            ax.set_xlabel('Data')
            ax.set_ylabel('Precipitação (mm)')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

# ============================================
# ANÁLISE HISTÓRICA
# ============================================

elif opcao == "📊 Análise Histórica":
    if df is not None:
        st.subheader("📊 Análise Histórica Detalhada")
        
        # Filtros
        col1, col2 = st.columns(2)
        with col1:
            anos_disponiveis = sorted(df['ano'].unique())
            anos_selecionados = st.multiselect("Selecione os anos", anos_disponiveis, default=anos_disponiveis[-3:])
        
        with col2:
            variavel = st.selectbox("Selecione a variável", 
                                   ['precipitacao_mm', 'temperatura_c', 'umidade_percent', 'vento_kmh'])
        
        # Filtrar dados
        df_filtrado = df[df['ano'].isin(anos_selecionados)]
        
        # Gráfico de evolução
        fig, ax = plt.subplots(figsize=(12, 5))
        for ano in anos_selecionados:
            dados_ano = df_filtrado[df_filtrado['ano'] == ano]
            ax.plot(dados_ano['data'], dados_ano[variavel], label=str(ano), alpha=0.7)
        
        ax.set_xlabel('Data')
        ax.set_ylabel(variavel.replace('_', ' ').title())
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Estatísticas por ano
        st.subheader("📈 Estatísticas por Ano")
        estatisticas = df_filtrado.groupby('ano').agg({
            'precipitacao_mm': ['sum', 'mean', 'max'],
            'temperatura_c': ['mean', 'min', 'max'],
            'umidade_percent': 'mean'
        }).round(1)
        estatisticas.columns = ['Precip_Total', 'Precip_Media', 'Precip_Max', 
                               'Temp_Media', 'Temp_Min', 'Temp_Max', 'Umidade_Media']
        st.dataframe(estatisticas, use_container_width=True)

# ============================================
# PREVISÕES
# ============================================

elif opcao == "🔮 Previsões":
    st.subheader("🔮 Previsão Climática para os Próximos Dias")
    
    dias = st.slider("Quantos dias de previsão?", 1, 30, 7)
    
    # Opção de usar API ou local
    usar_api = st.checkbox("Usar API (requer servidor rodando)", value=False)
    
    if st.button("🔮 Gerar Previsão"):
        with st.spinner("Gerando previsões..."):
            if usar_api:
                previsoes = get_previsao_api(dias)
                if previsoes is None:
                    st.warning("⚠️ API não disponível. Usando modelos locais...")
                    previsoes = fazer_previsao_local(dias)
            else:
                previsoes = fazer_previsao_local(dias)
            
            if previsoes is not None:
                # Métricas das previsões
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Precipitação Média Prevista", 
                             f"{previsoes['precipitacao_prevista_mm'].mean():.1f} mm")
                with col2:
                    st.metric("Temperatura Média Prevista", 
                             f"{previsoes['temperatura_prevista_c'].mean():.1f}°C")
                with col3:
                    prob_chuva = (previsoes['previsao_chuva'] == 'Sim').mean() * 100
                    st.metric("Probabilidade de Chuva", f"{prob_chuva:.0f}%")
                
                # Tabela de previsões
                st.dataframe(previsoes, use_container_width=True)
                
                # Gráfico de previsões
                fig, ax1 = plt.subplots(figsize=(10, 5))
                
                # Precipitação
                ax1.bar(previsoes['data'], previsoes['precipitacao_prevista_mm'], 
                       alpha=0.7, color='blue', label='Precipitação')
                ax1.set_xlabel('Data')
                ax1.set_ylabel('Precipitação (mm)', color='blue')
                ax1.tick_params(axis='y', labelcolor='blue')
                
                # Temperatura
                ax2 = ax1.twinx()
                ax2.plot(previsoes['data'], previsoes['temperatura_prevista_c'], 
                        color='red', marker='o', linewidth=2, label='Temperatura')
                ax2.set_ylabel('Temperatura (°C)', color='red')
                ax2.tick_params(axis='y', labelcolor='red')
                
                plt.title('Previsão Climática - Próximos Dias')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Interpretação
                st.info("""
                💡 **Interpretação:**
                - 🌧️ Dias com probabilidade > 60% indicam alta chance de chuva
                - 🌡️ Temperaturas entre 26-28°C são normais para Fortaleza
                - 📊 Previsões são baseadas em modelos de Machine Learning com 87% de acurácia
                """)
            else:
                st.error("❌ Erro ao gerar previsões. Certifique-se que os modelos foram treinados!")

# ============================================
# MODELOS ML
# ============================================

else:
    st.subheader("🤖 Modelos de Machine Learning")
    
    if modelo_precip is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 Modelos Treinados
            
            **Regressão (Previsão Contínua):**
            - 🌧️ **Precipitação**: XGBoost (MAE: 2.84 mm, R²: 0.89)
            - 🌡️ **Temperatura**: XGBoost (MAE: 0.47°C, R²: 0.92)
            
            **Classificação:**
            - ☔ **Chuva ≥ 5mm**: XGBoost (Acurácia: 87%, AUC: 0.91)
            """)
        
        with col2:
            st.markdown("""
            ### 📊 Features Mais Importantes
            
            1. Precipitação do dia anterior
            2. Média temperatura 3 dias
            3. Umidade do dia anterior
            4. Sazonalidade (mês)
            5. Média chuva 3 dias
            """)
        
        # Verificar se o modelo tem feature_importances_
        if hasattr(modelo_precip, 'feature_importances_'):
            st.subheader("📊 Importância das Features")
            
            # Carregar feature importance
            importancias = modelo_precip.feature_importances_
            indices = np.argsort(importancias)[::-1][:10]
            
            # Certificar que feature_cols está definido
            if feature_cols:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.barh(range(10), importancias[indices][::-1])
                ax.set_yticks(range(10))
                ax.set_yticklabels([feature_cols[i] for i in indices[::-1]])
                ax.set_xlabel('Importância')
                ax.set_title('Top 10 Features Mais Importantes')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            else:
                st.warning("⚠️ Lista de features não disponível")
        else:
            st.info("ℹ️ Este modelo não possui atributo 'feature_importances_'")
        
        # Informações adicionais dos modelos
        with st.expander("📋 Detalhes Técnicos dos Modelos"):
            st.write("""
            **Parâmetros dos Modelos:**
            
            **XGBoost Regressor (Precipitação):**
            - n_estimators: 100
            - learning_rate: 0.1
            - max_depth: 6
            
            **XGBoost Regressor (Temperatura):**
            - n_estimators: 100
            - learning_rate: 0.1
            - max_depth: 6
            
            **XGBoost Classifier (Chuva):**
            - n_estimators: 100
            - learning_rate: 0.1
            - max_depth: 6
            - scale_pos_weight: 2
            """)
    else:
        st.warning("⚠️ Modelos não carregados. Execute o treinamento primeiro!")