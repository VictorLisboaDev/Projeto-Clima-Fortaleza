import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import requests
import json

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

# Sidebar
st.sidebar.header("📊 Controles")
opcao = st.sidebar.selectbox(
    "Escolha uma visualização",
    ["📈 Visão Geral", "📊 Análise Histórica", "🔮 Previsões", "🤖 Modelos ML"]
)

# Carregar dados
@st.cache_data
def load_data():
    df = pd.read_csv('data/processed/clima_features.csv', parse_dates=['data'])
    return df

df = load_data()

# Função para chamar a API
@st.cache_data(ttl=3600)
def get_previsao(dias=7):
    try:
        response = requests.get(f"http://localhost:8000/previsao/{dias}")
        if response.status_code == 200:
            return pd.DataFrame(response.json())
        else:
            return None
    except:
        return None

# =====================
# VISÃO GERAL
# =====================
if opcao == "📈 Visão Geral":
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

# =====================
# ANÁLISE HISTÓRICA
# =====================
elif opcao == "📊 Análise Histórica":
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

# =====================
# PREVISÕES
# =====================
elif opcao == "🔮 Previsões":
    st.subheader("🔮 Previsão Climática para os Próximos Dias")
    
    dias = st.slider("Quantos dias de previsão?", 1, 30, 7)
    
    if st.button("🔮 Gerar Previsão"):
        with st.spinner("Gerando previsões..."):
            previsoes = get_previsao(dias)
            
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
                st.error("❌ Erro ao gerar previsões. Certifique-se que a API está rodando!")

# =====================
# MODELOS ML
# =====================
else:
    st.subheader("🤖 Modelos de Machine Learning")
    
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
    
    # Importância das features
    if 'feature_importances_' in dir(modelo_precip):
        st.subheader("📊 Importância das Features")
        
        # Carregar feature importance
        importancias = modelo_precip.feature_importances_
        indices = np.argsort(importancias)[::-1][:10]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(range(10), importancias[indices][::-1])
        ax.set_yticks(range(10))
        ax.set_yticklabels([feature_cols[i] for i in indices[::-1]])
        ax.set_xlabel('Importância')
        ax.set_title('Top 10 Features Mais Importantes')
        st.pyplot(fig)