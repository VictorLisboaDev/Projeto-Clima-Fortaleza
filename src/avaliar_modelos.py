import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# Carregar dados com features
print("="*60)
print("CARREGANDO DADOS PARA MODELAGEM")
print("="*60)

df = pd.read_csv('data/processed/clima_features.csv', parse_dates=['data'])
print(f"Dados carregados: {df.shape[0]} registros, {df.shape[1]} features")

# Definir features (excluir colunas não numéricas e targets)
exclude_cols = ['data', 'precipitacao_mm', 'temperatura_c', 'chuva_forte', 'estacao']
feature_cols = [col for col in df.columns if col not in exclude_cols]

print(f"\nFeatures utilizadas: {len(feature_cols)}")
print(f"Features: {feature_cols[:10]}...")

# ============================================
# MODELO 1: Regressão - Previsão de Precipitação
# ============================================
print("\n" + "="*60)
print("MODELO 1: PREVISÃO DE PRECIPITAÇÃO (REGRESSÃO)")
print("="*60)

X = df[feature_cols].values
y_precip = df['precipitacao_mm'].values

# Divisão temporal (respeitando ordem cronológica)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y_precip[:train_size], y_precip[train_size:]

# Escalonamento
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelos para regressão
modelos_regressao = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

resultados_regressao = {}

for nome, modelo in modelos_regressao.items():
    print(f"\nTreinando {nome}...")
    modelo.fit(X_train_scaled, y_train)
    
    # Predições
    y_pred = modelo.predict(X_test_scaled)
    
    # Métricas
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    resultados_regressao[nome] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'modelo': modelo}
    
    print(f"  MAE: {mae:.2f} mm")
    print(f"  RMSE: {rmse:.2f} mm")
    print(f"  R²: {r2:.4f}")

# Melhor modelo para precipitação
melhor_regressor = max(resultados_regressao, key=lambda x: resultados_regressao[x]['R2'])
print(f"\n🏆 Melhor modelo para precipitação: {melhor_regressor}")

# ============================================
# MODELO 2: Regressão - Previsão de Temperatura
# ============================================
print("\n" + "="*60)
print("MODELO 2: PREVISÃO DE TEMPERATURA (REGRESSÃO)")
print("="*60)

y_temp = df['temperatura_c'].values
y_train_temp, y_test_temp = y_temp[:train_size], y_temp[train_size:]

resultados_temp = {}

for nome, modelo in modelos_regressao.items():
    print(f"\nTreinando {nome}...")
    modelo.fit(X_train_scaled, y_train_temp)
    
    y_pred = modelo.predict(X_test_scaled)
    
    mae = mean_absolute_error(y_test_temp, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_temp, y_pred))
    r2 = r2_score(y_test_temp, y_pred)
    
    resultados_temp[nome] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'modelo': modelo}
    
    print(f"  MAE: {mae:.2f}°C")
    print(f"  RMSE: {rmse:.2f}°C")
    print(f"  R²: {r2:.4f}")

melhor_temp = max(resultados_temp, key=lambda x: resultados_temp[x]['R2'])
print(f"\n🏆 Melhor modelo para temperatura: {melhor_temp}")

# ============================================
# MODELO 3: Classificação - Chuva ou Não Chuva
# ============================================
print("\n" + "="*60)
print("MODELO 3: CLASSIFICAÇÃO (CHUVA FORTE ≥5mm)")
print("="*60)

y_class = df['chuva_forte'].values
y_train_class, y_test_class = y_class[:train_size], y_class[train_size:]

# Balanceamento de classes (opcional)
from sklearn.utils import class_weight
classes_weights = class_weight.compute_class_weight('balanced', 
                                                     classes=np.unique(y_train_class),
                                                     y=y_train_class)
class_weight_dict = dict(zip(np.unique(y_train_class), classes_weights))

modelos_classificacao = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, scale_pos_weight=2),
    'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced')
}

resultados_class = {}

for nome, modelo in modelos_classificacao.items():
    print(f"\nTreinando {nome}...")
    modelo.fit(X_train_scaled, y_train_class)
    
    y_pred = modelo.predict(X_test_scaled)
    y_pred_proba = modelo.predict_proba(X_test_scaled)[:, 1]
    
    acuracia = accuracy_score(y_test_class, y_pred)
    precisao = precision_score(y_test_class, y_pred)
    recall = recall_score(y_test_class, y_pred)
    f1 = f1_score(y_test_class, y_pred)
    auc = roc_auc_score(y_test_class, y_pred_proba)
    
    resultados_class[nome] = {
        'Acurácia': acuracia, 
        'Precisão': precisao, 
        'Recall': recall, 
        'F1': f1, 
        'AUC': auc,
        'modelo': modelo
    }
    
    print(f"  Acurácia: {acuracia:.4f}")
    print(f"  Precisão: {precisao:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  AUC-ROC: {auc:.4f}")

melhor_class = max(resultados_class, key=lambda x: resultados_class[x]['F1'])
print(f"\n🏆 Melhor modelo para classificação: {melhor_class}")

# ============================================
# SALVAR MODELOS TREINADOS
# ============================================
print("\n" + "="*60)
print("SALVANDO MODELOS")
print("="*60)

# Salvar melhores modelos
joblib.dump(resultados_regressao[melhor_regressor]['modelo'], 'models/modelo_precipitacao.pkl')
joblib.dump(resultados_temp[melhor_temp]['modelo'], 'models/modelo_temperatura.pkl')
joblib.dump(resultados_class[melhor_class]['modelo'], 'models/modelo_classificacao.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("✅ Modelos salvos em /models/")
print(f"  - modelo_precipitacao.pkl ({melhor_regressor})")
print(f"  - modelo_temperatura.pkl ({melhor_temp})")
print(f"  - modelo_classificacao.pkl ({melhor_class})")
print(f"  - scaler.pkl")

# Salvar resultados em CSV
resultados_df = pd.DataFrame({
    'Modelo_Precipitacao': list(resultados_regressao.keys()),
    'MAE': [resultados_regressao[m]['MAE'] for m in resultados_regressao],
    'RMSE': [resultados_regressao[m]['RMSE'] for m in resultados_regressao],
    'R2': [resultados_regressao[m]['R2'] for m in resultados_regressao]
})
resultados_df.to_csv('reports/resultados_modelos_precipitacao.csv', index=False)

resultados_class_df = pd.DataFrame({
    'Modelo_Classificacao': list(resultados_class.keys()),
    'Acurácia': [resultados_class[m]['Acurácia'] for m in resultados_class],
    'Precisão': [resultados_class[m]['Precisão'] for m in resultados_class],
    'Recall': [resultados_class[m]['Recall'] for m in resultados_class],
    'F1': [resultados_class[m]['F1'] for m in resultados_class],
    'AUC': [resultados_class[m]['AUC'] for m in resultados_class]
})
resultados_class_df.to_csv('reports/resultados_modelos_classificacao.csv', index=False)

print("\n✅ Resultados salvos em /reports/")