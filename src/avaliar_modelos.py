import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
import joblib

# Carregar dados e modelos
df = pd.read_csv('data/processed/clima_features.csv', parse_dates=['data'])
modelo_precip = joblib.load('models/modelo_precipitacao.pkl')
modelo_temp = joblib.load('models/modelo_temperatura.pkl')
modelo_class = joblib.load('models/modelo_classificacao.pkl')
scaler = joblib.load('models/scaler.pkl')

# Preparar dados
exclude_cols = ['data', 'precipitacao_mm', 'temperatura_c', 'chuva_forte', 'estacao']
feature_cols = [col for col in df.columns if col not in exclude_cols]
X = df[feature_cols].values
y_precip = df['precipitacao_mm'].values
y_temp = df['temperatura_c'].values
y_class = df['chuva_forte'].values

# Divisão temporal
train_size = int(0.8 * len(X))
X_test = X[train_size:]
y_test_precip = y_precip[train_size:]
y_test_temp = y_temp[train_size:]
y_test_class = y_class[train_size:]
datas_test = df['data'].values[train_size:]

X_test_scaled = scaler.transform(X_test)

# Predições
pred_precip = modelo_precip.predict(X_test_scaled)
pred_temp = modelo_temp.predict(X_test_scaled)
pred_class = modelo_class.predict(X_test_scaled)
pred_proba = modelo_class.predict_proba(X_test_scaled)[:, 1]

# 1. GRÁFICO DE PREVISÃO VS REAL (PRECIPITAÇÃO)
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Série temporal
axes[0, 0].plot(datas_test[-365:], y_test_precip[-365:], label='Real', alpha=0.7, linewidth=1)
axes[0, 0].plot(datas_test[-365:], pred_precip[-365:], label='Previsto', alpha=0.7, linewidth=1)
axes[0, 0].set_title('Previsão de Precipitação (último ano)', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Data')
axes[0, 0].set_ylabel('Precipitação (mm)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Scatter plot
axes[0, 1].scatter(y_test_precip, pred_precip, alpha=0.5, s=10)
axes[0, 1].plot([0, max(y_test_precip)], [0, max(y_test_precip)], 'r--', label='Ideal')
axes[0, 1].set_title('Valores Reais vs Previstos', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Precipitação Real (mm)')
axes[0, 1].set_ylabel('Precipitação Prevista (mm)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 2. GRÁFICO DE PREVISÃO VS REAL (TEMPERATURA)
axes[1, 0].plot(datas_test[-365:], y_test_temp[-365:], label='Real', alpha=0.7, linewidth=1)
axes[1, 0].plot(datas_test[-365:], pred_temp[-365:], label='Previsto', alpha=0.7, linewidth=1)
axes[1, 0].set_title('Previsão de Temperatura (último ano)', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Data')
axes[1, 0].set_ylabel('Temperatura (°C)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Distribuição dos erros
erros_temp = y_test_temp - pred_temp
axes[1, 1].hist(erros_temp, bins=50, alpha=0.7, color='orange', edgecolor='black')
axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[1, 1].set_title(f'Distribuição dos Erros (Média: {erros_temp.mean():.2f}°C)', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Erro (°C)')
axes[1, 1].set_ylabel('Frequência')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('reports/figuras/avaliacao_regressao.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. MATRIZ DE CONFUSÃO E CURVA ROC PARA CLASSIFICAÇÃO
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Matriz de confusão
cm = confusion_matrix(y_test_class, pred_class)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Sem Chuva', 'Chuva'])
disp.plot(ax=axes[0], cmap='Blues', values_format='d')
axes[0].set_title('Matriz de Confusão', fontsize=12, fontweight='bold')

# Curva ROC
fpr, tpr, thresholds = roc_curve(y_test_class, pred_proba)
auc = roc_auc_score(y_test_class, pred_proba)
axes[1].plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.3f}')
axes[1].plot([0, 1], [0, 1], 'r--', label='Aleatório')
axes[1].set_title('Curva ROC', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Taxa de Falsos Positivos')
axes[1].set_ylabel('Taxa de Verdadeiros Positivos')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('reports/figuras/avaliacao_classificacao.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. IMPORTÂNCIA DAS FEATURES
if hasattr(modelo_precip, 'feature_importances_'):
    importancias = modelo_precip.feature_importances_
    
    # Top 15 features
    indices = np.argsort(importancias)[::-1][:15]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(15), importancias[indices][::-1])
    plt.yticks(range(15), [feature_cols[i] for i in indices[::-1]])
    plt.xlabel('Importância')
    plt.title('Top 15 Features Mais Importantes (Precipitação)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('reports/figuras/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

print("\n✅ Avaliação dos modelos concluída!")