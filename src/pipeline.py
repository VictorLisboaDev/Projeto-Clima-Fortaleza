import subprocess
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def executar_pipeline():
    """Executa todo o pipeline de dados e modelos"""
    
    logging.info("🚀 Iniciando pipeline completo...")
    
    passos = [
        ("1. Gerando dados", "python src/gerar_dados_climaticos.py"),
        ("2. Feature Engineering", "python src/feature_engineering.py"),
        ("3. Treinando modelos", "python src/treinar_modelos.py"),
        ("4. Avaliando modelos", "python src/avaliar_modelos.py"),
        ("5. Gerando previsões", "python src/fazer_previsoes.py"),
        ("6. Atualizando dashboard", "python src/atualizar_dashboard.py")
    ]
    
    for nome, comando in passos:
        logging.info(f"🔧 {nome}...")
        try:
            resultado = subprocess.run(comando, shell=True, capture_output=True, text=True)
            if resultado.returncode == 0:
                logging.info(f"✅ {nome} concluído com sucesso!")
            else:
                logging.error(f"❌ Erro em {nome}: {resultado.stderr}")
                return False
        except Exception as e:
            logging.error(f"❌ Exceção em {nome}: {str(e)}")
            return False
    
    logging.info("🎉 Pipeline concluído com sucesso!")
    return True

if __name__ == "__main__":
    executar_pipeline()