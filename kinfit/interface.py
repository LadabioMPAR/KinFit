"""
Interface simplificada para o usuário
"""

from typing import Dict, List, Callable, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

from pyparsing import col
from .model import KineticModel
from .optimizer import Optimizer, available_optimizers
import pandas as pd
from tkinter import Tk, filedialog
import os
import re

# Variável global para o modelo atual
_current_model = None
_current_optimizer = "leastsq"

@dataclass
class FitResult:
    parameters: Dict[str, float]
    success: bool
    message: str
    objective_value: float
    predictions: Dict[str, np.ndarray]
    covariance_matrix: Optional[np.ndarray]
    confidence_intervals: Optional[Dict[str, Dict[str, float]]]

def create_model(
    ode_system: Callable,
    parameters: Dict[str, Dict[str, float]],
    initial_conditions: Dict[str, float],
    feeding_profiles: Optional[Dict[str, Callable]] = None
) -> None:
    """
    Cria e configura um modelo cinético
    
    Args:
        ode_system: Função que define o sistema de EDOs
        parameters: Dicionário de parâmetros com seus limites
                   Formato: {'param_name': {'guess': val, 'bounds': (min, max), 'fixed': False}}
        initial_conditions: Condições iniciais para as variáveis de estado
        feeding_profiles: Perfis de alimentação (opcional)
    """
    global _current_model
    
    model = KineticModel()
    model.set_ode_system(ode_system)
    
    for name, config in parameters.items():
        if config.get('fixed', False):
            model.fix_parameter(name, config['guess'])
        else:
            model.add_parameter(name, config['guess'], config.get('bounds', (0, np.inf)))
    
    model.set_initial_conditions(initial_conditions)
    
    if feeding_profiles:
        for component, profile in feeding_profiles.items():
            model.set_feeding_profile(component, profile)
    
    _current_model = model

def load_experimental_data(
    time_points: np.ndarray,
    data: Dict[str, np.ndarray]
) -> None:
    """
    Carrega dados experimentais para ajuste
    
    Args:
        time_points: Vetor de tempos experimentais
        data: Dicionário com dados experimentais para cada variável
    """
    if _current_model is None:
        raise ValueError("No model created. Call create_model() first.")
    
    _current_model.set_experimental_data(time_points, data)

def set_optimizer(optimizer_name: str, **kwargs) -> None:
    """
    Seleciona o otimizador a ser usado
    
    Args:
        optimizer_name: Nome do otimizador ('leastsq' ou 'annealing')
        **kwargs: Argumentos adicionais específicos do otimizador
    """
    global _current_optimizer
    _current_optimizer = optimizer_name.lower()
    
    # Configura parâmetros específicos do otimizador
    if _current_optimizer in available_optimizers:
        available_optimizers[_current_optimizer].configure(**kwargs)
    else:
        raise ValueError(f"Optimizer '{optimizer_name}' not available. Choose from: {list(available_optimizers.keys())}")

def fit_kinetic_parameters() -> FitResult:
    """
    Ajusta os parâmetros cinéticos usando o otimizador selecionado
    
    Returns:
        FitResult com os resultados do ajuste
    """
    if _current_model is None:
        raise ValueError("No model created. Call create_model() first.")
    
    if _current_model._experimental_data is None:
        raise ValueError("No experimental data loaded. Call load_experimental_data() first.")
    
    # Obtém o otimizador selecionado
    optimizer = available_optimizers[_current_optimizer]
    
    # Executa a otimização
    result = optimizer.optimize(_current_model)
    
    # Prepara resultados
    predictions = _current_model.solve_ode(result.parameters)
    
    return FitResult(
        parameters=result.parameters,
        success=result.success,
        message=result.message,
        objective_value=result.objective_value,
        predictions=predictions,
        covariance_matrix=result.covariance_matrix,
        confidence_intervals=result.confidence_intervals
    )

def plot_results(
    show_experimental: bool = True,
    variables: Optional[List[str]] = None
) -> None:
    """
    Plota os resultados do ajuste
    
    Args:
        show_experimental: Se True, mostra dados experimentais
        variables: Lista de variáveis para plotar (None para todas)
    """
    if _current_model is None:
        raise ValueError("No model created. Call create_model() first.")
    
    if _current_model._experimental_data is None:
        raise ValueError("No experimental data loaded. Call load_experimental_data() first.")
    
    
def processar_arquivo_texto():
    """
    Processa arquivos de texto (.txt) com diferentes formatos.
    Abre uma janela para seleção do arquivo e detecta automaticamente o formato.
    
    Returns:
        DataFrame: Dados processados ou None em caso de erro
    """
    try:
        # Configura interface gráfica para seleção de arquivo
        root = Tk()
        root.withdraw()
        
        # Abre diálogo para seleção do arquivo
        caminho_arquivo = filedialog.askopenfilename(
            title="Selecione o arquivo de texto",
            filetypes=[("Arquivos de texto", ".txt"), ("Todos os arquivos", ".*")]
        )
        
        if not caminho_arquivo:
            print("Nenhum arquivo selecionado.")
            return None
        
        # Verifica se o arquivo existe
        if not os.path.exists(caminho_arquivo):
            print(f"Erro: O arquivo '{caminho_arquivo}' não foi encontrado.")
            return None
            
        # Lê o conteúdo do arquivo
        with open(caminho_arquivo, 'r', encoding='utf-8') as file:
            conteudo = file.readlines()
        
        # Analisa o formato do arquivo
        formato = detectar_formato(conteudo)
        print(f"Formato detectado: {formato}")
        
        # Processa de acordo com o formato detectado
        if formato == "csv":
            print("csv")
            df = pd.read_csv(caminho_arquivo)
        elif formato == "tsv":
            
            df = pd.read_csv(caminho_arquivo, sep='\t',header=[0,1])
            
        elif formato == "fixed_width":
            print("fixed_width")
            df = processar_largura_fixa(conteudo)
        else:
            print("unknown")
            # Padrão: tentar ler como CSV com delimitador automático
            df = pd.read_csv(caminho_arquivo, sep=None, engine='python',header=[0,1])

        tempo = df.iloc[:,0].to_numpy()

        dados = {col: df[col].to_numpy() for col in df.columns[1:]}

        load_experimental_data(tempo, dados)


        
    except Exception as e:
        print(f"Erro ao processar arquivo: {str(e)}")

def detectar_formato(conteudo):
    """
    Detecta automaticamente o formato do arquivo de texto.
    
    Args:
        conteudo (list): Lista de linhas do arquivo
        
    Returns:
        str: Tipo de formato detectado ('csv', 'tsv', 'fixed_width', 'unknown')
    """
    if not conteudo:
        return "unknown"
    
    # Verifica se é CSV (vírgulas como separadores)
    primeira_linha = conteudo[0].strip()
    if primeira_linha.count(',') > 3:  # Pelo menos 4 campos
        return "csv"
    
    # Verifica se é TSV (tabulações como separadores)
    if primeira_linha.count('\t') > 3:  # Pelo menos 4 campos
        return "tsv"
    
    # Verifica se é de largura fixa (campos alinhados verticalmente)
    if len(conteudo) > 1:
        segunda_linha = conteudo[1].strip()
        # Procura por padrões de alinhamento
        if (len(primeira_linha) > 30 and len(segunda_linha) > 30 and
            any(c.isdigit() for c in primeira_linha) and
            any(c.isdigit() for c in segunda_linha)):
            return "fixed_width"
    
    return "unknown"

def processar_largura_fixa(conteudo):
    """
    Processa arquivos de texto de largura fixa.
    
    Args:
        conteudo (list): Lista de linhas do arquivo
        
    Returns:
        DataFrame: Dados processados
    """
    # Detecta posições das colunas procurando por mudanças no tipo de caractere
    posicoes_colunas = []
    
    # Usa a primeira linha para detectar possíveis cabeçalhos
    primeira_linha = conteudo[0].rstrip()
    
    # Encontra transições entre espaços e não-espaços
    em_campo = False
    inicio_campo = 0
    
    for i, char in enumerate(primeira_linha):
        if char != ' ' and not em_campo:
            inicio_campo = i
            em_campo = True
        elif char == ' ' and em_campo:
            posicoes_colunas.append((inicio_campo, i))
            em_campo = False
    
    # Se ainda está em um campo no final da linha
    if em_campo:
        posicoes_colunas.append((inicio_campo, len(primeira_linha)))
    
    # Se não detectou colunas, usa heurística simples
    if not posicoes_colunas:
        largura_estimada = max(len(linha) for linha in conteudo) // 5
        posicoes_colunas = [(i, i + largura_estimada) for i in range(0, len(primeira_linha), largura_estimada)]
    
    # Extrai dados
    dados = []
    for linha in conteudo:
        linha = linha.rstrip()
        linha_dados = []
        for inicio, fim in posicoes_colunas:
            if inicio < len(linha):
                campo = linha[inicio:min(fim, len(linha))].strip()
                linha_dados.append(campo)
            else:
                linha_dados.append("")
        dados.append(linha_dados)
    
    # Converte para DataFrame
    df = pd.DataFrame(dados[1:], columns=dados[1] if len(dados) > 1 else None)
    
    # Tenta converter colunas numéricas
    for coluna in df.columns:
        df[coluna] = pd.to_numeric(df[coluna], errors='ignore')
    
    return df