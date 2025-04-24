"""
Funções utilitárias para KinFit
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd

def read_excel_data(file_path: str, sheet_name: str = 'Sheet1') -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Lê dados experimentais de um arquivo Excel
    
    Args:
        file_path: Caminho para o arquivo Excel
        sheet_name: Nome da planilha (opcional)
        
    Returns:
        Tupla com (time_points, data_dict)
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    time_points = df.iloc[:, 0].values
    data_dict = {col: df[col].values for col in df.columns[1:]}
    return time_points, data_dict

def create_batch_ode(A0: float = 1.0, products: List[str] = ['B']) -> Callable:
    """
    Cria uma função ODE padrão para reação em batelada A -> produtos
    
    Args:
        A0: Concentração inicial de A (padrão: 1.0)
        products: Lista de nomes dos produtos (padrão: ['B'])
        
    Returns:
        Função ODE para o sistema
    """
    def ode_func(y, t, params):
        derivatives = []
        A = y[0]
        r = params['k'] * A**params.get('n', 1.0)
        
        # Derivada para A
        derivatives.append(-r)
        
        # Derivadas para produtos (assume estequiometria 1:1)
        for _ in products:
            derivatives.append(r)
        
        return derivatives
    
    return ode_func