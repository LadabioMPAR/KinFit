"""
Interface simplificada para o usuário
"""

from typing import Dict, List, Callable, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from .model import KineticModel
from .optimizer import Optimizer, available_optimizers

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
        predictions=predictions
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
    
    # Implementação similar à anterior, adaptada para a nova estrutura
    # ...