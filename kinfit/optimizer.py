"""
Implementação dos otimizadores
"""

from typing import Dict, Callable, Optional
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares, dual_annealing
from .model import KineticModel

@dataclass
class OptimizationResult:
    parameters: Dict[str, float]
    success: bool
    message: str
    objective_value: float

class Optimizer:
    def __init__(self, name: str):
        self.name = name
        self.config = {}
    
    def configure(self, **kwargs):
        self.config.update(kwargs)
    
    def optimize(self, model: KineticModel) -> OptimizationResult:
        raise NotImplementedError

class LeastSquaresOptimizer(Optimizer):
    def __init__(self):
        super().__init__('leastsq')
        self.configure(method='trf', max_nfev=1000)
    
    def optimize(self, model: KineticModel) -> OptimizationResult:
        # Prepara parâmetros para otimização
        param_names = []
        initial_guess = []
        bounds = []
        
        for name, param_data in model._parameters.items():
            if param_data['optimize']:
                param_names.append(name)
                initial_guess.append(param_data['value'])
                bounds.append(param_data['bounds'])
        
        if not initial_guess:
            return OptimizationResult(
                parameters={name: p['value'] for name, p in model._parameters.items()},
                success=False,
                message="No parameters to optimize",
                objective_value=np.inf
            )
        
        # Função para cálculo dos resíduos
        def residuals(params):
            param_dict = {}
            opt_idx = 0
            for name, param_data in model._parameters.items():
                if param_data['optimize']:
                    param_dict[name] = params[opt_idx]
                    opt_idx += 1
                else:
                    param_dict[name] = param_data['value']
            
            solution = model.solve_ode(param_dict)
            residuals = []
            
            for var_name, exp_data in model._experimental_data.items():
                if var_name in solution:
                    residuals.extend(exp_data - solution[var_name])
            
            return np.array(residuals)
        
        # Executa otimização
        result = least_squares(
            residuals,
            x0=initial_guess,
            bounds=list(zip(*bounds)) if bounds else None,
            **self.config
        )
        
        # Prepara resultados
        opt_idx = 0
        fitted_params = {}
        for name, param_data in model._parameters.items():
            if param_data['optimize']:
                fitted_params[name] = result.x[opt_idx]
                opt_idx += 1
            else:
                fitted_params[name] = param_data['value']
        
        return OptimizationResult(
            parameters=fitted_params,
            success=result.success,
            message=result.message,
            objective_value=result.cost
        )

class SimulatedAnnealingOptimizer(Optimizer):
    def __init__(self):
        super().__init__('annealing')
        self.configure(maxiter=1000, initial_temp=5230)
    
    def optimize(self, model: KineticModel) -> OptimizationResult:
        # Prepara parâmetros para otimização
        param_names = []
        initial_guess = []
        bounds = []
        
        for name, param_data in model._parameters.items():
            if param_data['optimize']:
                param_names.append(name)
                initial_guess.append(param_data['value'])
                bounds.append(param_data['bounds'])
        
        if not initial_guess:
            return OptimizationResult(
                parameters={name: p['value'] for name, p in model._parameters.items()},
                success=False,
                message="No parameters to optimize",
                objective_value=np.inf
            )
        
        # Função objetivo
        def objective(params):
            param_dict = {}
            opt_idx = 0
            for name, param_data in model._parameters.items():
                if param_data['optimize']:
                    param_dict[name] = params[opt_idx]
                    opt_idx += 1
                else:
                    param_dict[name] = param_data['value']
            
            return model.evaluate_objective(param_dict)
        
        # Executa otimização
        result = dual_annealing(
            objective,
            bounds=bounds,
            x0=initial_guess,
            **self.config
        )
        
        # Prepara resultados
        opt_idx = 0
        fitted_params = {}
        for name, param_data in model._parameters.items():
            if param_data['optimize']:
                fitted_params[name] = result.x[opt_idx]
                opt_idx += 1
            else:
                fitted_params[name] = param_data['value']
        
        return OptimizationResult(
            parameters=fitted_params,
            success=result.success,
            message=result.message,
            objective_value=result.fun
        )

# Otimizadores disponíveis
available_optimizers = {
    'leastsq': LeastSquaresOptimizer(),
    'annealing': SimulatedAnnealingOptimizer()
}

def add_custom_optimizer(name: str, optimizer: Optimizer) -> None:
    """
    Adiciona um otimizador personalizado à lista de disponíveis
    
    Args:
        name: Nome do otimizador
        optimizer: Instância do otimizador
    """
    available_optimizers[name.lower()] = optimizer

def list_available_optimizers() -> Dict[str, Optimizer]:
    """
    Retorna a lista de otimizadores disponíveis
    
    Returns:
        Dicionário com nomes e instâncias dos otimizadores
    """
    return available_optimizers