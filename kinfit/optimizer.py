"""
Implementação dos otimizadores
"""

from typing import Dict, Callable, Optional
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares, dual_annealing
from .model import KineticModel
from scipy.stats import t

@dataclass
class OptimizationResult:
    parameters: Dict[str, float]
    success: bool
    message: str
    objective_value: float
    covariance_matrix: Optional[np.ndarray]
    confidence_intervals: Optional[Dict[str, Dict[str, float]]]

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

        # Prepara o dicionário de parâmetros ajustados
        fitted_params = {name: p['value'] for name, p in model._parameters.items()}
        fitted_params.update(dict(zip(param_names, result.x)))

        # Inicializa todas as variáveis de resultado como None
        dof, cov_matrix, corr_matrix = None, None, None
        std_errors_dict, conf_intervals, conf_bands = None, None, None

        if result.success:
            # Equivalente a 'm' e 'n' no MATLAB
            num_data_points = len(result.fun)
            num_params = len(initial_guess)

            # Equivalente a 'gl' no MATLAB
            dof = num_data_points - num_params

            if dof > 0:
                try:
                    # 1. Obter a Jacobiana ('X' no MATLAB)
                    jacobian = result.jac

                    # 2. Obter a Soma dos Quadrados dos Resíduos ('Fobjetivo' no MATLAB)
                    ssr = 2 * result.cost

                    # 3. Calcular a variância dos resíduos
                    residual_variance = ssr / dof

                    # 4. Calcular a matriz de covariância ('covar' no MATLAB)
                    cov_matrix = np.linalg.pinv(jacobian.T @ jacobian) * residual_variance

                    # 5. Calcular o desvio padrão ('dp') e o IC ('ic')
                    std_errors = np.sqrt(np.diag(cov_matrix))
                    alpha = 1.0 - 0.95  # Para 95% de confiança
                    t_critical = t.ppf(1.0 - alpha / 2.0, df=dof)  # 't' no MATLAB

                    std_errors_dict = {}
                    conf_intervals = {}
                    for i, param_name in enumerate(param_names):
                        value = result.x[i]
                        se = std_errors[i]
                        std_errors_dict[param_name] = se
                        conf_intervals[param_name] = {
                            'ci_lower': value - t_critical * se,
                            'ci_upper': value + t_critical * se
                        }

                except np.linalg.LinAlgError:
                    print("Aviso: Não foi possível calcular as estatísticas (matriz singular).")

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
            objective_value=result.cost,
            covariance_matrix=cov_matrix,
            confidence_intervals=conf_intervals
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