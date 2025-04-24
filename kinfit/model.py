"""
Definição das classes de modelo cinético
"""

from typing import Dict, List, Callable, Optional, Tuple
import numpy as np
from scipy.integrate import odeint
from dataclasses import dataclass

@dataclass
class ModelResult:
    parameters: Dict[str, float]
    success: bool
    message: str
    objective_value: float

class KineticModel:
    def __init__(self):
        self._ode_system = None
        self._parameters = {}
        self._initial_conditions = {}
        self._experimental_data = None
        self._time_points = None
        self._feeding_profiles = {}
    
    def set_ode_system(self, ode_func: Callable) -> None:
        self._ode_system = ode_func
    
    def add_parameter(self, name: str, initial_guess: float, bounds: Tuple[float, float] = (0, np.inf)) -> None:
        self._parameters[name] = {
            'value': initial_guess,
            'bounds': bounds,
            'optimize': True
        }
    
    def fix_parameter(self, name: str, value: float) -> None:
        self._parameters[name] = {
            'value': value,
            'bounds': (value, value),
            'optimize': False
        }
    
    def set_initial_conditions(self, conditions: Dict[str, float]) -> None:
        self._initial_conditions = conditions
    
    def set_experimental_data(self, time_points: np.ndarray, data: Dict[str, np.ndarray]) -> None:
        self._time_points = time_points
        self._experimental_data = data
    
    def set_feeding_profile(self, component: str, profile: Callable) -> None:
        self._feeding_profiles[component] = profile
    
    def solve_ode(self, params: Dict[str, float], time_points: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        if time_points is None:
            time_points = self._time_points
        
        y0 = list(self._initial_conditions.values())
        solution = odeint(self._ode_system, y0, time_points, args=(params,))
        
        return {
            var_name: solution[:, i]
            for i, var_name in enumerate(self._initial_conditions.keys())
        }
    
    def evaluate_objective(self, params: Dict[str, float]) -> float:
        solution = self.solve_ode(params)
        ssr = 0.0
        
        for var_name, exp_data in self._experimental_data.items():
            if var_name in solution:
                ssr += np.sum((exp_data - solution[var_name])**2)
        
        return ssr