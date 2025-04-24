"""
Exemplo de ajuste de parâmetros para reator em batelada
"""

import numpy as np
from kinfit import (
    create_model, 
    load_experimental_data, 
    fit_kinetic_parameters, 
    plot_results,
    set_optimizer  # Agora importado corretamente da interface
)

# 1. Definir o sistema de EDOs
def batch_ode(y, t, params):
    A, B = y
    r = params['k'] * A**params['n']
    return [-r, r]

# 2. Criar o modelo
parameters = {
    'k': {'guess': 0.1, 'bounds': (0, 10)},
    'n': {'guess': 1.0, 'bounds': (0.5, 2), 'fixed': False}
}

initial_conditions = {'A': 1.0, 'B': 0.0}

create_model(
    ode_system=batch_ode,
    parameters=parameters,
    initial_conditions=initial_conditions
)

# 3. Carregar dados experimentais (simulados aqui para exemplo)
time_points = np.linspace(0, 10, 20)
A_exp = np.exp(-0.3 * time_points)  # Dados com k=0.3
B_exp = 1 - A_exp

load_experimental_data(
    time_points=time_points,
    data={'A': A_exp, 'B': B_exp}
)

# 4. Selecionar otimizador e ajustar parâmetros
set_optimizer('leastsq')  # Ou 'annealing'
result = fit_kinetic_parameters()

# 5. Visualizar resultados
print("\nFitted parameters:")
for name, value in result.parameters.items():
    print(f"{name}: {value:.4f}")

plot_results()