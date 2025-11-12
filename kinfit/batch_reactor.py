"""
Exemplo de ajuste de parâmetros para reator em batelada
"""

import numpy as np
from kinfit import (
    create_model, 
    load_experimental_data, 
    fit_kinetic_parameters, 
    plot_results,
    set_optimizer,
    optimizer # Agora importado corretamente da interface
)
from kinfit.optimizer import LeastSquaresOptimizer
from kinfit import plots
from model import KineticModel
from plots import plot_kinetic_results

'''# 1. Definir o sistema de EDOs
def batch_ode(y, t, params):
    A, B = y
    r = params['k'] * A**params['n']
    return [-r, r]

# 2. Criar o modelo usando model.py diretamente
model = KineticModel()
model.set_ode_system(batch_ode)
model.add_parameter('k', initial_guess=0.1, bounds=(0, 10))
model.add_parameter('n', initial_guess=1.0, bounds=(0.5, 2))
model.set_initial_conditions({'A': 1.0, 'B': 0.0})

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
optimizer = LeastSquaresOptimizer()
result = optimizer.optimize(model)

# 5. Visualizar resultados
print("\nFitted parameters:")
for name, value in result.parameters.items():
    print(f"{name}: {value:.4f}")

plot_results()
print(result.parameters.values())
result = fit_kinetic_parameters()

print("\nParâmetros ajustados:")
for name, value in result.parameters.items():
    ci = result.confidence_intervals.get(name, None)
    if ci:
        print(f"{name}: {value:.4f} (IC95%: {ci['ci_lower']} – {ci['ci_upper']})")
    else:
        print(f"{name}: {value:.4f} (sem IC calculado)")

print(time_points)

import matplotlib.pyplot as plt
import numpy as np
# 6. Plot usando plot_kinetic_results
plot_kinetic_results(
    model=model,  # Usa o modelo retornado por create_model
    result=result,
    title="Reator em Batelada: A → B",
    xlabel="Tempo (min)",
    ylabel="Concentração (mol/L)",
    colors={'A': 'blue', 'B': 'red'},
    grid=True
)'''

import numpy as np
from kinfit import (
    load_experimental_data,
    fit_kinetic_parameters,
    plot_results,
    set_optimizer,
    optimizer
)
from kinfit.optimizer import LeastSquaresOptimizer
from model import KineticModel  # ← Importa KineticModel do model.py
from plots import plot_kinetic_results

# 1. Definir o sistema de EDOs
def batch_ode(y, t, params):
    A, B = y
    r = params['k'] * A**params['n']
    return [-r, r]

# 2. Criar o modelo usando model.py diretamente
model = KineticModel()
model.set_ode_system(batch_ode)
model.add_parameter('k', initial_guess=0.1, bounds=(0, 10))
model.add_parameter('n', initial_guess=1.0, bounds=(0.5, 2))
model.set_initial_conditions({'A': 1.0, 'B': 0.0})

# 3. Carregar dados experimentais (simulados aqui para exemplo)
np.random.seed(42)
time_points = np.linspace(0, 10, 20)
A_exp = np.exp(-0.3 * time_points) + np.random.normal(0, 0.1, 20)
B_exp = 1 - A_exp + np.random.normal(0, 0.1, 20)

'''time_points = np.linspace(0, 10, 20)
A_exp = np.exp(-0.3 * time_points)  # Dados com k=0.3
B_exp = 1 - A_exp'''

model.set_experimental_data(
    time_points=time_points,
    data={'A': A_exp, 'B': B_exp}
)

# 4. Selecionar otimizador e ajustar parâmetros
optimizer = LeastSquaresOptimizer()
result = optimizer.optimize(model)

# 5. Visualizar resultados
print("\nFitted parameters:")
for name, value in result.parameters.items():
    print(f"{name}: {value:.4f}")

print(result.parameters.values())

print(time_points)

# 6. Plot usando plot_kinetic_results
plot_kinetic_results(
    model=model,
    result=result,
    title="Reator em Batelada: A → B",
    xlabel="Tempo (min)",
    ylabel="Concentração (mol/L)",
    colors={'A': 'blue', 'B': 'red'},
    grid=False,
    separate_plots=True
)
