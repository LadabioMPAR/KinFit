import kinfit
from kinfit import list_available_optimizers

# Verifique otimizadores disponíveis
print(list_available_optimizers())
# Saída esperada: {'leastsq': <LeastSquaresOptimizer>, 'annealing': <SimulatedAnnealingOptimizer>}

# Teste criação de modelo
def ode_system(y, t, params):
    return [-params['k'] * y[0]]

kinfit.create_model(
    ode_system=ode_system,
    parameters={'k': {'guess': 0.1, 'bounds': (0, 10)}},
    initial_conditions={'A': 1.0}
)

# Se não houver erros, a biblioteca está funcionando!