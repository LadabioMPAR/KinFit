# Testes unitários futuros
import numpy as np
from kinfit import create_model, load_experimental_data

def test_model_creation():
    """Testa se um modelo básico é criado corretamente."""
    def ode_system(y, t, params):
        return [-params['k'] * y[0]]
    
    create_model(
        ode_system=ode_system,
        parameters={'k': {'guess': 0.1, 'bounds': (0, 10)}},
        initial_conditions={'A': 1.0}
    )
    
    # Simula dados experimentais
    time = np.linspace(0, 10, 5)
    data = {'A': np.exp(-0.1 * time)}
    load_experimental_data(time, data)
    
    assert True  # Se chegou aqui sem erros, o teste passa