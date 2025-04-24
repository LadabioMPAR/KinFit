"""
KinFit - Kinetic parameter fitting for chemical engineers
"""

from .interface import (
    fit_kinetic_parameters,
    create_model,
    load_experimental_data,
    plot_results,
    set_optimizer,
)

from .optimizer import (
    add_custom_optimizer,
    list_available_optimizers
)

__all__ = [
    'fit_kinetic_parameters',
    'create_model',
    'load_experimental_data',
    'plot_results',
    'set_optimizer',
    'add_custom_optimizer',
    'list_available_optimizers'
]

__version__ = "0.1.0"