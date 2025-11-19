.. image:: logo.png
   :alt: KinFit Logo
   :align: center
   :width: 300px

MPA Ribeiro's KinFit
====================

**KinFit** is a Python package designed for kinetic model fitting and kinetic parameter optimization. The main goal is to provide a framework for modeling various chemical reactions and reactor configurations using customizable systems of differential equations, robust optimization methods, and professional visualization.

Features
--------

- **Flexible Kinetic Modeling:** Define custom ODE systems for different reaction schemes and reactor types.
- **Multiple Optimizers:** Built-in support for least squares and other optimization algorithms.
- **Statistical Analysis:** Calculation of confidence intervals and statistical evaluation of parameter estimates.
- **Professional Visualization:** Ready-made plotting functions for model results with confidence bands.
- **Extensible Architecture:** Integrate new optimizers, models, and experimental setups with minimal effort.

Installation
------------

To install KinFit, clone the repository and install the dependencies:

.. code-block:: bash

    git clone <REPOSITORY_URL>
    cd KinFit
    pip install -r requirements.txt

Required dependencies:

- numpy
- scipy
- matplotlib

Quickstart
----------

This guide demonstrates the basic workflow for fitting kinetic parameters using KinFit. Follow these steps to set up your model, load data, optimize parameters, and visualize results.

**Step 1: Import Required Modules**

.. code-block:: python

    import numpy as np
    from kinfit.model import KineticModel
    from kinfit.optimizer import LeastSquaresOptimizer
    from kinfit.plots import plot_kinetic_results

**Step 2: Define Your ODE System**

.. code-block:: python

    def your_ode_system(y, t, params):
        """
        Define your system of differential equations.

        Args:
            y: Current state vector (concentrations, etc.)
            t: Current time point
            params: Dictionary of kinetic parameters

        Returns:
            List of derivatives dy/dt
        """
        # Example structure (customize for your reaction):
        component1, component2 = y
        rate = params['rate_constant'] * component1**params['order']
        # Replace derivative1 and derivative2 as appropriate:
        derivative1 = -rate
        derivative2 = rate
        return [derivative1, derivative2]

**Step 3: Create and Configure the Model**

.. code-block:: python

    model = KineticModel()
    model.set_ode_system(your_ode_system)
    model.add_parameter('rate_constant', initial_guess=0.1, bounds=(0, 10))
    model.add_parameter('order', initial_guess=1.0, bounds=(0.5, 2.5))
    model.set_initial_conditions({'component1': initial_value1,
                                  'component2': initial_value2})

**Step 4: Load Experimental Data**

.. code-block:: python

    time_points = np.array([...])  # Times of measurements
    component1_data = np.array([...])  # Experimental data for component1
    component2_data = np.array([...])  # Experimental data for component2

    model.set_experimental_data(
        time_points=time_points,
        data={'component1': component1_data,
              'component2': component2_data}
    )

**Step 5: Select Optimizer and Fit Parameters**

.. code-block:: python

    set_optimizer('leastsq')
    # Optional: optimizer.configure(method='trf', max_nfev=1000)
    result = fit_kinetic_parameters()

**Step 6: Display Fitted Results**

.. code-block:: python

    print("\nFitted parameters:")
    for name, value in result.parameters.items():
        print(f"{name}: {value:.4f}")
    print("Optimization successful:", result.success)
    print("Objective value:", result.objective_value)
    if result.confidence_intervals:
        print("Confidence intervals (95%):")
        for param, ci in result.confidence_intervals.items():
            print(f"{param}: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")

**Step 7: Visualize Results**

.. code-block:: python

    plot_kinetic_results(
        model=model,
        result=result,
        title="Your Kinetic Model Fit",
        xlabel="Time (units)",
        ylabel="Concentration (units)",
        colors={'component1': 'blue', 'component2': 'red'},
        show_confidence_bands=True,
        separate_plots=False
    )

You can adapt this workflow for batch, CSTR, enzyme kinetics, or complex reaction networks.

Core Concepts
-------------

KineticModel
~~~~~~~~~~~~

The ``KineticModel`` class forms the basis for defining your kinetic system:

.. code-block:: python

    model = KineticModel()
    model.set_ode_system(your_ode_system)
    model.add_parameter('rate_constant', initial_guess=0.5, bounds=(0, 10))
    model.fix_parameter('another_param', value=123)
    model.set_initial_conditions({'A': 1.0, 'B': 0.0})
    model.set_experimental_data(time_points, data)

Optimizers
~~~~~~~~~~

KinFit provides built-in optimizers:

**LeastSquaresOptimizer**: Fast and robust for most problems. Yields parameter estimates and confidence intervals.

.. code-block:: python

    from kinfit.optimizer import LeastSquaresOptimizer
    optimizer = LeastSquaresOptimizer()
    result = optimizer.optimize(model)

**SimulatedAnnealingOptimizer**: Suitable for complex or multimodal problems.

.. code-block:: python

    from kinfit.optimizer import SimulatedAnnealingOptimizer
    optimizer = SimulatedAnnealingOptimizer()
    result = optimizer.optimize(model)

Visualization
~~~~~~~~~~~~~

Use ``plot_kinetic_results`` for model and data visualization:

.. code-block:: python

    plot_kinetic_results(
        model=model,
        result=result,
        title="Kinetic Model Fit",
        xlabel="Time (min)",
        ylabel="Concentration (mol/L)",
        show_confidence_bands=True,
        separate_plots=True,
        figsize=(10, 6),
        save_path="results.png"
    )

Advanced Usage
--------------

Custom Optimizers
~~~~~~~~~~~~~~~~~
Add your own optimization algorithm:

.. code-block:: python

    from kinfit.optimizer import Optimizer, add_custom_optimizer

    class MyOptimizer(Optimizer):
        def __init__(self):
            super().__init__('my_optimizer')
        def optimize(self, model):
            pass
    add_custom_optimizer('my_optimizer', MyOptimizer())

Complex Reactor Systems
~~~~~~~~~~~~~~~~~~~~~~
Model CSTRs, fed-batch, or other reactors by defining suitable ODEs and system parameters.

API Reference
-------------

See detailed class and function documentation:

- :doc:`model` — KineticModel class and methods
- :doc:`optimizer` — Optimization algorithms and settings
- :doc:`plots` — Visualization functions
- :doc:`utils` — Utilities to support model fitting

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   model
   optimizer
   plots
   examples

Examples
--------

Explore more use cases in the :doc:`examples` section, including:

- Multiple simultaneous reactions
- CSTR with recycle
- Enzyme kinetics
- Complex reaction networks

Troubleshooting
---------------

**Optimization not converging:**
- Check initial guesses and bounds
- Check data quality
- Try a different optimizer
- Adjust optimizer settings

**No confidence intervals:**
- Ensure enough data points
- Check for identifiable parameters
- Ensure model structure is adequate

Contributing
------------

We welcome contributions! If you have suggestions or find bugs, please open an issue on our `GitHub repository <REPOSITORY_URL>`_.

License
-------

KinFit is licensed under the GNU General Public License v3.0. See the `LICENSE <REPOSITORY_URL/LICENSE>`_ file for details.

Contact
-------

For questions, contact the author (same as MultiCal repository).

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

----

*This project is powered by* **Sphinx** *and the* **ReadTheDocs** *theme.*