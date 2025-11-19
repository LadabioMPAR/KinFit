import numpy as np
from scipy.stats import t
from typing import Dict, Optional
from .optimizer import OptimizationResult



def conf_interval_jac(
    result: OptimizationResult,
    param_names: list[str],
    initial_guess: list[float],
    confidence_level: float = 0.95
) -> OptimizationResult:
    """
    Calcula matriz de covariância, desvios padrão e intervalos de confiança
    a partir do resultado de uma otimização.

    Parâmetros
    ----------
    result : OptimizationResult
        Resultado do ajuste contendo jacobiano, resíduos e custo.
    param_names : list[str]
        Nomes dos parâmetros ajustados.
    initial_guess : list[float]
        Valores iniciais dos parâmetros.
    confidence_level : float
        Nível de confiança (default = 0.95).

    Retorna
    -------
    result : OptimizationResult
        Mesmo objeto, mas com covariance_matrix e confidence_intervals preenchidos.
    """
    if not result.success:
        return result

    try:
        # 1. Obter número de dados e parâmetros
        num_data_points = len(result.fun)
        num_params = len(initial_guess)
        dof = num_data_points - num_params

        if dof <= 0:
            print("Aviso: graus de liberdade insuficientes para cálculo de IC.")
            return result

        # 2. Obter Jacobiana e SSR
        jacobian = result.jac
        ssr = 2 * result.cost  # Compatível com least_squares do SciPy
        residual_variance = ssr / dof

        # 3. Calcular matriz de covariância
        cov_matrix = np.linalg.pinv(jacobian.T @ jacobian) * residual_variance

        # 4. Calcular desvios padrão e ICs
        std_errors = np.sqrt(np.diag(cov_matrix))
        alpha = 1.0 - confidence_level
        t_critical = t.ppf(1.0 - alpha / 2.0, df=dof)

        conf_intervals = {}
        for i, param_name in enumerate(param_names):
            value = result.x[i]
            se = std_errors[i]
            conf_intervals[param_name] = {
                'ci_lower': value - t_critical * se,
                'ci_upper': value + t_critical * se,
                'std_error': se
            }

        result.covariance_matrix = cov_matrix
        result.confidence_intervals = conf_intervals

    except np.linalg.LinAlgError:
        print("Aviso: não foi possível calcular estatísticas (matriz singular).")

    return result