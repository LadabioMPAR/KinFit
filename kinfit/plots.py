import matplotlib.pyplot as plt

def plot_fit(t, y_true, y_pred, label='Fit'):
    plt.plot(t, y_true, 'o', label='Experimental')
    plt.plot(t, y_pred, '-', label=label)
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.legend()
    plt.show()

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import norm
from optimizer import LeastSquaresOptimizer  # substitua pelo nome do arquivo onde está a classe

# ===== 1. Modelo Kinetico Simples =====
'''class SimpleModel:
    def __init__(self):
        # parâmetros (value, bounds, optimize)
        self._parameters = {
            'a': {'value': 0.5, 'bounds': (0, 2), 'optimize': True},
            'b': {'value': 5.0, 'bounds': (0, 10), 'optimize': True},
            'c': {'value': 2.0, 'bounds': (-5, 5), 'optimize': True}
        }
        self._experimental_data = {}

    def generate_data(self, u, noise_std=5.0):
        """Gera dados experimentais com ruído"""
        y_true = 0.5 * u**2 + 5 * u + 2
        y_noisy = y_true + np.random.normal(0, noise_std, size=len(u))
        self._experimental_data['y'] = y_noisy
        return u, y_true, y_noisy

    def solve_ode(self, params):
        """Aqui não tem ODE, apenas o modelo analítico"""
        u = self._u
        y = params['a'] * u**2 + params['b'] * u + params['c']
        return {'y': y}

    def set_input(self, u):
        self._u = u


# ===== 2. Main =====
if __name__ == "__main__":
    np.random.seed(42)

    # Geração de dados
    model1 = SimpleModel()
    u = np.linspace(0, 10, 100)
    model1.set_input(u)
    u, y_true, y_noisy = model1.generate_data(u)

    # Rodar otimização
    optimizer = LeastSquaresOptimizer()
    result = optimizer.optimize(model1)

    print("Parâmetros ajustados:", result.parameters)
    print("IC 95%:", result.confidence_intervals)

    # Predição do modelo ajustado
    y_fit = model1.solve_ode(result.parameters)['y']

    # Bandas de confiança
    ci_lower = result.confidence_bands['y']['ci_lower']
    ci_upper = result.confidence_bands['y']['ci_upper']

    # ===== 3. Plot =====
    plt.figure(figsize=(8, 6))
    plt.scatter(u, y_noisy, s=15, label="Dados experimentais", color="gray")
    plt.plot(u, y_true, "k--", label="Modelo verdadeiro")
    plt.plot(u, y_fit, "b", label="Ajuste LS")
    plt.fill_between(u, ci_lower, ci_upper, color="blue", alpha=0.2, label="IC 95%")

    plt.xlabel("u")
    plt.ylabel("y")
    plt.legend()
    plt.title("Ajuste por Mínimos Quadrados com Intervalo de Confiança")
    plt.show()'''


'''import numpy as np
import matplotlib.pyplot as plt
from optimizer import LeastSquaresOptimizer  # substitua pelo nome real do arquivo onde está a classe


# ===== Modelo Simples =====
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.stats import t

# Função do modelo simples
def simple_model(u, a, b, c):
    return a * u**2 + b * u + c

# Função para gerar dados experimentais
def generate_data(u, noise_std=15.0, missing="none"):
    y_true = 0.5 * u**2 + 5 * u + 2
    y_noisy = y_true + np.random.normal(0, noise_std, size=len(u))

    # aplica dados faltantes
    if missing == "middle":
        mask = (u < 3) | (u > 7)
        u, y_true, y_noisy = u[mask], y_true[mask], y_noisy[mask]
    elif missing == "boundary":
        mask = (u > 2) & (u < 8)
        u, y_true, y_noisy = u[mask], y_true[mask], y_noisy[mask]

    return u, y_true, y_noisy

# Função de resíduos para o least_squares
def residuals(params, u, y_exp):
    a, b, c = params
    y_pred = simple_model(u, a, b, c)
    return y_exp - y_pred

# Função para rodar ajuste e plot

def run_case(title, missing="none"):
    u_full = np.linspace(0, 10, 100)
    u, y_true, y_noisy = generate_data(u_full, missing=missing)

    # Ajuste por mínimos quadrados usando apenas os dados disponíveis
    initial_guess = [0.5, 5.0, 2.0]
    bounds = ([0, 0, -5], [2, 10, 5])
    result = least_squares(residuals, initial_guess, args=(u, y_noisy), bounds=bounds)

    # Parâmetros ajustados
    a_fit, b_fit, c_fit = result.x

    # Predição e bandas de confiança para todo o intervalo [0, 10]
    y_fit_full = simple_model(u_full, a_fit, b_fit, c_fit)

    # Estatísticas para bandas de confiança (usando u_full)
    dof = len(u) - len(result.x)
    if dof > 0:
        jacobian = np.empty((len(u_full), 3))
        jacobian[:, 0] = u_full**2
        jacobian[:, 1] = u_full
        jacobian[:, 2] = 1
        ssr = 2 * result.cost
        residual_variance = ssr / dof
        cov_matrix = np.linalg.pinv(result.jac.T @ result.jac) * residual_variance
        std_error_pred = np.sqrt(np.diag(jacobian @ cov_matrix @ jacobian.T))
        alpha = 1.0 - 0.95
        t_critical = t.ppf(1.0 - alpha / 2.0, df=dof)
        delta = t_critical * std_error_pred
        ci_lower_full = y_fit_full - delta
        ci_upper_full = y_fit_full + delta
    else:
        ci_lower_full = ci_upper_full = y_fit_full

    # Modelo verdadeiro para todo o intervalo
    y_true_full = 0.5 * u_full**2 + 5 * u_full + 2

    # Plot
    plt.figure(figsize=(7, 5))
    plt.scatter(u, y_noisy, s=15, color="gray", label="Dados experimentais")
    plt.plot(u_full, y_true_full, "k--", label="Modelo verdadeiro")
    plt.plot(u_full, y_fit_full, "b", label="Ajuste LS")
    plt.fill_between(u_full, ci_lower_full, ci_upper_full, color="blue", alpha=0.2, label="IC 95%")
    plt.xlim(0,10)
    plt.xlabel("u")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.show()


# MAIN
if __name__ == "__main__":
    np.random.seed(42)
    run_case("Caso 1: Todos os dados disponíveis", missing="none")
    run_case("Caso 3: Dados faltando nas extremidades", missing="boundary")
    run_case("Caso 2: Dados faltando no meio", missing="middle")
'''
import numpy as np
import matplotlib.pyplot as plt
from optimizer import LeastSquaresOptimizer
from model import KineticModel


'''def create_model(u_full):
    """Cria e configura o modelo"""
    model = KineticModel()

    # Adiciona parâmetros
    model.add_parameter('a', initial_guess=0.5, bounds=(0, 2))
    model.add_parameter('b', initial_guess=5.0, bounds=(0, 10))
    model.add_parameter('c', initial_guess=2.0, bounds=(-5, 5))

    # Configuração inicial
    model.set_initial_conditions({'y': 0})
    model._time_points = u_full
    model.u = u_full

    return model


def generate_data(model, noise_std=15.0, missing="none"):
    """Gera dados experimentais com ruído"""
    u = model._time_points.copy()
    y_true = 0.5 * u**2 + 5 * u + 2
    y_noisy = y_true + np.random.normal(0, noise_std, size=len(u))

    if missing == "middle":
        mask = (u < 3) | (u > 7)
        u, y_noisy = u[mask], y_noisy[mask]
    elif missing == "boundary":
        mask = (u > 2) & (u < 8)
        u, y_noisy = u[mask], y_noisy[mask]

    model.u = u
    model._time_points = u
    model.set_experimental_data(u, {'y': y_noisy})
    return u, y_noisy


# Sobrescreve o método solve_ode do modelo
def solve_simple_model(model, params):
    u = model.u
    y = params['a'] * u ** 2 + params['b'] * u + params['c']
    return {'y': y}


# MAIN
if __name__ == "__main__":
    np.random.seed(42)
    u_full = np.linspace(0, 10, 100)
    y_true_full = 0.5 * u_full ** 2 + 5 * u_full + 2

    optimizer = LeastSquaresOptimizer()

    for title, missing in [
        ("Caso 1: Todos os dados disponíveis", "none"),
        ("Caso 2: Dados faltando no meio", "middle"),
        ("Caso 3: Dados faltando nas extremidades", "boundary")
    ]:
        # Cria modelo e gera dados
        model = create_model(u_full)
        u_exp, y_noisy = generate_data(model, missing=missing)

        # Substitui o método solve_ode
        model.solve_ode = lambda params, time_points=None: solve_simple_model(model, params)

        # Otimiza
        result = optimizer.optimize(model)

        # Calcula predição paratodo o intervalo [0, 10]
        model.u = u_full
        y_fit_full = model.solve_ode(result.parameters)['y']

        # Extrai bandas de confiança
        ci_lower_full = result.confidence_bands['y']['ci_lower']
        ci_upper_full = result.confidence_bands['y']['ci_upper']

        # Plot
        plt.figure(figsize=(7, 5))
        plt.scatter(u_exp, y_noisy, s=15, color="gray", label="Dados experimentais")
        plt.plot(u_full, y_true_full, "k--", label="Modelo verdadeiro")
        plt.plot(u_full, y_fit_full, "b", label="Ajuste LS")
        plt.fill_between(u_full, ci_lower_full, ci_upper_full, color="blue", alpha=0.2, label="IC 95%")
        plt.xlim(0, 10)
        plt.xlabel("u")
        plt.ylabel("y")
        plt.title(title)
        plt.legend()
        plt.show()

        # Imprime resultados
        print(f"\n{title}")
        print(f"Sucesso: {result.success}")
        print(f"Parâmetros ajustados:")
        for param, value in result.parameters.items():
            ci = result.confidence_intervals[param]
            print(f"  {param} = {value:.4f} [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")'''


def plot_kinetic_results(model, result, time_points=None, variables=None,
                         title="Ajuste de Modelo Cinético", xlabel="Tempo",
                         ylabel="Concentração", xlim=None, ylim=None,
                         figsize=(10, 6), show_confidence_bands=True,
                         colors=None, line_width=2, marker_size=15,
                         confidence_alpha=0.2, legend_loc="best",
                         grid=False, save_path=None, dpi=300,
                         separate_plots=False, n_points_pred=200):
    """
    Plota resultados do ajuste de modelo cinético com intervalos de confiança

    Args:
        model: Instância do KineticModel
        result: OptimizationResult do ajuste
        time_points: Pontos de tempo para predição (se None, usa linspace automático)
        variables: Lista de variáveis para plotar (se None, plota todas)
        title, xlabel, ylabel: Textos do gráfico
        xlim, ylim: Limites dos eixos
        figsize: Tamanho da figura
        show_confidence_bands: Mostrar IC 95%
        colors: Dicionário de cores por variável
        line_width, marker_size: Espessuras
        confidence_alpha: Transparência do IC
        legend_loc: Posição da legenda
        grid: Mostrar grade
        save_path: Caminho para salvar
        dpi: Resolução da imagem
        separate_plots: Subplot por variável
        n_points_pred: Número de pontos para predição suave
    """
    if model is None or result is None:
        raise RuntimeError("Modelo ou resultado não fornecido.")

    # Dados experimentais
    exp_data = model._experimental_data
    time_exp = model._time_points

    # Pontos para predição suave
    if time_points is None:
        time_points = np.linspace(time_exp[0], time_exp[-1], n_points_pred)

    # Calcula predição do modelo ajustado
    predictions = model.solve_ode(result.parameters, time_points)

    # Define quais variáveis plotar
    variables = variables or list(exp_data.keys())

    # Cores padrão
    if colors is None:
        default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        colors = {var: default_colors[i % len(default_colors)]
                  for i, var in enumerate(variables)}

    # Cria figura
    if separate_plots:
        fig, axes = plt.subplots(len(variables), 1,
                                 figsize=(figsize[0], figsize[1] * len(variables) / 2))
        axes = [axes] if len(variables) == 1 else axes
    else:
        fig, ax = plt.subplots(figsize=figsize)
        axes = [ax] * len(variables)

    # Plota cada variável
    for idx, var in enumerate(variables):
        if var not in exp_data:
            print(f"Aviso: Variável '{var}' não encontrada.")
            continue

        ax = axes[idx] if separate_plots else axes[0]
        color = colors.get(var, 'blue')

        # Dados experimentais
        ax.scatter(time_exp, exp_data[var], s=marker_size, color=color,
                   alpha=0.6, label=f"{var} (experimental)", zorder=3)

        # Predição do modelo ajustado
        ax.plot(time_points, predictions[var], color=color,
                linewidth=line_width, label=f"{var} (ajuste)", zorder=2)

        # Bandas de confiança
        if show_confidence_bands and result.confidence_bands:
            if var in result.confidence_bands:
                ci_lower_exp = result.confidence_bands[var]['ci_lower']
                ci_upper_exp = result.confidence_bands[var]['ci_upper']

                # Interpola para time_points se necessário
                if len(ci_lower_exp) != len(time_points):
                    ci_lower = np.interp(time_points, time_exp, ci_lower_exp)
                    ci_upper = np.interp(time_points, time_exp, ci_upper_exp)
                else:
                    ci_lower = ci_lower_exp
                    ci_upper = ci_upper_exp

                ax.fill_between(time_points, ci_lower, ci_upper,
                                color=color, alpha=confidence_alpha,
                                label=f"{var} IC 95%", zorder=1)

        # Configurações do eixo
        if xlim: ax.set_xlim(xlim)
        if ylim: ax.set_ylim(ylim)
        if grid: ax.grid(False, alpha=0.3)

        if separate_plots:
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel(f"{var} - {ylabel}", fontsize=11)
            ax.set_title(f"{title} - {var}", fontsize=12)

        ax.legend(loc=legend_loc, fontsize=9)

    # Configurações gerais para plot único
    if not separate_plots: #separar os plots para imagens diferentes
        axes[0].set_xlabel(xlabel, fontsize=11)
        axes[0].set_ylabel(ylabel, fontsize=11)
        axes[0].set_title(title, fontsize=12)

    plt.tight_layout()
    if save_path: #mudar lógica quando colocar plots em duas folhas separadas
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()


def print_fit_results(result, title="Resultados do Ajuste"):
    """Imprime os resultados do ajuste formatados"""
    print(f"\n{title}")
    print(f"Sucesso: {result.success}")
    print(f"Valor objetivo: {result.objective_value:.6e}")
    print(f"\nParâmetros ajustados:")
    for param, value in result.parameters.items():
        if result.confidence_intervals and param in result.confidence_intervals:
            ci = result.confidence_intervals[param]
            print(f"  {param} = {value:.4f} [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
        else:
            print(f"  {param} = {value:.4f}")
