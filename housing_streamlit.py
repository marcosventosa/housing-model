
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

# Configuración general de la página en modo "wide"
st.set_page_config(page_title="Simulación Mercado Viviendas", layout="wide")
plt.style.use('default')

def simulate_housing_market_with_arrays(
    T,
    alpha_d, beta_d, gamma_d, delta_d, theta_d,  # Parámetros de la DEMANDA (θ_d es nuevo)
    alpha_s, beta_s,                             # Parámetros de la OFERTA (nuevas construcciones)
    k,                                           # Velocidad de ajuste de precios
    P0, Qs0,                                     # Precio y stock de vivienda iniciales
    r_array,                                     # Tasa de interés en cada periodo (longitud >= T+1)
    C_array,                                     # Coste base de construcción en cada periodo (longitud >= T+1)
    c_fin,                                       # Coste financiero (se multiplica por r)
    X_array=None,                                # Array con variable exógena X, si None => 0
    Y_array=None,                                # Array con la renta disponible, si None => 0
    R_const=0,                                   # Demoliciones fijas por periodo
    construction_lag=1                           # Retardo en la construcción de viviendas
):
    """
    Simula un modelo dinámico de oferta y demanda de vivienda con las ecuaciones extendidas:

      1) Qd(t) = alpha_d - beta_d·P(t) - gamma_d·r(t) + delta_d·X(t) + theta_d·Y(t)
      2) cost_of_construction(t) = C(t) + c_fin·r(t)
      3) I(t) = max(0, alpha_s + beta_s·[P(t) - costC(t)])
      4) Qs(t+1) = Qs(t) + I(t - construction_lag) - R_const,  (si t - construction_lag >= 0)
         de lo contrario, Qs(t+1) = Qs(t) - R_const
      5) P(t+1) = P(t) + k·[Qd(t) - Qs(t)]
    """

    if X_array is None:
        X_array = np.zeros(T+1)
    if Y_array is None:
        Y_array = np.zeros(T+1)

    # Verificaciones mínimas de longitud
    if len(r_array) < T+1 or len(C_array) < T+1 or len(X_array) < T+1 or len(Y_array) < T+1:
        raise ValueError("Los arrays r_array, C_array, X_array y/o Y_array deben tener al menos T+1 elementos.")

    # Arrays de resultados
    P = np.zeros(T+1)
    Qs = np.zeros(T+1)
    Qd = np.zeros(T+1)
    I = np.zeros(T+1)      # Guardamos I hasta T+1 para manejar fácilmente el lag
    costC = np.zeros(T+1)

    # Condiciones iniciales
    P[0] = P0
    Qs[0] = Qs0

    for t in range(T):
        # 1) Demanda
        Qd[t] = (
            alpha_d
            - beta_d * P[t]
            - gamma_d * r_array[t]
            + delta_d * X_array[t]
            + theta_d * Y_array[t]
        )
        if Qd[t] < 0:
            Qd[t] = 0

        # 2) Coste de construcción
        costC[t] = C_array[t] + c_fin * r_array[t]

        # 3) Construcción (nuevas viviendas)
        rentabilidad = P[t] - costC[t]
        I[t] = max(0, alpha_s + beta_s * rentabilidad)

        # 4) Nueva oferta con retardo (construction_lag)
        if t - construction_lag >= 0:
            Qs[t+1] = Qs[t] + I[t - construction_lag] - R_const
        else:
            # Antes de que se cumpla el retardo, no se suman las construcciones
            Qs[t+1] = Qs[t] - R_const

        if Qs[t+1] < 0:
            Qs[t+1] = 0

        # 5) Ajuste de precio al final del periodo t
        P[t+1] = P[t] + k * (Qd[t] - Qs[t])
        if P[t+1] < 0:
            P[t+1] = 0

    # Último periodo: Qd[T] y costC[T]
    costC[T] = C_array[T] + c_fin * r_array[T]
    Qd[T] = (
        alpha_d
        - beta_d * P[T]
        - gamma_d * r_array[T]
        + delta_d * X_array[T]
        + theta_d * Y_array[T]
    )
    if Qd[T] < 0:
        Qd[T] = 0

    return {
        'P': P,
        'Qs': Qs,
        'Qd': Qd,
        'I': I,
        'costC': costC
    }

def main():
    st.title("Simulación Dinámica del Mercado de Viviendas")

    st.write("""
    Este modelo lineal básico simula la evolución de la oferta (stock), la demanda y el precio  
    de las viviendas a lo largo de T periodos. Permite introducir shocks en la tasa de interés,  
    el coste de construcción, la variable exógena (X) y la renta (Y). Igualmente, incorpora un  
    retraso (construction_lag) con el que las nuevas construcciones (I) se añaden al stock (Qs).  
    """)

    # Sección con las ecuaciones en LaTeX (extendidas), centradas
    st.markdown(r"""
    ### Formulación del Modelo (Extendida)
    $$
    \begin{aligned}
    Q_{d}(t) &= \alpha_{d}\;-\;\beta_{d}\,P(t)\;-\;\gamma_{d}\,r(t)\;+\;\delta_{d}\,X(t)\;+\;\theta_{d}\,Y(t),\\[6pt]
    \text{cost\_of\_construction}(t) &= C(t)\;+\;c_{\text{fin}}\;r(t),\\[6pt]
    I(t) &= \max\!\Bigl(0,\;\alpha_{s}\;+\;\beta_{s}\bigl[P(t)\;-\;\text{cost\_of\_construction}(t)\bigr]\Bigr),\\[6pt]
    Q_{s}(t+1) &= 
      \begin{cases}
        Q_s(t) + I(t - \text{construction\_lag}) - R_{\text{const}}, & \text{si } t \ge \text{construction\_lag},\\
        Q_s(t) - R_{\text{const}}, & \text{si } t < \text{construction\_lag}.
      \end{cases}\\[6pt]
    P(t+1) &= P(t)\;+\;k\;\bigl[Q_{d}(t)\;-\;Q_{s}(t)\bigr].
    \end{aligned}
    $$
    """)

    # Parámetros y descripciones en tabla HTML (versión "larga" con filas adicionales)
    st.markdown("""
    <h3>Parámetros y Descripciones</h3>
    <table style="width:100%; border-collapse: collapse;">
      <tr style="border-bottom: 1px solid black;">
        <th style="text-align:left; padding: 4px;">Parámetro</th>
        <th style="text-align:left; padding: 4px;">Descripción</th>
      </tr>
      <tr>
        <td>Q<sub>s</sub></td>
        <td>Stock de viviendas en el mercado.</td>
      </tr>
      <tr>
        <td>P</td>
        <td>Precio de las viviendas (monetario).</td>
      </tr>
      <tr>
        <td>Q<sub>d</sub></td>
        <td>Demanda de viviendas.</td>
      </tr>
      <tr>
        <td>I</td>
        <td>Nuevas viviendas construidas por periodo. Se añade al stock con un retardo (<em>construction_lag</em>).</td>
      </tr>
      <tr>
        <td>T</td>
        <td>Número total de periodos a simular.</td>
      </tr>
      <tr>
        <td>α<sub>d</sub></td>
        <td>Demanda base cuando P, r, X y Y están en valores normales.</td>
      </tr>
      <tr>
        <td>β<sub>d</sub></td>
        <td>Sensibilidad (positiva) de la demanda ante un cambio en el precio.</td>
      </tr>
      <tr>
        <td>γ<sub>d</sub></td>
        <td>Sensibilidad (positiva) de la demanda ante cambios en la tasa de interés (r).</td>
      </tr>
      <tr>
        <td>δ<sub>d</sub></td>
        <td>Sensibilidad de la demanda ante la variable exógena X(t).</td>
      </tr>
      <tr>
        <td>θ<sub>d</sub></td>
        <td>Sensibilidad de la demanda ante la renta Y(t).</td>
      </tr>
      <tr>
        <td>α<sub>s</sub></td>
        <td>Término base de nueva construcción (puede ser negativo).</td>
      </tr>
      <tr>
        <td>β<sub>s</sub></td>
        <td>Sensibilidad de la oferta a la rentabilidad (P - C).</td>
      </tr>
      <tr>
        <td>k</td>
        <td>Velocidad de ajuste de precio, controla la retroalimentación (Q<sub>d</sub> - Q<sub>s</sub>).</td>
      </tr>
      <tr>
        <td>P<sub>0</sub></td>
        <td>Precio inicial en el periodo t=0.</td>
      </tr>
      <tr>
        <td>Qs<sub>0</sub></td>
        <td>Stock inicial de viviendas en t=0.</td>
      </tr>
      <tr>
        <td>r<sub>0</sub></td>
        <td>Tasa de interés inicial (porcentaje).</td>
      </tr>
      <tr>
        <td>C<sub>0</sub></td>
        <td>Coste base de construcción inicial.</td>
      </tr>
      <tr>
        <td>X<sub>0</sub></td>
        <td>Variable exógena inicial (por ejemplo, tendencias demográficas).</td>
      </tr>
      <tr>
        <td>Y<sub>0</sub></td>
        <td>Renta base inicial (por ejemplo, ingreso disponible).</td>
      </tr>
      <tr>
        <td>c<sub>fin</sub></td>
        <td>Componente de coste financiero (se multiplica por r(t)).</td>
      </tr>
      <tr>
        <td>R<sub>const</sub></td>
        <td>Demoliciones fijas por periodo (viviendas que se retiran).</td>
      </tr>
      <tr>
        <td>construction_lag</td>
        <td>Retardo (en periodos) con el que las nuevas construcciones se agregan al stock.</td>
      </tr>
    </table>
    """, unsafe_allow_html=True)

    # Barra lateral: parámetros de configuración del modelo
    with st.sidebar:
        st.header("Parámetros del Modelo")

        T = st.number_input(
            "Número de periodos (T)",
            min_value=10, max_value=200000, value=20000, step=1000,
            help="Cantidad total de periodos a simular."
        )

        st.subheader("Demanda")
        alpha_d = st.number_input("α_d (Demanda base)", value=400000, step=10000)
        beta_d  = st.number_input("β_d (Sensib. al precio)", value=0.2, step=0.01)
        gamma_d = st.number_input("γ_d (Sensib. a interés)", value=20000, step=1000)
        delta_d = st.number_input("δ_d (Sensib. a variable X)", value=200, step=10)
        theta_d = st.number_input("θ_d (Sensib. a la renta Y)", value=50.0, step=10.0)

        st.subheader("Oferta")
        alpha_s = st.number_input("α_s (Construcción base)", value=-10, step=1)
        beta_s  = st.number_input("β_s (Sensib. a rentabilidad)", value=0.001, step=0.0001, format="%.6f")

        st.subheader("Ajuste de Precio")
        k = st.number_input("k (Velocidad de ajuste)", min_value=0.0001, value=0.01, step=0.001, format="%.4f")

        st.subheader("Costos y Financiación")
        c_fin = st.number_input("c_fin (Coste financiero)", value=20000, step=1000)

        st.subheader("Condiciones Iniciales")
        P0 = st.number_input("P0 (Precio inicial)", value=300000, step=10000)
        Qs0 = st.number_input("Qs0 (Stock inicial)", value=50000, step=1000)
        r0 = st.number_input("r0 (Tasa de interés inicial, %)", value=5.0, step=0.5, format="%.2f")
        C0 = st.number_input("C0 (Coste base construcción inicial)", value=200000, step=10000)
        X0 = st.number_input("X0 (Variable exógena inicial)", value=5, step=1)
        Y0 = st.number_input("Y0 (Renta inicial)", value=2000, step=100)
        R0 = st.number_input("R_const (Demoliciones fijas)", value=100, step=10)
        construction_lag = st.number_input("construction_lag (Retardo en construcción)", min_value=0, value=1, step=1)

        st.subheader("Shocks Opcionales")

        apply_interest_shock = st.checkbox("Aplicar shock en la tasa de interés", value=True)
        if apply_interest_shock:
            shock_time_r = st.number_input(
                "Periodo para el shock de interés",
                min_value=1, max_value=T, value=int(T/2)
            )
            interest_after_shock = st.number_input(
                "Tasa de interés tras el shock (%)",
                value=1.0, step=0.5, format="%.2f"
            )
        else:
            shock_time_r = None
            interest_after_shock = None

        apply_cost_shock = st.checkbox("Aplicar shock en el coste de construcción", value=True)
        if apply_cost_shock:
            shock_time_c = st.number_input(
                "Periodo para el shock de coste",
                min_value=1, max_value=T, value=int(T/1.5)
            )
            cost_after_shock = st.number_input(
                "Nuevo coste de construcción (C')",
                value=100000, step=5000
            )
        else:
            shock_time_c = None
            cost_after_shock = None

        apply_X_shock = st.checkbox("Aplicar shock en la variable X")
        if apply_X_shock:
            shock_time_X = st.number_input(
                "Periodo para el shock de X",
                min_value=1, max_value=T, value=int(T/2)
            )
            X_after_shock = st.number_input(
                "Nuevo valor de X tras el shock",
                value=8.0, step=0.5, format="%.2f"
            )
        else:
            shock_time_X = None
            X_after_shock = None

        
        apply_Y_shock = st.checkbox("Aplicar shock en la renta (Y)")
        if apply_Y_shock:
            shock_time_Y = st.number_input(
                "Periodo para el shock de Y",
                min_value=1, max_value=T, value=int(T/2)
            )
            Y_after_shock = st.number_input(
                "Nuevo valor de Y tras el shock",
                value=3000.0, step=100.0, format="%.2f"
            )
        else:
            shock_time_Y = None
            Y_after_shock = None

    # Construir los arrays según los shocks
    r_array = np.full(T+1, r0)
    C_array = np.full(T+1, C0)
    X_array = np.full(T+1, X0)
    Y_array = np.full(T+1, Y0)

    # Aplicar shocks
    if apply_interest_shock and shock_time_r is not None:
        r_array[shock_time_r:] = interest_after_shock

    if apply_cost_shock and shock_time_c is not None:
        C_array[shock_time_c:] = cost_after_shock

    if apply_X_shock and shock_time_X is not None:
        X_array[shock_time_X:] = X_after_shock

    if apply_Y_shock and shock_time_Y is not None:
        Y_array[shock_time_Y:] = Y_after_shock

    # Ejecutar la simulación
    results = simulate_housing_market_with_arrays(
        T,
        alpha_d, beta_d, gamma_d, delta_d, theta_d,
        alpha_s, beta_s,
        k,
        P0, Qs0,
        r_array, C_array, c_fin,
        X_array=X_array,
        Y_array=Y_array,
        R_const=R0,
        construction_lag=construction_lag
    )

    P = results['P']
    Qs = results['Qs']
    Qd = results['Qd']
    I = results['I']
    costC = results['costC']

    # Mostrar resultados (gráficos)
    st.subheader("Resultados de la Simulación")

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    def plot_shocks(ax):
        """Dibuja en el eje 'ax' líneas verticales para cada shock (si aplica)."""
        if apply_interest_shock and shock_time_r is not None:
            ax.axvline(shock_time_r, color='red', linestyle='--', linewidth=1.0, label="Shock interés")
        if apply_cost_shock and shock_time_c is not None:
            ax.axvline(shock_time_c, color='blue', linestyle='--', linewidth=1.0, label="Shock coste")
        if apply_X_shock and shock_time_X is not None:
            ax.axvline(shock_time_X, color='green', linestyle='--', linewidth=1.0, label="Shock X")
        if apply_Y_shock and shock_time_Y is not None:
            ax.axvline(shock_time_Y, color='orange', linestyle='--', linewidth=1.0, label="Shock Y")

    # (a) Precio (monetario)
    axs[0,0].plot(range(T+1), P, label="Precio")
    plot_shocks(axs[0,0])
    axs[0,0].set_title("Evolución del Precio")
    axs[0,0].grid(True)
    axs[0,0].legend()
    # Formato miles y símbolo monetario
    axs[0,0].yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}'))

    # (b) Oferta vs. Demanda (no monetario, pero miles)
    axs[0,1].plot(range(T+1), Qs, 'r', label="Stock ofertado (Qs)")
    axs[0,1].plot(range(T+1), Qd, 'g', label="Demanda (Qd)")
    plot_shocks(axs[0,1])
    axs[0,1].set_title("Oferta vs. Demanda")
    axs[0,1].grid(True)
    axs[0,1].legend()
    axs[0,1].yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

    # (c) Construcción I (cantidad)
    axs[1,0].plot(range(T+1), I, 'm', label="Construcción (I)")
    plot_shocks(axs[1,0])
    axs[1,0].set_title("Nuevas Viviendas Construidas (por periodo)")
    axs[1,0].grid(True)
    axs[1,0].legend()
    axs[1,0].yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

    # (d) Coste de construcción (monetario)
    axs[1,1].plot(range(T+1), costC, 'k', label="Coste construcción (costC)")
    plot_shocks(axs[1,1])
    axs[1,1].set_title("Coste Efectivo de Construcción")
    axs[1,1].grid(True)
    axs[1,1].legend()
    axs[1,1].yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}'))

    plt.tight_layout()
    st.pyplot(fig)

if __name__ == "__main__":
    main()
