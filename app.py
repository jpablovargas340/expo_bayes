import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import gamma as gamma_func
import io
import contextlib

# ═══════════════════════════════════════════════════════════
# CONFIGURACIÓN GLOBAL
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Métodos Computacionales · Estadística Bayesiana",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.stApp { background: #0d1117; color: #e6edf3; }

section[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #21262d;
}

.card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.hero-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: #58a6ff;
    letter-spacing: -0.5px;
}

.hero-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: #8b949e;
    margin-top: 0.2rem;
}

.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #3fb950;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.5rem;
}

h2, h3 { color: #e6edf3 !important; font-family: 'IBM Plex Sans', sans-serif !important; }

.code-block {
    background: #0d1117;
    border: 1px solid #21262d;
    border-left: 3px solid #58a6ff;
    border-radius: 4px;
    padding: 1rem 1.2rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    color: #e6edf3;
    overflow-x: auto;
    white-space: pre;
    line-height: 1.7;
}

.graph-interp {
    background: #0d1f0d;
    border: 1px solid #3fb950;
    border-radius: 6px;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0 1rem 0;
    font-size: 0.88rem;
    line-height: 1.65;
    color: #b5e4b5;
}
.graph-interp strong { color: #3fb950; }
.graph-interp .graph-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #3fb950;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    display: block;
    margin-bottom: 0.4rem;
}

.interp-box {
    background: #0f2027;
    border: 1px solid #1f6feb;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    font-size: 0.92rem;
    line-height: 1.6;
    color: #cdd9e5;
}
.interp-box strong { color: #58a6ff; }

.note-box {
    background: #1a1a0f;
    border: 1px solid #d29922;
    border-radius: 6px;
    padding: 0.8rem 1.2rem;
    margin: 0.6rem 0;
    font-size: 0.88rem;
    color: #e3b341;
}

.result-value {
    font-family: 'IBM Plex Mono', monospace;
    color: #3fb950;
    font-weight: 600;
}

div[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 0.8rem;
}
div[data-testid="metric-container"] label {
    color: #8b949e !important;
    font-size: 0.78rem !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #58a6ff !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

div[data-baseweb="tab-list"] {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 0.55rem !important;
    align-items: center !important;
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    padding: 0.75rem !important;
    margin: 0.8rem 0 1.2rem 0 !important;
}
button[data-baseweb="tab"],
div[data-baseweb="tab"] button {
    min-height: 42px !important;
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    padding: 0.65rem 0.95rem !important;
    color: #c9d1d9 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 0 rgba(255,255,255,0.03) inset;
}
button[data-baseweb="tab"] p,
div[data-baseweb="tab"] button p {
    margin: 0 !important;
    white-space: nowrap !important;
}
button[data-baseweb="tab"]:hover,
div[data-baseweb="tab"] button:hover {
    background: #1f2937 !important;
    border-color: #58a6ff !important;
    color: #e6edf3 !important;
}
button[data-baseweb="tab"][aria-selected="true"],
div[data-baseweb="tab"][aria-selected="true"] button {
    color: #ffffff !important;
    background: #0f2a45 !important;
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 1px rgba(88,166,255,0.25), 0 8px 20px rgba(1,4,9,0.25);
}
div[data-baseweb="tab-highlight"] { background-color: transparent !important; }
div[data-baseweb="tab-border"] { display: none !important; }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #21262d; border-radius: 3px; }

details {
    border: 1px solid #21262d !important;
    border-radius: 6px !important;
    background: #161b22 !important;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# UTILIDADES
# ═══════════════════════════════════════════════════════════
def fig_style():
    plt.rcParams.update({
        "figure.facecolor": "#0d1117",
        "axes.facecolor": "#0d1117",
        "axes.edgecolor": "#21262d",
        "axes.labelcolor": "#8b949e",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "text.color": "#e6edf3",
        "grid.color": "#21262d",
        "grid.alpha": 0.5,
        "font.family": "monospace",
        "font.size": 10,
        "legend.facecolor": "#161b22",
        "legend.edgecolor": "#21262d",
    })

def run_code(code_str: str):
    buf = io.StringIO()
    local_vars = {}
    with contextlib.redirect_stdout(buf):
        exec(code_str, {
            "np": np, "pd": pd, "plt": plt,
            "stats": stats, "gamma_func": gamma_func,
            "__builtins__": __builtins__
        }, local_vars)
    return buf.getvalue(), local_vars

def interp(text: str):
    st.markdown(f'<div class="interp-box">💡 {text}</div>', unsafe_allow_html=True)

def graph_interp(titulo: str, texto: str):
    st.markdown(
        f'<div class="graph-interp">'
        f'<span class="graph-title">📊 Lectura de la gráfica — {titulo}</span>'
        f'{texto}</div>',
        unsafe_allow_html=True
    )

def nota(text: str):
    st.markdown(f'<div class="note-box">⚠️ {text}</div>', unsafe_allow_html=True)

def show_code(code: str, key: str):
    st.markdown(f'<div class="code-block">{code}</div>', unsafe_allow_html=True)
    if st.button("▶ Ejecutar código", key=key):
        try:
            out, _ = run_code(code)
            if out.strip():
                st.code(out, language="text")
            else:
                st.success("✓ Código ejecutado sin errores.")
        except Exception as e:
            st.error(f"Error: {e}")

fig_style()

# ═══════════════════════════════════════════════════════════
# CABECERA
# ═══════════════════════════════════════════════════════════
st.markdown('<div class="section-label">tablero interactivo · exposición</div>', unsafe_allow_html=True)
st.markdown("# 🧮 Métodos Computacionales en Estadística Bayesiana")
st.markdown("Presentado por: Juan Pablo Vargas - Edward Mora - Natalia Carrero")
st.markdown("---")

# ═══════════════════════════════════════════════════════════
# TABS PRINCIPALES
# ═══════════════════════════════════════════════════════════
(tab_inicio, tab_bayes, tab_dist, tab_inf,
 tab_mcmc, tab_reg, tab_sel, tab_mc, tab_refs) = st.tabs([
    "🏠 Inicio",
    "1 · Probabilidad y Bayes",
    "2 · Distribuciones",
    "3 · Inferencia Bayesiana",
    "4 · MCMC",
    "5 · Regresión Bayesiana",
    "6 · Selección de Modelos",
    "7 · Monte Carlo",
    "📚 Referencias",
])

# 
# TAB 0 — INICIO
# 
with tab_inicio:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Temas", "8")
    col2.metric("Módulos con código", "7")
    col3.metric("Algoritmos clave", "3+")
    col4.metric("Distribuciones", "6+")

    st.markdown("---")
    st.markdown("""
    <div class="card">
    <div class="section-label">¿De qué trata?</div>
    <h3 style="color:#58a6ff">Estadística Bayesiana + métodos computacionales</h3>
    <p>La estadística bayesiana es una forma de <strong>razonar bajo incertidumbre</strong>.
    En lugar de asumir que un parámetro tiene un valor fijo desconocido, dice:<br><br>
    <span style="color:#3fb950;font-family:'IBM Plex Mono',monospace">
    "Tengo creencias iniciales sobre ese parámetro, y las actualizo con los datos que observo."
    </span><br><br>
    Toda la maquinaria computacional del curso sirve para hacer esa actualización de manera eficiente.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    | Tab | Tema central |
    |-----|-------------|
    | 1 · Probabilidad y Bayes | Regla de Bayes, probabilidad condicional |
    | 2 · Distribuciones | Normal, Beta, Gamma, Poisson, Binomial |
    | 3 · Inferencia Bayesiana | Prior → Posterior (modelos conjugados) |
    | 4 · MCMC | Metropolis-Hastings, traceplot, diagnóstico |
    | 5 · Regresión Bayesiana | Incertidumbre en rectas de regresión |
    | 6 · Selección de Modelos | BIC y Factor de Bayes |
    | 7 · Monte Carlo | Estimación de π, integración, convergencia |
    | Referencias | Fuente bibliográfica principal |
    """)

# 
# TAB 1 — PROBABILIDAD Y BAYES
# 
with tab_bayes:
    st.markdown("## Regla de Bayes")
    st.latex(r"P(\theta \mid D) \;=\; \frac{P(D \mid \theta)\; P(\theta)}{P(D)}")

    st.markdown("""
    | Símbolo | Nombre | ¿Qué significa? |
    |---------|--------|----------------|
    | $P(\\theta \\mid D)$ | **Posterior** | Lo que creemos sobre θ **después** de ver los datos |
    | $P(D \\mid \\theta)$ | **Verosimilitud** | Qué tan probables son los datos si θ fuese cierto |
    | $P(\\theta)$ | **Prior** | Lo que creíamos sobre θ **antes** de los datos |
    | $P(D)$ | **Evidencia** | Constante normalizadora — garantiza que todo sume 1 |
    """)

    interp("La regla de Bayes es una <strong>máquina de aprendizaje</strong>: parte de una creencia (prior), "
           "recibe evidencia (datos) y actualiza la creencia (posterior). Cada nuevo dato refina la estimación.")

    st.markdown("---")
    st.markdown("### Ejemplo interactivo — diagnóstico médico")
    interp("Una prueba médica no es infalible. Bayes nos dice cuál es la probabilidad <em>real</em> "
           "de tener la enfermedad dado un resultado positivo, teniendo en cuenta qué tan común es la enfermedad.")

    col1, col2 = st.columns(2)
    prevalencia   = col1.slider("Prevalencia P(E)", 0.001, 0.5, 0.01, 0.001, format="%.3f",
                                help="Fracción de la población que tiene la enfermedad")
    sensibilidad  = col1.slider("Sensibilidad P(+|E)", 0.5, 1.0, 0.99, 0.01,
                                help="Probabilidad de que el test dé positivo en un enfermo")
    especificidad = col2.slider("Especificidad P(−|no E)", 0.5, 1.0, 0.95, 0.01,
                                help="Probabilidad de que el test dé negativo en un sano")

    falso_positivo = 1 - especificidad
    p_pos          = sensibilidad * prevalencia + falso_positivo * (1 - prevalencia)
    p_e_dado_pos   = (sensibilidad * prevalencia) / p_pos

    col2.markdown(f"""
    <div class="card">
    <div class="section-label">resultado bayesiano</div>
    <p>P(test +) = <span class="result-value">{p_pos:.4f}</span></p>
    <p style="font-size:1.4rem;margin-top:0.5rem">
    P(enfermo | +) = <span class="result-value" style="font-size:1.8rem">{p_e_dado_pos:.2%}</span>
    </p>
    <p style="font-size:0.8rem;color:#8b949e">Probabilidad real de enfermedad dado un test positivo</p>
    </div>
    """, unsafe_allow_html=True)

    if p_e_dado_pos < 0.5:
        nota(f"Aunque el test salió positivo, solo hay {p_e_dado_pos:.1%} de probabilidad real. "
             "Esto es la <strong>falacia de la tasa base</strong>: ignorar la prevalencia infla el miedo.")
    else:
        interp(f"Con esta prevalencia y precisión, un positivo implica {p_e_dado_pos:.1%} de probabilidad real.")

    prevs   = np.linspace(0.001, 0.5, 300)
    p_pos_v = sensibilidad * prevs + falso_positivo * (1 - prevs)
    post_v  = (sensibilidad * prevs) / p_pos_v

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(prevs, post_v, color="#58a6ff", lw=2.5, label="P(E | test +)")
    ax.axvline(prevalencia, color="#3fb950", lw=1.5, linestyle="--",
               label=f"Prevalencia = {prevalencia:.3f}")
    ax.axhline(p_e_dado_pos, color="#f78166", lw=1.5, linestyle="--",
               label=f"P(E|+) = {p_e_dado_pos:.2%}")
    ax.scatter([prevalencia], [p_e_dado_pos], color="#f78166", zorder=5, s=70)
    ax.set_xlabel("Prevalencia P(E)")
    ax.set_ylabel("P(E | test positivo)")
    ax.set_title("Valor predictivo positivo según prevalencia")
    ax.legend(fontsize=8)
    ax.grid(True)
    st.pyplot(fig, use_container_width=True)

    graph_interp("Valor predictivo positivo",
        "La <strong>curva azul</strong> muestra cómo cambia la probabilidad real de estar enfermo "
        "dependiendo de cuán común sea la enfermedad en la población. "
        "La <strong>línea verde vertical</strong> marca la prevalencia actual seleccionada. "
        "La <strong>línea roja horizontal</strong> y el punto indican el resultado bayesiano en ese punto. "
        "Con prevalencias bajas (lado izquierdo), la curva está pegada al suelo: incluso un test muy bueno "
        "tiene bajo poder predictivo. Esto explica por qué hacer pruebas masivas en poblaciones sanas genera muchos falsos positivos.")

    st.markdown("### Código — Regla de Bayes")
    code_bayes = """\
import numpy as np

#Parámetros del problema 

prevalencia    = 0.01   # P(E): 1 de cada 100 personas tiene la enfermedad
sensibilidad   = 0.99   # P(+ | E): el test detecta el 99 % de los enfermos
especificidad  = 0.95   # P(- | no E): el test descarta el 95 % de los sanos

#Derivado
falso_positivo = 1 - especificidad   # P(+ | no E): falla en sanos

#Denominador de Bayes: P(+)
#Tiene dos caminos para dar positivo:
#1. Estar enfermo Y test positivo
#2. Estar sano Y test positivo (falso positivo)
p_positivo = (sensibilidad   * prevalencia +       # camino 1
              falso_positivo * (1 - prevalencia))  # camino 2

#Posterior: numerador / denominador 
#Numerador = P(+|E) · P(E): solo el camino de estar enfermo
p_enfermo_dado_positivo = (sensibilidad * prevalencia) / p_positivo

#Resultados
print(f"P(test positivo)        = {p_positivo:.4f}")
print(f"P(enfermo | positivo)   = {p_enfermo_dado_positivo:.4f}  ({p_enfermo_dado_positivo:.1%})")
print()
print("Conclusión: con prevalencia baja, incluso un test preciso")
print("tiene un valor predictivo positivo sorprendentemente bajo.")
"""
    show_code(code_bayes, "b_bayes")
 
#TAB 2 — DISTRIBUCIONES 
with tab_dist:
    st.markdown("## Distribuciones de Probabilidad")
    interp("Las distribuciones son la <strong>materia prima</strong> de la estadística bayesiana: "
           "modelan los datos (verosimilitud) y expresan creencias sobre parámetros (prior/posterior).")

    sub1, sub2, sub3, sub4 = st.tabs(["Normal", "Beta", "Gamma · Poisson", "Binomial"])

    with sub1:
        st.markdown("### Distribución Normal $\\mathcal{N}(\\mu, \\sigma^2)$")
        st.latex(r"f(x \mid \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)")
        st.markdown("""
        | Variable | Significado |
        |----------|-------------|
        | $\\mu$ | Media: centro de la campana |
        | $\\sigma$ | Desviación estándar: qué tan ancha es la campana |
        | $x$ | Valor para el cual calculamos la densidad |
        """)
        c1, c2 = st.columns(2)
        mu    = c1.slider("μ (media)", -5.0, 5.0, 0.0, 0.1, key="mu_n")
        sigma = c2.slider("σ (desv. estándar)", 0.1, 5.0, 1.0, 0.1, key="sig_n")
        x = np.linspace(-12, 12, 600)
        y = stats.norm.pdf(x, mu, sigma)
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(x, y, color="#58a6ff", lw=2.5)
        ax.fill_between(x, y, where=(x >= mu-sigma) & (x <= mu+sigma),
                        alpha=0.35, color="#58a6ff", label=f"μ±σ = 68%")
        ax.fill_between(x, y, where=(x >= mu-2*sigma) & (x <= mu+2*sigma),
                        alpha=0.15, color="#58a6ff", label=f"μ±2σ = 95%")
        ax.axvline(mu, color="#3fb950", lw=1.8, linestyle="--", label=f"μ = {mu}")
        ax.set_title(f"Normal(μ={mu}, σ={sigma})")
        ax.legend(fontsize=8)
        ax.grid(True)
        st.pyplot(fig, use_container_width=True)
        graph_interp("Distribución Normal",
            f"La <strong>curva azul</strong> es la campana de Gauss centrada en μ = {mu}. "
            f"La <strong>zona oscura</strong> cubre μ±σ = [{mu-sigma:.1f}, {mu+sigma:.1f}]: "
            "el 68% de los datos vive aquí. "
            f"La <strong>zona clara</strong> añade hasta μ±2σ = [{mu-2*sigma:.1f}, {mu+2*sigma:.1f}]: "
            "el 95% de los datos. "
            "A mayor σ, más ancha y baja la campana → más incertidumbre.")
        code_norm = """\
import numpy as np
from scipy import stats

#Definir los parámetros
mu, sigma = 0.0, 1.0   # media y desviación estándar

#Evaluar la PDF en varios puntos
x = np.linspace(-4, 4, 300)
#f(x) = (1/σ√2π) · exp(-(x-μ)²/2σ²)
y = stats.norm.pdf(x, mu, sigma)

#Regla empírica (68 - 95 - 99.7)
print(f"Rango μ ± 1σ  → cubre el 68 %  : [{mu - sigma:.2f}, {mu + sigma:.2f}]")
print(f"Rango μ ± 2σ  → cubre el 95 %  : [{mu - 2*sigma:.2f}, {mu + 2*sigma:.2f}]")
print(f"Rango μ ± 3σ  → cubre el 99.7 %: [{mu - 3*sigma:.2f}, {mu + 3*sigma:.2f}]")

#Probabilidad acumulada (CDF)
#P(X < valor) usando la función de distribución acumulada
print(f"\\nP(X < 0)  = {stats.norm.cdf(0,  mu, sigma):.4f}")
print(f"P(X < 1)  = {stats.norm.cdf(1,  mu, sigma):.4f}")
print(f"P(X < -1) = {stats.norm.cdf(-1, mu, sigma):.4f}")

#Percentiles 
#Inversa de la CDF: ¿cuál es el valor x tal que P(X < x) = q?
print(f"\\nPercentil 2.5 % : {stats.norm.ppf(0.025, mu, sigma):.4f}")
print(f"Percentil 97.5 %: {stats.norm.ppf(0.975, mu, sigma):.4f}")
"""
        show_code(code_norm, "b_norm")

    with sub2:
        st.markdown("### Distribución Beta — prior para proporciones")
        st.latex(r"f(p \mid \alpha, \beta) = \frac{p^{\alpha-1}(1-p)^{\beta-1}}{B(\alpha,\beta)}")
        st.markdown("""
        | Variable | Significado |
        |----------|-------------|
        | $p$ | Proporción a estimar (entre 0 y 1) |
        | $\\alpha$ | "Éxitos previos" — mueve la masa hacia 1 |
        | $\\beta$ | "Fracasos previos" — mueve la masa hacia 0 |
        | $B(\\alpha,\\beta)$ | Función Beta: constante normalizadora |
        """)
        interp("La Beta es el <strong>prior conjugado</strong> de la Binomial. Si prior = Beta(α,β) "
               "y observas k éxitos en n intentos → posterior = Beta(α+k, β+n−k). Sin integrales.")
        c1, c2 = st.columns(2)
        alpha_b = c1.slider("α", 0.1, 20.0, 2.0, 0.1, key="ab2")
        beta_b  = c2.slider("β", 0.1, 20.0, 2.0, 0.1, key="bb2")
        x     = np.linspace(0.001, 0.999, 500)
        y     = stats.beta.pdf(x, alpha_b, beta_b)
        media = alpha_b / (alpha_b + beta_b)
        moda  = (alpha_b-1)/(alpha_b+beta_b-2) if alpha_b > 1 and beta_b > 1 else None
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(x, y, color="#d2a8ff", lw=2.5)
        ax.fill_between(x, y, alpha=0.15, color="#d2a8ff")
        ax.axvline(media, color="#3fb950", lw=1.8, linestyle="--", label=f"Media = {media:.3f}")
        if moda:
            ax.axvline(moda, color="#f78166", lw=1.5, linestyle=":", label=f"Moda = {moda:.3f}")
        ax.set_title(f"Beta(α={alpha_b}, β={beta_b})")
        ax.set_xlabel("p")
        ax.legend(fontsize=8)
        ax.grid(True)
        st.pyplot(fig, use_container_width=True)
        forma = ("uniforme" if abs(alpha_b-1)<0.15 and abs(beta_b-1)<0.15
                 else "sesgada a la derecha" if alpha_b > beta_b
                 else "sesgada a la izquierda" if beta_b > alpha_b
                 else "simétrica")
        graph_interp("Distribución Beta",
            f"La <strong>curva morada</strong> representa creencias sobre una proporción. "
            f"Con α={alpha_b:.1f} y β={beta_b:.1f} la distribución es <strong>{forma}</strong>. "
            f"La <strong>línea verde</strong> marca la media = {media:.3f}: mejor estimado puntual. "
            "Con α = β = 1 la curva es plana (total incertidumbre). "
            "Cuanto mayores sean α y β, más estrecha y concentrada es la curva (más certeza).")
        code_beta = """\
import numpy as np
from scipy import stats

#Parámetros de la Beta 
alpha = 2.0   # éxitos previos: desplaza la masa hacia la derecha (hacia 1)
beta  = 2.0   # fracasos previos: desplaza la masa hacia la izquierda (hacia 0)

#Estadísticas 
media    = alpha / (alpha + beta)
varianza = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
moda     = (alpha - 1) / (alpha + beta - 2) if alpha > 1 and beta > 1 else "no definida"

print(f"Media    = {media:.4f}  (estimado puntual de la proporción)")
print(f"Varianza = {varianza:.4f}  (incertidumbre)")
print(f"Moda     = {moda}")

#Intervalo de credibilidad al 95 % 
#"Con 95 % de probabilidad, la proporción está en este rango"
ic_bajo = stats.beta.ppf(0.025, alpha, beta)   # percentil 2.5 %
ic_alto = stats.beta.ppf(0.975, alpha, beta)   # percentil 97.5 %
print(f"IC 95 %  = [{ic_bajo:.4f}, {ic_alto:.4f}]")

#Actualización conjugada Beta-Binomial 
#Escenario: observamos k=7 éxitos en n=10 intentos
k, n = 7, 10
#La actualización es solo sumar: alpha += éxitos, beta += fracasos
alpha_post = alpha + k
beta_post  = beta  + (n - k)
media_post = alpha_post / (alpha_post + beta_post)
print(f"\\nTras {k} éxitos en {n} intentos:")
print(f"Posterior = Beta({alpha_post:.1f}, {beta_post:.1f})  → media = {media_post:.4f}")
"""
        show_code(code_beta, "b_beta")

    with sub3:
        col_g, col_p = st.columns(2)
        with col_g:
            st.markdown("### Gamma — prior para tasas")
            st.latex(r"f(x\mid k,\theta)=\frac{x^{k-1}e^{-x/\theta}}{\theta^k \Gamma(k)}")
            st.markdown("""
            | Variable | Significado |
            |----------|-------------|
            | $k$ | Forma: controla si hay pico o la curva decrece |
            | $\\theta$ | Escala: estira los valores hacia la derecha |
            """)
            k_g = st.slider("k (forma)", 0.5, 10.0, 2.0, 0.5, key="kg2")
            t_g = st.slider("θ (escala)", 0.1, 5.0, 1.0, 0.1, key="tg2")
            x  = np.linspace(0.01, 30, 500)
            y  = stats.gamma.pdf(x, a=k_g, scale=t_g)
            mg = k_g * t_g
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.plot(x, y, color="#ffa657", lw=2.2)
            ax.fill_between(x, y, alpha=0.15, color="#ffa657")
            ax.axvline(mg, color="#3fb950", lw=1.5, linestyle="--", label=f"Media = {mg:.2f}")
            ax.set_title(f"Gamma(k={k_g}, θ={t_g})")
            ax.legend(fontsize=8)
            ax.grid(True)
            st.pyplot(fig, use_container_width=True)
            graph_interp("Gamma",
                f"Modela valores <strong>positivos continuos</strong> como tasas o tiempos de espera. "
                f"Media = k·θ = {mg:.2f}. "
                "Si k < 1 la curva decrece desde el cero (muchos valores pequeños). "
                "Si k > 1 aparece un pico. Aumentar θ estira la distribución hacia valores mayores.")

        with col_p:
            st.markdown("### Poisson — conteo de eventos")
            st.latex(r"P(X=k \mid \lambda) = \frac{e^{-\lambda}\lambda^k}{k!}")
            st.markdown("""
            | Variable | Significado |
            |----------|-------------|
            | $\\lambda$ | Tasa promedio de eventos por unidad de tiempo |
            | $k$ | Número de eventos observados |
            """)
            lam = st.slider("λ (tasa)", 0.5, 20.0, 3.0, 0.5, key="lp2")
            k_v = np.arange(0, int(lam*3)+1)
            pr  = stats.poisson.pmf(k_v, lam)
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.bar(k_v, pr, color="#3fb950", alpha=0.8, edgecolor="#21262d")
            ax.axvline(lam, color="#f78166", lw=1.8, linestyle="--", label=f"λ = {lam}")
            ax.set_title(f"Poisson(λ={lam})")
            ax.legend(fontsize=8)
            ax.grid(True, axis="y")
            st.pyplot(fig, use_container_width=True)
            graph_interp("Poisson",
                f"Cada barra es la probabilidad de observar exactamente ese número de eventos. "
                f"La <strong>línea roja</strong> marca λ = {lam}: el pico de la distribución. "
                "Con λ pequeño el histograma está pegado al cero. "
                "Con λ grande se vuelve casi simétrico y se parece a una Normal.")

        interp("La <strong>Gamma es el prior conjugado de la Poisson</strong>: "
               "Prior Gamma(α,β) + observar n eventos en tiempo t → Posterior Gamma(α+n, β+t).")

        code_gp = """\
import numpy as np
from scipy import stats

#Distribución Gamma 
k_forma, theta = 2.0, 1.0

#Media y varianza de la Gamma
media_g    = k_forma * theta
varianza_g = k_forma * theta**2
print(f"Gamma(k={k_forma}, θ={theta})")
print(f"  Media    = {media_g:.2f}")
print(f"  Varianza = {varianza_g:.2f}")

#Distribución Poisson 
lam = 3.0   # λ: eventos esperados por unidad de tiempo
print(f"\\nPoisson(λ={lam})")

#P(X = k): probabilidad de observar exactamente k eventos
for k in range(8):
    p = stats.poisson.pmf(k, lam)
    print(f"  P(X={k}) = {p:.4f}")

#Actualización conjugada Gamma-Poisson 
#Prior Gamma(α, β): representa nuestra creencia inicial sobre λ
alpha_prior, beta_prior = 2.0, 1.0
# Observamos n=15 eventos en un periodo t=5
n_eventos, t = 15, 5

#La actualización es simplemente sumar:
alpha_post = alpha_prior + n_eventos   # suma los eventos observados
beta_post  = beta_prior  + t           # suma el tiempo transcurrido
media_post = alpha_post / beta_post    # nueva estimación de λ

print(f"\\nPosterior de λ:")
print(f"  Gamma({alpha_post}, {beta_post})  →  media = {media_post:.4f}")
"""
        show_code(code_gp, "b_gp")

    with sub4:
        st.markdown("### Distribución Binomial")
        st.latex(r"P(X=k \mid n,p) = \binom{n}{k} p^k (1-p)^{n-k}")
        st.markdown("""
        | Variable | Significado |
        |----------|-------------|
        | $n$ | Número de ensayos |
        | $k$ | Número de éxitos observados |
        | $p$ | Probabilidad de éxito en cada intento |
        """)
        c1, c2 = st.columns(2)
        n_bin = c1.slider("n (intentos)", 5, 100, 20, key="nb2")
        p_bin = c2.slider("p (éxito)", 0.01, 0.99, 0.5, 0.01, key="pb2")
        k_v   = np.arange(0, n_bin+1)
        pr    = stats.binom.pmf(k_v, n_bin, p_bin)
        mu_b  = n_bin * p_bin
        sd_b  = np.sqrt(n_bin * p_bin * (1-p_bin))
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.bar(k_v, pr, color="#58a6ff", alpha=0.8, edgecolor="#21262d")
        ax.axvline(mu_b, color="#3fb950", lw=2, linestyle="--", label=f"E[X] = {mu_b:.1f}")
        ax.axvspan(mu_b - sd_b, mu_b + sd_b, alpha=0.1, color="#3fb950",
                   label=f"±σ = [{mu_b-sd_b:.1f}, {mu_b+sd_b:.1f}]")
        ax.set_title(f"Binomial(n={n_bin}, p={p_bin})")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y")
        st.pyplot(fig, use_container_width=True)
        graph_interp("Distribución Binomial",
            f"Cada barra es la probabilidad de obtener exactamente ese número de éxitos en {n_bin} intentos. "
            f"La <strong>línea verde</strong> es la media E[X] = n·p = {mu_b:.1f}: el resultado más probable. "
            f"La <strong>franja verde</strong> es ±σ: rango donde cae la mayoría de resultados. "
            f"Con p = {p_bin:.2f} {'la distribución es casi simétrica (p cercano a 0.5)' if 0.4 < p_bin < 0.6 else 'la distribución está sesgada (p lejos de 0.5)'}.")
        code_binom = """\
import numpy as np
from scipy import stats

#Parámetros 
n = 20    # número de intentos (ej. 20 lanzamientos de moneda)
p = 0.5   # probabilidad de éxito en cada intento

#Estadísticas 
media    = n * p                  # valor esperado de X
varianza = n * p * (1 - p)        # dispersión de la distribución
std      = np.sqrt(varianza)

print(f"Binomial(n={n}, p={p})")
print(f"  Media    = {media}")
print(f"  Std      = {std:.4f}")

#Probabilidades puntuales y acumuladas 
print(f"  P(X = 10) = {stats.binom.pmf(10, n, p):.4f}  # exactamente 10 éxitos")
print(f"  P(X ≤ 10) = {stats.binom.cdf(10, n, p):.4f}  # 10 o menos éxitos")
print(f"  P(X ≥ 15) = {1 - stats.binom.cdf(14, n, p):.4f}  # 15 o más éxitos")

#Intervalo de confianza del 95 % 
# Valores de k que cubren el 95 % central de la distribución
ic_bajo = stats.binom.ppf(0.025, n, p)   # percentil 2.5 %
ic_alto = stats.binom.ppf(0.975, n, p)   # percentil 97.5 %
print(f"  IC 95 %   = [{ic_bajo:.0f}, {ic_alto:.0f}]")
"""
        show_code(code_binom, "b_binom")

# 
# TAB 3 — INFERENCIA BAYESIANA
# 
with tab_inf:
    st.markdown("## Inferencia Bayesiana — Prior → Posterior")
    interp("La inferencia bayesiana <strong>combina lo que sabíamos (prior) con los datos observados</strong> "
           "para obtener una creencia actualizada (posterior). A más datos, más domina la evidencia.")

    st.markdown("### Modelo Beta-Binomial conjugado")
    st.latex(r"""
    \underbrace{Beta(\alpha,\beta)}_{\text{prior}}
    \;\xrightarrow{\text{k éxitos en n intentos}}\;
    \underbrace{Beta(\alpha + k,\; \beta + (n-k))}_{\text{posterior}}
    """)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**Prior (creencia inicial)**")
        alpha_prior = st.slider("α prior", 0.1, 10.0, 1.0, 0.1, key="ap3")
        beta_prior  = st.slider("β prior", 0.1, 10.0, 1.0, 0.1, key="bp3")
        st.markdown("**Datos observados**")
        n_obs = st.slider("n (intentos)", 1, 200, 20, key="n3")
        k_obs = st.slider("k (éxitos)", 0, n_obs, 12, key="k3")

    alpha_post = alpha_prior + k_obs
    beta_post  = beta_prior  + (n_obs - k_obs)
    media_prior = alpha_prior / (alpha_prior + beta_prior)
    media_post  = alpha_post  / (alpha_post  + beta_post)
    mle         = k_obs / n_obs if n_obs > 0 else 0.0

    with col2:
        st.markdown(f"""
        <div class="card">
        <div class="section-label">resultados</div>
        <p>Prior   : <span class="result-value">Beta({alpha_prior:.1f}, {beta_prior:.1f})</span> → media = {media_prior:.3f}</p>
        <p>Datos   : <span class="result-value">{k_obs} éxitos en {n_obs} intentos</span> → MLE = {mle:.3f}</p>
        <p>Posterior: <span class="result-value">Beta({alpha_post:.1f}, {beta_post:.1f})</span> → media = {media_post:.3f}</p>
        </div>
        """, unsafe_allow_html=True)

    x   = np.linspace(0.001, 0.999, 500)
    lik = stats.binom.pmf(k_obs, n_obs, x)
    lik_s = lik / lik.max() * stats.beta.pdf(x, alpha_post, beta_post).max() * 0.9

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, stats.beta.pdf(x, alpha_prior, beta_prior),
            color="#8b949e", lw=2, linestyle="--",
            label=f"Prior  Beta({alpha_prior:.1f},{beta_prior:.1f})")
    ax.plot(x, lik_s, color="#3fb950", lw=1.8, linestyle=":",
            label="Verosimilitud (escalada)")
    ax.plot(x, stats.beta.pdf(x, alpha_post, beta_post),
            color="#58a6ff", lw=2.8,
            label=f"Posterior  Beta({alpha_post:.1f},{beta_post:.1f})")
    ax.axvline(media_prior, color="#8b949e", lw=1.0, linestyle="--", alpha=0.6)
    ax.axvline(mle,         color="#3fb950", lw=1.0, linestyle=":",  alpha=0.7)
    ax.axvline(media_post,  color="#58a6ff", lw=1.8, linestyle="--", alpha=0.9)
    ax.set_xlabel("p (proporción)")
    ax.set_ylabel("Densidad")
    ax.set_title("Actualización bayesiana: Prior → Posterior")
    ax.legend(fontsize=9)
    ax.grid(True)
    st.pyplot(fig, use_container_width=True)

    graph_interp("Prior → Posterior",
        f"La <strong>curva gris punteada</strong> es el prior: creencia antes de datos (media = {media_prior:.3f}). "
        f"La <strong>curva verde punteada</strong> es la verosimilitud: lo que dicen los datos solos (MLE = {mle:.3f}). "
        f"La <strong>curva azul</strong> es el posterior: combinación de ambos (media = {media_post:.3f}). "
        "El posterior siempre está entre el prior y la verosimilitud. "
        f"{'Con muchos datos (' + str(n_obs) + ' intentos), el posterior se acerca al MLE: los datos dominan.' if n_obs > 30 else 'Con pocos datos (' + str(n_obs) + ' intentos), el prior todavía influye bastante en el posterior.'}")

    st.markdown("### Código — actualización Beta-Binomial")
    code_inf = f"""\
import numpy as np
from scipy import stats

#Prior: Beta(α, β) 
#Interpretación: como si hubiéramos visto α-1 éxitos y β-1 fracasos antes
alpha_prior = {alpha_prior}   # cuanto mayor, más confianza inicial en p alto
beta_prior  = {beta_prior}    # cuanto mayor, más confianza inicial en p bajo

#Datos observados 
n_obs = {n_obs}   # total de intentos
k_obs = {k_obs}   # número de éxitos observados

#Actualización conjugada 
#La regla es simplemente SUMAR: no hace falta calcular integrales
alpha_post = alpha_prior + k_obs           # prior éxitos + éxitos observados
beta_post  = beta_prior  + (n_obs - k_obs) # prior fracasos + fracasos observados

#Estadísticas del posterior 
media_post = alpha_post / (alpha_post + beta_post)
mle        = k_obs / n_obs   # estimación frecuentista pura (sin prior)

print(f"Prior     : Beta({alpha_prior}, {beta_prior})  →  media = {{alpha_prior/(alpha_prior+beta_prior):.4f}}")
print(f"MLE       : {k_obs}/{n_obs} = {{mle:.4f}}   (solo datos, sin prior)")
print(f"Posterior : Beta({{alpha_post}}, {{beta_post}})  →  media = {{media_post:.4f}}")

#Intervalo de credibilidad al 95 % 
#A diferencia del IC frecuentista, este tiene interpretación directa:
#"Con 95 % de probabilidad, la proporción real está en este rango"
ic_bajo = stats.beta.ppf(0.025, alpha_post, beta_post)
ic_alto = stats.beta.ppf(0.975, alpha_post, beta_post)
print(f"IC 95 %   : [{{ic_bajo:.4f}}, {{ic_alto:.4f}}]")
"""
    show_code(code_inf, "b_inf")

# 
# TAB 4 — MCMC
# 
with tab_mcmc:
    st.markdown("## MCMC — Metropolis-Hastings")
    interp("Cuando la posterior <strong>no tiene forma cerrada</strong>, necesitamos muestrearla. "
           "MCMC construye una cadena de Markov cuya distribución estacionaria es la posterior. "
           "No necesitamos calcularla directamente, solo evaluarla punto a punto.")

    st.markdown("### Algoritmo paso a paso")
    st.markdown("""
    <div class="card">
    <ol style="line-height:2.4;font-family:'IBM Plex Mono',monospace;font-size:0.88rem">
    <li>Empieza en un valor inicial θ₀ (cualquiera)</li>
    <li>Propón un candidato θ* ~ Normal(θ_actual, σ_propuesta)</li>
    <li>Calcula la razón: <strong>r = posterior(θ*) / posterior(θ_actual)</strong></li>
    <li>Acepta θ* con probabilidad min(1, r)  →  si mejora siempre acepta; si empeora acepta con prob r</li>
    <li>Repite miles de veces → colección de valores aceptados ≈ muestra de la posterior</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

    st.latex(r"\alpha = \min\!\left(1,\; \frac{p(\theta^* \mid D)}{p(\theta_t \mid D)}\right)")
    st.markdown("""
    | Símbolo | Significado |
    |---------|-------------|
    | $\\alpha$ | Probabilidad de aceptar el salto al candidato |
    | $\\theta^*$ | Valor candidato propuesto |
    | $\\theta_t$ | Valor actual en la iteración $t$ |
    | $p(\\theta \\mid D)$ | Posterior ∝ prior × verosimilitud |
    """)

    st.markdown("---")
    col1, col2 = st.columns([1, 2])
    with col1:
        n_iter  = st.slider("Iteraciones", 500, 10000, 3000, 500, key="ni4")
        sigma_p = st.slider("σ propuesta", 0.05, 3.0, 0.5, 0.05, key="sp4")
        mu_real = st.slider("μ real (oculto)", -3.0, 3.0, 1.5, 0.1, key="mr4")
        n_datos = st.slider("Cantidad de datos", 5, 100, 20, key="nd4")
        seed4   = st.number_input("Semilla", 0, 9999, 42, key="s4")

    np.random.seed(int(seed4))
    datos = np.random.normal(mu_real, 1.0, n_datos)

    # Posterior analítica (conjugada, σ=1 conocida)
    mu_pr, sig_pr = 0.0, 2.0
    var_post = 1.0 / (1.0/sig_pr**2 + n_datos)
    mu_post  = var_post * (mu_pr/sig_pr**2 + datos.sum())
    sig_post = np.sqrt(var_post)

    def log_post(theta, datos, mu0=0.0, s0=2.0, sd=1.0):
        return np.sum(stats.norm.logpdf(datos, theta, sd)) + stats.norm.logpdf(theta, mu0, s0)

    theta_c = 0.0
    cadena  = [theta_c]
    accept  = 0
    np.random.seed(int(seed4)+1)
    for _ in range(n_iter):
        prop  = np.random.normal(theta_c, sigma_p)
        log_r = log_post(prop, datos) - log_post(theta_c, datos)
        if np.log(np.random.uniform()) < log_r:
            theta_c = prop
            accept += 1
        cadena.append(theta_c)

    cadena  = np.array(cadena)
    burnin  = int(n_iter * 0.2)
    muestra = cadena[burnin:]
    tasa    = accept / n_iter

    with col2:
        c1m, c2m, c3m = st.columns(3)
        c1m.metric("Media MCMC",      f"{muestra.mean():.4f}")
        c2m.metric("Media analítica", f"{mu_post:.4f}")
        c3m.metric("Tasa aceptación", f"{tasa:.1%}")
        if tasa < 0.15:
            nota("Tasa muy baja → σ propuesta demasiado grande. La cadena propone saltos enormes y casi siempre rechaza.")
        elif tasa > 0.70:
            nota("Tasa muy alta → σ propuesta muy pequeña. La cadena se mueve muy lentamente.")
        else:
            interp(f"Tasa {tasa:.1%} — en rango saludable. Óptimo teórico: ≈ 23–44 %.")

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))

    axes[0].plot(cadena, color="#58a6ff", lw=0.5, alpha=0.8)
    axes[0].axvline(burnin, color="#f78166", lw=1.8, linestyle="--", label="Fin burn-in")
    axes[0].axhline(mu_real, color="#3fb950", lw=1.2, linestyle=":", label=f"μ real={mu_real}")
    axes[0].set_title("Traceplot")
    axes[0].set_xlabel("Iteración")
    axes[0].set_ylabel("θ")
    axes[0].legend(fontsize=8)
    axes[0].grid(True)

    xr = np.linspace(muestra.min()-1, muestra.max()+1, 300)
    axes[1].hist(muestra, bins=50, density=True, color="#58a6ff",
                 alpha=0.5, edgecolor="none", label="MCMC")
    axes[1].plot(xr, stats.norm.pdf(xr, mu_post, sig_post),
                 color="#3fb950", lw=2.2, label="Analítica")
    axes[1].axvline(mu_real, color="#f78166", lw=2, linestyle="--",
                    label=f"μ real={mu_real}")
    axes[1].set_title("Posterior: MCMC vs analítica")
    axes[1].set_xlabel("θ")
    axes[1].legend(fontsize=8)
    axes[1].grid(True)

    max_lag = min(50, len(muestra)//2)
    lags    = np.arange(max_lag)
    acf     = [1.0 if l==0 else np.corrcoef(muestra[:-l], muestra[l:])[0,1]
               for l in range(max_lag)]
    umbral  = 1.96 / np.sqrt(len(muestra))
    axes[2].bar(lags, acf, color="#d2a8ff", alpha=0.8, width=0.8)
    axes[2].axhline(0,       color="#8b949e", lw=1)
    axes[2].axhline( umbral, color="#3fb950", lw=1.2, linestyle="--",
                     label=f"±{umbral:.3f}")
    axes[2].axhline(-umbral, color="#3fb950", lw=1.2, linestyle="--")
    axes[2].set_title("Autocorrelación (ACF)")
    axes[2].set_xlabel("Lag")
    axes[2].legend(fontsize=8)
    axes[2].grid(True)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    graph_interp("Traceplot (izquierda)",
        "Muestra el recorrido de la cadena a lo largo de las iteraciones. "
        "Un traceplot <strong>saludable</strong> parece ruido blanco: oscila rápidamente sin quedar pegado. "
        f"La <strong>línea roja</strong> marca el fin del burn-in ({burnin} iteraciones descartadas). "
        "Si la cadena tarda en moverse o queda atrapada en un valor, la mezcla es mala y necesitas ajustar σ propuesta.")
    graph_interp("Posterior MCMC vs analítica (centro)",
        f"El <strong>histograma azul</strong> es la distribución de las muestras MCMC. "
        f"La <strong>curva verde</strong> es la solución analítica exacta. "
        "Si se superponen bien, el MCMC funcionó correctamente. "
        f"El <strong>μ real = {mu_real}</strong> (línea roja) debería caer cerca del pico de ambas. "
        f"La diferencia entre MCMC ({muestra.mean():.3f}) y analítica ({mu_post:.3f}) es el error de simulación.")
    graph_interp("Autocorrelación ACF (derecha)",
        "Mide la correlación entre muestras separadas por 'lag' pasos. "
        "<strong>Cae rápido a cero</strong> → muestras casi independientes → buena mezcla. "
        f"Las <strong>líneas verdes</strong> marcan el umbral de significancia (±{umbral:.3f}). "
        "Barras altas en lags grandes indican que la cadena se mueve lentamente: las muestras son muy similares entre sí.")

    st.markdown("### Código — Metropolis-Hastings completo")
    code_mcmc = """\
import numpy as np
from scipy import stats

np.random.seed(42)

#Datos simulados: μ real = 1.5, σ = 1 
datos = np.random.normal(1.5, 1.0, 20)

#Función de log-posterior 
#Usamos logaritmos para estabilidad numérica (evitar underflow)
#log P(θ|D) ∝ log P(D|θ) + log P(θ)
def log_posterior(theta, datos):
    # Log-verosimilitud: P(datos | θ) asumiendo Normal con σ=1 conocida
    log_vero  = np.sum(stats.norm.logpdf(datos, theta, 1.0))
    # Log-prior: creencia inicial θ ~ N(0, 2)  (prior difuso)
    log_prior = stats.norm.logpdf(theta, 0.0, 2.0)
    return log_vero + log_prior

#Algoritmo Metropolis-Hastings 
n_iter       = 3000
sigma_prop   = 0.5    # tamaño de cada salto propuesto
theta_actual = 0.0    # punto de inicio arbitrario
cadena       = [theta_actual]
aceptados    = 0

for _ in range(n_iter):
    # 1. Proponer un candidato cerca del valor actual
    propuesta = np.random.normal(theta_actual, sigma_prop)

    # 2. Calcular log-razón de aceptación
    #    Si log_r >= 0 (propuesta mejor): aceptar siempre
    #    Si log_r < 0  (propuesta peor): aceptar con probabilidad exp(log_r)
    log_r = log_posterior(propuesta, datos) - log_posterior(theta_actual, datos)

    # 3. Aceptar/rechazar usando truco log-uniforme
    if np.log(np.random.uniform()) < log_r:
        theta_actual = propuesta   # se acepta: avanzar
        aceptados   += 1
    # Si se rechaza: theta_actual se repite en la cadena

    cadena.append(theta_actual)

#Diagnóstico post-muestreo 
cadena  = np.array(cadena)
burnin  = 600      # descartar el 20 % inicial (fase de calentamiento)
muestra = cadena[burnin:]

print(f"Tasa de aceptación  : {aceptados/n_iter:.2%}  (ideal: 23-44 %)")
print(f"Media posterior     : {muestra.mean():.4f}  (real: 1.5)")
print(f"Desv. estándar post : {muestra.std():.4f}")
print(f"IC 95 %             : [{np.percentile(muestra,2.5):.4f}, {np.percentile(muestra,97.5):.4f}]")
"""
    show_code(code_mcmc, "b_mcmc")

# 
# TAB 5 — REGRESIÓN BAYESIANA
# 
with tab_reg:
    st.markdown("## Regresión Lineal Bayesiana")
    interp("En regresión frecuentista obtenemos <em>un único valor</em> para pendiente e intercepto. "
           "En la versión bayesiana obtenemos una <strong>distribución</strong> sobre esos parámetros: "
           "incertidumbre cuantificada sobre toda la recta.")

    st.markdown("### Modelo")
    st.latex(r"y_i = \beta_0 + \beta_1 x_i + \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0,\sigma^2)")
    st.latex(r"\beta_0 \sim \mathcal{N}(\mu_0, \tau_0^2), \quad \beta_1 \sim \mathcal{N}(\mu_1, \tau_1^2)")
    st.markdown("""
    | Símbolo | Significado |
    |---------|-------------|
    | $\\beta_0$ | Intercepto: valor de $y$ cuando $x = 0$ |
    | $\\beta_1$ | Pendiente: cambio en $y$ por cada unidad de $x$ |
    | $\\sigma^2$ | Varianza del ruido (incertidumbre irreducible) |
    | $\\mu_0, \\tau_0$ | Hiperparámetros del prior sobre $\\beta_0$ |
    | $\\mu_1, \\tau_1$ | Hiperparámetros del prior sobre $\\beta_1$ |
    """)

    c1, c2 = st.columns([1, 2])
    with c1:
        b0_r  = st.slider("β₀ real", -5.0, 5.0, 1.0, 0.5, key="b0r")
        b1_r  = st.slider("β₁ real", -3.0, 3.0, 2.0, 0.1, key="b1r")
        sig_r = st.slider("σ ruido", 0.1, 3.0, 0.8, 0.1, key="sr5")
        n_pts = st.slider("Puntos", 10, 100, 30, key="np5")
        n_lin = st.slider("Líneas posteriores", 20, 200, 80, 10, key="nl5")
        sd5   = st.number_input("Semilla", 0, 9999, 7, key="s5")

    np.random.seed(int(sd5))
    x_d  = np.linspace(0, 5, n_pts)
    y_d  = b0_r + b1_r * x_d + np.random.normal(0, sig_r, n_pts)

    X       = np.column_stack([np.ones(n_pts), x_d])
    s2      = sig_r**2
    pri_inv = np.diag([0.1, 0.1])
    pri_m   = np.array([0.0, 0.0])
    p_cov   = np.linalg.inv(pri_inv + X.T @ X / s2)
    p_mean  = p_cov @ (pri_inv @ pri_m + X.T @ y_d / s2)

    with c2:
        st.markdown(f"""
        <div class="card">
        <div class="section-label">estimaciones posteriores</div>
        <p>β₀: <span class="result-value">{p_mean[0]:.3f} ± {np.sqrt(p_cov[0,0]):.3f}</span>  (real: {b0_r})</p>
        <p>β₁: <span class="result-value">{p_mean[1]:.3f} ± {np.sqrt(p_cov[1,1]):.3f}</span>  (real: {b1_r})</p>
        </div>
        """, unsafe_allow_html=True)

    samples = np.random.multivariate_normal(p_mean, p_cov, n_lin)
    xp      = np.linspace(x_d.min()-0.3, x_d.max()+0.3, 200)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for s in samples:
        axes[0].plot(xp, s[0]+s[1]*xp, color="#58a6ff", alpha=0.06, lw=1)
    axes[0].scatter(x_d, y_d, color="#e6edf3", s=25, zorder=5, label="Datos")
    axes[0].plot(xp, b0_r+b1_r*xp, color="#f78166", lw=2, linestyle="--", label="Verdadera")
    axes[0].plot(xp, p_mean[0]+p_mean[1]*xp, color="#3fb950", lw=2.2, label="Media posterior")
    axes[0].set_title("Incertidumbre en la recta")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend(fontsize=8)
    axes[0].grid(True)

    b0r = np.linspace(p_mean[0]-3*np.sqrt(p_cov[0,0]), p_mean[0]+3*np.sqrt(p_cov[0,0]), 200)
    b1r = np.linspace(p_mean[1]-3*np.sqrt(p_cov[1,1]), p_mean[1]+3*np.sqrt(p_cov[1,1]), 200)
    B0, B1 = np.meshgrid(b0r, b1r)
    Z  = stats.multivariate_normal(p_mean, p_cov).pdf(np.dstack((B0,B1)))
    cp = axes[1].contourf(B0, B1, Z, levels=25, cmap="Blues")
    axes[1].scatter(*p_mean, color="#3fb950", s=80, zorder=5, label="Media post.")
    axes[1].scatter([b0_r], [b1_r], color="#f78166", s=100, marker="*",
                    zorder=5, label="Valores reales")
    axes[1].set_xlabel("β₀ (intercepto)")
    axes[1].set_ylabel("β₁ (pendiente)")
    axes[1].set_title("Posterior conjunta β₀, β₁")
    axes[1].legend(fontsize=8)
    plt.colorbar(cp, ax=axes[1])
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    graph_interp("Incertidumbre en la recta (izquierda)",
        f"Cada <strong>línea azul transparente</strong> es una recta posible muestreada de la posterior ({n_lin} rectas). "
        "Donde las líneas se juntan hay mayor certeza; donde se abren, mayor incertidumbre. "
        "La <strong>recta verde</strong> es la mejor estimación (media posterior). "
        "La <strong>recta roja</strong> es la verdad oculta. "
        "Este abanico de rectas es la ventaja bayesiana: no solo una respuesta, sino toda la incertidumbre.")
    graph_interp("Posterior conjunta (derecha)",
        "El mapa de calor muestra la densidad sobre el espacio (β₀, β₁). "
        "<strong>Zonas oscuras</strong> = combinaciones más probables. "
        f"El <strong>punto verde</strong> es la media posterior ({p_mean[0]:.2f}, {p_mean[1]:.2f}). "
        f"El <strong>asterisco rojo</strong> es el valor real ({b0_r}, {b1_r}). "
        "La forma elíptica revela la correlación entre intercepto y pendiente: si aumentas uno, el otro cambia.")

    st.markdown("### Código")
    code_reg = """\
import numpy as np
from scipy import stats

np.random.seed(7)

#Datos simulados: β₀=1, β₁=2, σ=0.8 
x     = np.linspace(0, 5, 30)
y     = 1.0 + 2.0 * x + np.random.normal(0, 0.8, 30)

#Matriz de diseño 
#Primera columna de unos → intercepto; segunda columna → pendiente
X      = np.column_stack([np.ones(30), x])
sigma2 = 0.64   # σ² = 0.8² = 0.64  (asumida conocida)

#Prior: β ~ N(0, 10·I)  (prior difuso, poco informativo) 
prior_mean    = np.array([0.0, 0.0])
prior_cov_inv = np.diag([0.1, 0.1])   # inversa de diag(10, 10)

#Posterior analítico 
#Fórmula del modelo lineal gaussiano con prior normal:
#Σ_post = (Σ_prior⁻¹ + Xᵀ X / σ²)⁻¹
post_cov  = np.linalg.inv(prior_cov_inv + X.T @ X / sigma2)
#μ_post  = Σ_post · (Σ_prior⁻¹ · μ_prior + Xᵀ y / σ²)
post_mean = post_cov @ (prior_cov_inv @ prior_mean + X.T @ y / sigma2)

#Resultados 
print(f"β₀ posterior: {post_mean[0]:.4f}  ±  {np.sqrt(post_cov[0,0]):.4f}")
print(f"β₁ posterior: {post_mean[1]:.4f}  ±  {np.sqrt(post_cov[1,1]):.4f}")

#Muestrear rectas de la posterior 
#Cada fila es una recta posible: [β₀_i, β₁_i]
muestras = np.random.multivariate_normal(post_mean, post_cov, 5)
print("\\nEjemplos de rectas muestreadas:")
for i, (b0, b1) in enumerate(muestras):
    print(f"  Recta {i+1}: y = {b0:.3f} + {b1:.3f}·x")
"""
    show_code(code_reg, "b_reg")

# 
# TAB 6 — SELECCIÓN DE MODELOS
# 
with tab_sel:
    st.markdown("## Selección de Modelos")
    interp("Con varios modelos candidatos necesitamos un criterio para elegir el mejor. "
           "Los métodos bayesianos <strong>equilibran ajuste y complejidad</strong>, "
           "penalizando modelos que usan parámetros innecesarios.")

    sub_bic, sub_bf = st.tabs(["BIC", "Factor de Bayes"])

    with sub_bic:
        st.markdown("### BIC — Criterio de Información Bayesiana")
        st.latex(r"\text{BIC} = k \ln(n) - 2\ln(\hat{L})")
        st.markdown("""
        | Símbolo | Significado |
        |---------|-------------|
        | $k$ | Número de parámetros del modelo |
        | $n$ | Número de observaciones |
        | $\\hat{L}$ | Máxima verosimilitud alcanzada |
        | $k\\ln(n)$ | Penalización por complejidad (crece con los datos) |
        | $-2\\ln(\\hat{L})$ | Medida de mal ajuste: menor = mejor ajuste |
        """)
        interp("El BIC es una competencia: <strong>mejor ajuste vs menos parámetros</strong>. "
               "Gana el modelo con el BIC más bajo. "
               "La penalización $k\\ln(n)$ crece con los datos: a más datos, más estricto contra la complejidad.")

        c1, c2 = st.columns(2)
        grado_r  = c1.slider("Grado real", 1, 5, 2, key="gr6")
        n_d      = c2.slider("Puntos", 10, 100, 30, key="nd6")
        sig_bic  = c1.slider("σ ruido", 0.1, 3.0, 0.5, 0.1, key="sbic")
        sd6      = c2.number_input("Semilla", 0, 9999, 99, key="s6")

        np.random.seed(int(sd6))
        xb = np.linspace(-2, 2, n_d)
        cr = np.random.randn(grado_r+1) * 0.5
        yb = sum(cr[i]*xb**i for i in range(grado_r+1)) + np.random.normal(0, sig_bic, n_d)

        bics   = []
        grados = list(range(1,9))
        for g in grados:
            cf   = np.polyfit(xb, yb, g)
            yhat = np.polyval(cf, xb)
            sse  = np.sum((yb-yhat)**2)
            s2   = sse/n_d
            ll   = -n_d/2*np.log(2*np.pi*s2) - sse/(2*s2)
            bics.append((g+1)*np.log(n_d) - 2*ll)

        mejor_g = grados[np.argmin(bics)]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].bar(grados, bics,
                    color=["#3fb950" if g==mejor_g else "#58a6ff" for g in grados],
                    alpha=0.85, edgecolor="#21262d")
        axes[0].axvline(grado_r, color="#f78166", lw=2, linestyle="--",
                        label=f"Grado real = {grado_r}")
        axes[0].set_xlabel("Grado del polinomio")
        axes[0].set_ylabel("BIC")
        axes[0].set_title("BIC por grado (verde = ganador)")
        axes[0].legend()
        axes[0].grid(True, axis="y")

        xf = np.linspace(-2.2, 2.2, 300)
        axes[1].scatter(xb, yb, color="#e6edf3", s=25, label="Datos", zorder=5)
        for g, col, lab in [(mejor_g, "#3fb950", f"Mejor BIC (g={mejor_g})"),
                            (grado_r, "#f78166", f"Grado real ({grado_r})")]:
            c = np.polyfit(xb, yb, g)
            axes[1].plot(xf, np.polyval(c, xf), color=col, lw=2, label=lab)
        axes[1].set_title("Ajuste del modelo elegido")
        axes[1].legend(fontsize=8)
        axes[1].grid(True)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        graph_interp("BIC por grado (izquierda)",
            f"Cada barra muestra el BIC del polinomio de ese grado. "
            f"La <strong>barra verde</strong> (grado {mejor_g}) tiene el valor mínimo: es el modelo elegido. "
            f"La <strong>línea roja</strong> es el grado real ({grado_r}). "
            f"{'¡El BIC identificó correctamente el grado real!' if mejor_g == grado_r else f'El BIC eligió grado {mejor_g} ≠ real ({grado_r}). Con más datos o menos ruido convergen.'} "
            "Los grados altos (derecha) tienen BIC peor a pesar de ajustar mejor: $k\\ln(n)$ los penaliza.")
        graph_interp("Ajuste (derecha)",
            f"La <strong>curva verde</strong> es el modelo de grado {mejor_g} elegido por BIC. "
            f"La <strong>curva roja</strong> es el modelo real (grado {grado_r}). "
            "Los <strong>puntos blancos</strong> son los datos con ruido. "
            "Si las curvas son similares, el BIC detectó bien la complejidad del fenómeno.")

        if mejor_g == grado_r:
            interp(f"✅ BIC identificó correctamente el grado {mejor_g}.")
        else:
            nota(f"BIC eligió grado {mejor_g}, real era {grado_r}. Prueba aumentar datos o bajar σ.")

        st.markdown("### Código")
        code_bic = """\
import numpy as np

np.random.seed(99)

#Datos: polinomio de grado 2 con ruido 
x = np.linspace(-2, 2, 30)
y = 0.5*x**2 - 1.0*x + 0.3 + np.random.normal(0, 0.5, 30)

print(f"{'Grado':>6} {'k (params)':>11} {'Log-L':>10} {'BIC':>10}  {'':>10}")
print("-" * 52)

for grado in range(1, 8):
    #Ajustar el polinomio por mínimos cuadrados
    coefs  = np.polyfit(x, y, grado)
    y_pred = np.polyval(coefs, x)

    #Suma de cuadrados de residuos
    sse = np.sum((y - y_pred)**2)

    #Varianza MLE del ruido
    sigma2 = sse / len(x)

    #Log-verosimilitud del modelo gaussiano
    n     = len(x)
    log_L = -n/2 * np.log(2*np.pi*sigma2) - sse / (2*sigma2)

    #BIC: penaliza parámetros extras con k·ln(n)
    k   = grado + 1   # número de coeficientes = grado + 1
    bic = k * np.log(n) - 2 * log_L

    marca = "← MEJOR" if grado == 2 else ""
    print(f"{grado:>6} {k:>11} {log_L:>10.2f} {bic:>10.2f}  {marca}")
"""
        show_code(code_bic, "b_bic")

    with sub_bf:
        st.markdown("### Factor de Bayes")
        st.latex(r"BF_{12} = \frac{P(D \mid M_1)}{P(D \mid M_2)}")
        st.markdown("""
        | Símbolo | Significado |
        |---------|-------------|
        | $BF_{12}$ | Cuánto más apoya la evidencia a M₁ sobre M₂ |
        | $P(D \\mid M_i)$ | Evidencia marginal: verosimilitud promediada sobre el prior |
        """)
        st.markdown("""
        **Escala de Jeffreys:**
        | BF | Evidencia para M₁ |
        |----|------------------|
        | 1–3 | Apenas mencionable |
        | 3–10 | Sustancial |
        | 10–30 | Fuerte |
        | 30–100 | Muy fuerte |
        | >100 | Decisiva |
        """)
        interp("El BF penaliza complejidad automáticamente: un modelo con parámetros innecesarios "
               "obtiene baja evidencia marginal aunque su máximo de verosimilitud sea similar (<strong>efecto Occam</strong>).")

        st.markdown("### Código")
        code_bf = """\
import numpy as np
from scipy import stats
from scipy.special import betaln   # log de la función Beta

np.random.seed(42)

#Experimento: moneda, 7 caras en 10 lanzamientos 
n = 10   # total de lanzamientos
k = 7    # caras observadas

#Modelo 1: moneda justa  →  p = 0.5 fijo 
#Evidencia marginal exacta = Binomial(k=7, n=10, p=0.5)
#No hay parámetros libre → la evidencia es solo la verosimilitud
log_ev_M1 = stats.binom.logpmf(k, n, 0.5)
print(f"M1 (justa, p=0.5):")
print(f"  log P(D|M1) = {log_ev_M1:.4f}")

#Modelo 2: moneda libre  →  p ~ Beta(1,1) = Uniforme 
#Evidencia marginal = ∫₀¹ Binomial(k|n,p) · Beta(p|1,1) dp
#Esta integral tiene forma cerrada usando la función Beta:
alpha, beta_p = 1.0, 1.0   # prior uniforme
log_ev_M2 = (betaln(alpha + k, beta_p + n - k)   # log B(α+k, β+n-k)
             - betaln(alpha, beta_p))             # log B(α, β) = normalizador
print(f"M2 (libre, p ~ Uniforme):")
print(f"  log P(D|M2) = {log_ev_M2:.4f}")

#Factor de Bayes 
#BF_12 > 1: datos favorecen M1 (moneda justa)
#BF_12 < 1: datos favorecen M2 (moneda libre)
log_BF = log_ev_M1 - log_ev_M2
BF     = np.exp(log_BF)
print(f"\\nBF_12 = P(D|M1) / P(D|M2) = {BF:.4f}")

if BF < 1/3:
    print(f"→ Evidencia sustancial para M2 (moneda cargada): BF_21 = {1/BF:.2f}")
elif BF < 1:
    print(f"→ Evidencia leve para M2 (moneda libre)")
elif BF < 3:
    print(f"→ Evidencia apenas mencionable para M1 (moneda justa)")
else:
    print(f"→ Evidencia sustancial para M1 (moneda justa): BF_12 = {BF:.2f}")
"""
        show_code(code_bf, "b_bf")

# 
# TAB 7 — MONTE CARLO
# 
with tab_mc:
    st.markdown("## Simulación Monte Carlo")
    interp("Monte Carlo usa <strong>números aleatorios para calcular cantidades difíciles</strong>. "
           "Si no puedes integrar analíticamente, muestrea y promedia. "
           "Con suficientes muestras el promedio converge al valor exacto (Ley de los Grandes Números).")

    sub_pi, sub_int, sub_conv = st.tabs(["Estimar π", "Integración MC", "Convergencia"])

    with sub_pi:
        st.markdown("### Estimación de π con Monte Carlo")
        st.latex(r"\pi \approx 4 \cdot \frac{\#\{\text{puntos dentro del círculo}\}}{N}")
        interp("Si lanzamos N puntos al azar en el cuadrado [−1,1]², "
               "la fracción dentro del círculo unitario es π/4. Multiplicamos por 4.")

        col1, col2 = st.columns(2)
        n_mc  = col1.slider("N puntos", 100, 50000, 5000, 100, key="nmc7")
        sd7   = col2.number_input("Semilla", 0, 9999, 1, key="s7")

        np.random.seed(int(sd7))
        xm     = np.random.uniform(-1,1, n_mc)
        ym     = np.random.uniform(-1,1, n_mc)
        dentro = xm**2 + ym**2 <= 1.0
        pi_est = 4 * dentro.sum() / n_mc
        error  = abs(pi_est - np.pi)

        c1, c2, c3 = st.columns(3)
        c1.metric("π estimado", f"{pi_est:.6f}")
        c2.metric("π real",     f"{np.pi:.6f}")
        c3.metric("Error",      f"{error:.6f}")

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        mv = min(n_mc, 4000)
        axes[0].scatter(xm[:mv][dentro[:mv]],  ym[:mv][dentro[:mv]],
                        color="#58a6ff", s=1.5, alpha=0.5, label="Dentro")
        axes[0].scatter(xm[:mv][~dentro[:mv]], ym[:mv][~dentro[:mv]],
                        color="#f78166", s=1.5, alpha=0.4, label="Fuera")
        tc = np.linspace(0, 2*np.pi, 300)
        axes[0].plot(np.cos(tc), np.sin(tc), color="#3fb950", lw=2)
        axes[0].set_aspect("equal")
        axes[0].set_title(f"π ≈ {pi_est:.5f}  (N = {n_mc:,})")
        axes[0].legend(markerscale=6, fontsize=8)
        axes[0].grid(True)

        ns = np.logspace(2, np.log10(n_mc), 120).astype(int)
        pc = [4*(xm[:k]**2 + ym[:k]**2 <= 1).sum()/k for k in ns]
        axes[1].semilogx(ns, pc,  color="#58a6ff", lw=1.8, label="Estimación de π")
        axes[1].axhline(np.pi,    color="#f78166", lw=1.8, linestyle="--", label="π real")
        axes[1].fill_between(ns, np.pi-0.1, np.pi+0.1, alpha=0.07, color="#3fb950")
        axes[1].set_xlabel("N (escala log)")
        axes[1].set_ylabel("Estimación de π")
        axes[1].set_title("Convergencia hacia π")
        axes[1].legend()
        axes[1].grid(True)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        graph_interp("Dispersión de puntos (izquierda)",
            "Los <strong>puntos azules</strong> caen dentro del círculo (x²+y² ≤ 1). "
            "Los <strong>puntos rojos</strong> quedan fuera. "
            f"Con N = {n_mc:,}, la razón azul/total × 4 = {pi_est:.5f} ≈ π. "
            "La intuición: el círculo ocupa π/4 del cuadrado, así que la proporción de puntos azules estima π/4.")
        graph_interp("Convergencia (derecha)",
            "Eje X en escala logarítmica. La <strong>curva azul</strong> es la estimación con N puntos. "
            "La <strong>línea roja</strong> es el valor real. "
            "La estimación oscila mucho al inicio (N pequeño) y se estabiliza conforme N crece. "
            "La <strong>franja verde</strong> es π ± 0.1. La convergencia es lenta: esto es intrínseco al método Monte Carlo.")

        st.markdown("### Código")
        code_pi = """\
import numpy as np

np.random.seed(1)

#Parámetros 
N = 10_000   # número de puntos aleatorios a lanzar

#Generar puntos uniformes en el cuadrado [-1,1]² 
x = np.random.uniform(-1, 1, N)
y = np.random.uniform(-1, 1, N)

#Decidir si cada punto está dentro del círculo 
#Condición geométrica: distancia al origen ≤ radio = 1
#Es decir: x² + y² ≤ 1
dentro = x**2 + y**2 <= 1.0

#Estimación de π 
#Área del círculo = π·r² = π  (r=1)
#Área del cuadrado = (2r)² = 4
#Proporción = π/4  →  π ≈ 4 · (puntos dentro / total)
pi_estimado = 4 * dentro.sum() / N

print(f"Puntos totales  : {N:,}")
print(f"Puntos dentro   : {dentro.sum():,}")
print(f"π estimado      : {pi_estimado:.6f}")
print(f"π real          : {np.pi:.6f}")
print(f"Error absoluto  : {abs(pi_estimado - np.pi):.6f}")
print(f"Error relativo  : {abs(pi_estimado - np.pi)/np.pi:.4%}")
"""
        show_code(code_pi, "b_pi")

    with sub_int:
        st.markdown("### Integración Monte Carlo")
        st.latex(r"\int_a^b f(x)\,dx \;\approx\; (b-a)\cdot\frac{1}{N}\sum_{i=1}^{N} f(x_i), \quad x_i \sim U(a,b)")
        st.markdown("""
        | Símbolo | Significado |
        |----------|-------------|
        | $a, b$ | Límites de integración |
        | $N$ | Muestras: mayor → mayor precisión |
        | $x_i$ | Puntos aleatorios uniformes en $[a,b]$ |
        | $f(x_i)$ | Valor de la función en cada punto |
        """)
        st.markdown("Calculamos $\\int_0^1 e^{-x^2}\\,dx$ (no tiene forma cerrada elemental).")

        n_int = st.slider("N muestras", 100, 100000, 10000, 100, key="ni7")
        sd_i  = st.number_input("Semilla", 0, 9999, 3, key="si7")
        np.random.seed(int(sd_i))
        xi      = np.random.uniform(0, 1, n_int)
        fv      = np.exp(-xi**2)
        int_mc  = fv.mean()
        int_r   = 0.7468241328124271
        se_mc   = fv.std() / np.sqrt(n_int)

        c1, c2, c3 = st.columns(3)
        c1.metric("Estimación MC", f"{int_mc:.6f}")
        c2.metric("Valor exacto",  f"{int_r:.6f}")
        c3.metric("Error",         f"{abs(int_mc-int_r):.6f}")

        interp(f"Error estándar = {se_mc:.2e}. Con N = {n_int:,}, error = {abs(int_mc-int_r):.2e}. "
               "Para dividir el error entre 2 necesitas 4× más muestras (regla √N).")

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        xf = np.linspace(0, 1, 300)
        axes[0].plot(xf, np.exp(-xf**2), color="#58a6ff", lw=2.5, label="$e^{-x^2}$")
        axes[0].fill_between(xf, np.exp(-xf**2), alpha=0.18, color="#58a6ff")
        axes[0].scatter(xi[:400], np.exp(-xi[:400]**2),
                        color="#3fb950", s=5, alpha=0.5, label="Muestras MC")
        axes[0].axhline(int_mc, color="#f78166", lw=1.8, linestyle="--",
                        label=f"Media = {int_mc:.4f}")
        axes[0].set_title("Muestras sobre la función")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("f(x)")
        axes[0].legend(fontsize=8)
        axes[0].grid(True)

        ns_c = np.logspace(2, np.log10(n_int), 100).astype(int)
        ests = np.array([np.exp(-xi[:k]**2).mean() for k in ns_c])
        ses  = np.array([np.exp(-xi[:k]**2).std()/np.sqrt(k) for k in ns_c])
        axes[1].semilogx(ns_c, ests, color="#58a6ff", lw=1.8, label="Estimación")
        axes[1].fill_between(ns_c, ests-2*ses, ests+2*ses,
                             alpha=0.2, color="#58a6ff", label="±2 SE")
        axes[1].axhline(int_r, color="#f78166", lw=1.8, linestyle="--",
                        label=f"Valor real = {int_r:.4f}")
        axes[1].set_xlabel("N (escala log)")
        axes[1].set_ylabel("Estimación")
        axes[1].set_title("Convergencia de la integral MC")
        axes[1].legend(fontsize=8)
        axes[1].grid(True)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        graph_interp("Muestras sobre la función (izquierda)",
            "El <strong>área bajo la curva azul</strong> es la integral. "
            "Los <strong>puntos verdes</strong> son muestras aleatorias de f(x). "
            f"La <strong>línea roja</strong> es la media de todos los f(xᵢ) = {int_mc:.4f}. "
            "Como x ~ U(0,1), la media de f(x) ES la integral. "
            "La intuición: la media 'barre' la función con puntos aleatorios.")
        graph_interp("Convergencia (derecha)",
            "La <strong>curva azul</strong> converge al valor real (línea roja) conforme N crece. "
            "La <strong>franja celeste</strong> es el intervalo de confianza del 95 % (±2 SE). "
            "A mayor N, la franja se estrecha: más muestras = más precisión. "
            "La convergencia es lenta por diseño: error ∝ 1/√N.")

        st.markdown("### Código")
        code_int = """\
import numpy as np

np.random.seed(3)

#Función a integrar 
#∫₀¹ e^(-x²) dx  no tiene forma cerrada en términos elementales
def f(x):
    return np.exp(-x**2)

N = 100_000   # número de muestras

#Integración Monte Carlo 
#Clave teórica: E[f(X)] = ∫ f(x)·p(x) dx
#Si X ~ U(0,1), entonces p(x) = 1, y E[f(X)] = ∫₀¹ f(x) dx
x_muestras = np.random.uniform(0, 1, N)

#Promedio de f evaluada en cada muestra ≈ la integral
estimacion = f(x_muestras).mean()

#Error estándar: σ_f / √N  (incertidumbre de la estimación)
error_std  = f(x_muestras).std() / np.sqrt(N)

valor_real = 0.7468241328124271

print(f"N              = {N:,}")
print(f"Estimación MC  = {estimacion:.6f}")
print(f"Valor real     = {valor_real:.6f}")
print(f"Error absoluto = {abs(estimacion - valor_real):.6f}")
print(f"Error estándar = {error_std:.6f}   (±2 SE = {2*error_std:.6f})")
"""
        show_code(code_int, "b_int")

    with sub_conv:
        st.markdown("### Ley de los Grandes Números y tasa de convergencia")
        st.latex(r"\text{SE}(\hat{\mu}) = \frac{\hat{\sigma}}{\sqrt{N}} \;\propto\; \frac{1}{\sqrt{N}}")
        st.markdown("""
        | Símbolo | Significado |
        |---------|-------------|
        | $\\hat{\\mu}$ | Estimación Monte Carlo de la media |
        | $\\hat{\\sigma}$ | Desviación estándar de las muestras |
        | $N$ | Número de muestras |
        | $1/\\sqrt{N}$ | Tasa de convergencia del error |
        """)
        interp("Para reducir el error a la <strong>mitad</strong> necesitas <strong>4×</strong> más muestras. "
               "Para reducirlo a la décima parte, 100×. "
               "Esta lentitud es intrínseca al método Monte Carlo.")

        st.markdown("### Código")
        code_conv = """\
import numpy as np

np.random.seed(42)

#Objetivo: estimar E[X²] donde X ~ N(0,1) 
#Valor exacto: E[X²] = Var(X) + E[X]² = 1 + 0 = 1
valor_real = 1.0

print(f"{'N':>10} {'Estimación':>12} {'Error std (SE)':>16} {'Error real':>12}")
print("-" * 55)

Ns = [100, 500, 1_000, 5_000, 10_000, 100_000]
for N in Ns:
    #Generar N muestras de N(0,1)
    muestras  = np.random.normal(0, 1, N)

    #Estadística de interés: X²
    cuadrados = muestras**2

    #Estimación MC: promedio de X²
    estimacion = cuadrados.mean()

    #Error estándar: std(X²) / sqrt(N)
    #Cuantifica la incertidumbre de nuestra estimación
    se = cuadrados.std() / np.sqrt(N)

    #Error real: distancia al valor exacto
    error = abs(estimacion - valor_real)

    print(f"{N:>10,} {estimacion:>12.5f} {se:>16.5f} {error:>12.5f}")

print()
print("Regla: multiplicar N por 4 → error se divide por 2 (≈ √4 = 2)")
print("Multiplicar N por 100 → error se divide por 10 (≈ √100 = 10)")
"""
        show_code(code_conv, "b_conv")

        # Gráfica de convergencia
        np.random.seed(42)
        Ns_p = np.logspace(1.5, 5, 80).astype(int)
        errs = []
        ses_p = []
        for N in Ns_p:
            m = np.random.normal(0,1,N)**2
            errs.append(abs(m.mean()-1.0))
            ses_p.append(m.std()/np.sqrt(N))

        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.loglog(Ns_p, errs,  color="#58a6ff", lw=2,   label="Error real")
        ax.loglog(Ns_p, ses_p, color="#3fb950", lw=1.8, linestyle="--",
                  label="Error estándar (SE)")
        ref = np.array([Ns_p[0], Ns_p[-1]], dtype=float)
        ax.loglog(ref, 1/np.sqrt(ref) * errs[0]*np.sqrt(Ns_p[0]),
                  color="#f78166", lw=1.2, linestyle=":", label="Referencia 1/√N")
        ax.set_xlabel("N (escala log)")
        ax.set_ylabel("Error (escala log)")
        ax.set_title("Convergencia Monte Carlo: error ∝ 1/√N")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.4)
        st.pyplot(fig, use_container_width=True)

        graph_interp("Convergencia en escala log-log",
            "En escala log-log, una relación 1/√N aparece como <strong>recta con pendiente −½</strong>. "
            "La <strong>curva azul</strong> es el error real de estimación. "
            "La <strong>curva verde punteada</strong> es el error estándar teórico. "
            "La <strong>línea roja punteada</strong> es la referencia exacta 1/√N. "
            "Que las tres sean paralelas confirma que el error MC sigue esta ley. "
            "Para bajar el error de 0.1 a 0.01 hay que pasar de N ≈ 100 a N ≈ 10,000.")

# 
# TAB 8 — REFERENCIAS
# 
with tab_refs:
    st.markdown("## Referencias")
    interp("Esta exposición se apoya principalmente en el capítulo 14 del texto de Correa Morales "
           "y Barrera Causil, que presenta herramientas de software para estadística bayesiana y "
           "sirve como puente entre la teoría bayesiana y su implementación computacional.")

    st.markdown("""
    <div class="card">
    <div class="section-label">fuente principal</div>
    <h3 style="color:#58a6ff;margin-top:0">Introducción a la Estadística Bayesiana</h3>
    <p><strong>Autores:</strong> Juan Carlos Correa Morales y Carlos Javier Barrera Causil</p>
    <p><strong>Editorial:</strong> Instituto Tecnológico Metropolitano - ITM, Medellín</p>
    <p><strong>Edición:</strong> 1.ª edición, 2018</p>
    <p><strong>Capítulo usado:</strong> Capítulo 14, <em>Software para estadística bayesiana</em></p>
    <p><strong>ISBN:</strong> 978-958-5414-24-2</p>
    <p><strong>DOI:</strong>
        <a href="https://dx.doi.org/10.22430/9789585414242" target="_blank" style="color:#58a6ff">
        10.22430/9789585414242
        </a>
    </p>
    <p><strong>Repositorio:</strong>
        <a href="https://hdl.handle.net/20.500.12622/1793" target="_blank" style="color:#58a6ff">
        https://hdl.handle.net/20.500.12622/1793
        </a>
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Relación con esta app")
    st.markdown("""
    | Tema de la app | Relación con la fuente |
    |----------------|------------------------|
    | Estadística bayesiana computacional | Motivación para usar software en problemas bayesianos |
    | MCMC | Implementación didáctica de simulación y cadenas de Markov |
    | Monte Carlo | Uso de simulación para aproximar cantidades difíciles |
    | Selección de modelos | Criterios y evidencia para comparar modelos |
    | Regresión bayesiana | Ejemplo aplicado de inferencia con incertidumbre posterior |
    """)

    nota("Las implementaciones de esta app fueron reescritas en Python con Streamlit, NumPy, SciPy y Matplotlib "
         "para fines de visualización interactiva. No corresponden a una reproducción literal del código del libro.")
