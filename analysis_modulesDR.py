"""
Modules d'analyse pour la génération de graphiques
Chaque fonction retourne une figure matplotlib avec un seul graphique
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns
from pandas.plotting import table
from statsmodels.tsa.stattools import kpss
from statsmodels.stats.weightstats import DescrStatsW
from sklearn.linear_model import LinearRegression
import logging
import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning

from utils import (
    perf_annualized, volatility_annualized, tracking_error,
    information_ratio, sharpe_ratio, alpha_beta,
    rolling_alpha_beta, compute_drawdown, correlation_rolling, first_valid_nonzero
)
from config import WINDOW, PERIODS_PER_YEAR

logger = logging.getLogger(__name__)
ppw = PERIODS_PER_YEAR


def plot_performance_absolute_relative(df, figsize=(8, 6)):
    """
    Performance absolue et relative avec ticks alignés
    
    Args:
        df: DataFrame avec les données de prix
        figsize: Tuple (width, height) pour la taille de la figure
    
    Returns:
        Figure matplotlib
    """
    logger.info("  → Calcul des performances absolue et relative...")
    
    fonds_col = df.columns[1]
    benchmark_col = df.columns[0]
    
    # Calcul des performances
    ref_fonds = first_valid_nonzero(df[fonds_col])
    ref_benchmark = first_valid_nonzero(df[benchmark_col])

    if not np.isnan(ref_fonds) and not np.isnan(ref_benchmark):
        perf_fonds = df[fonds_col] / ref_fonds * 100
        perf_benchmark = df[benchmark_col] / ref_benchmark * 100
        perf_relative = (perf_fonds / perf_benchmark - 1) * 100
    else:
        perf_fonds = perf_benchmark = perf_relative = pd.Series(np.nan, index=df.index)
    
    fig, ax1 = plt.subplots(figsize=figsize)
    
    ax1.plot(df.index, perf_fonds, label=f"{fonds_col} (absolu)", color='blue')
    ax1.plot(df.index, perf_benchmark, label="Benchmark", color='grey', linestyle='--')
    ax1.set_ylabel("Perf. cumulée (base 100)", color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.plot(df.index, perf_relative, label="Perf relative (%)", color='red')
    ax2.set_ylabel("Perf. relative vs Benchmark (%)", color="red")
    ax2.legend(loc="upper right")
    ax1.set_title(f"Performance absolue et relative - {fonds_col}")
    ax2.tick_params(axis='y', labelcolor="red")
    ax1.xaxis.set_minor_locator(AutoMinorLocator(0.25))
    ax1.xaxis.minorticks_on()
    
    # Alignement des ticks
    # 1. Forcer matplotlib à finaliser les ticks de ax1
    fig.canvas.draw()
    
    # 2. Récupérer les positions des ticks de ax1 (en coordonnées de données)
    ax1_ticks = ax1.get_yticks()
    
    # 3. Calculer la plage de perf_relative
    perf_rel_min = perf_relative.min()
    perf_rel_max = perf_relative.max()
    
    # 4. Calculer le nombre de ticks voulus (même nombre que ax1)
    n_ticks = len(ax1_ticks)
    
    # 5. Créer des ticks "ronds" pour ax2
    # Trouver un pas rond qui donne environ n_ticks valeurs
    perf_range = perf_rel_max - perf_rel_min
    rough_step = perf_range / (n_ticks - 1) if n_ticks > 1 else perf_range
    
    # Arrondir le pas à une valeur "ronde" (0.5, 1, 2, 5, 10, 20, 50, etc.)
    magnitude = 10 ** np.floor(np.log10(rough_step))
    normalized = rough_step / magnitude
    if normalized < 1.5:
        nice_step = 1 * magnitude
    elif normalized < 3.5:
        nice_step = 2 * magnitude
    elif normalized < 7.5:
        nice_step = 5 * magnitude
    else:
        nice_step = 10 * magnitude
    
    # 5. Créer exactement n_ticks valeurs rondes centrées sur les données
    # Trouver une valeur de départ ronde proche du milieu
    mid_value = (perf_rel_min + perf_rel_max) / 2
    mid_tick = np.round(mid_value / nice_step) * nice_step
    
    # Créer les ticks symétriquement autour du milieu
    half_ticks = (n_ticks - 1) // 2
    ax2_ticks = np.array([mid_tick + (i - half_ticks) * nice_step for i in range(n_ticks)])
    
    # 6. Appliquer les ticks à ax2
    ax2.set_yticks(ax2_ticks)
    
    # 7. Ajuster les limites de ax2 pour aligner avec ax1
    # Les ticks doivent être à la même position verticale sur les deux axes
    ax1_min, ax1_max = ax1.get_ylim()
    ax1_tick_range = ax1_ticks[-1] - ax1_ticks[0]
    ax2_tick_range = ax2_ticks[-1] - ax2_ticks[0]
    
    # Calculer les marges proportionnelles
    if ax1_tick_range != 0:
        bottom_margin_ratio = (ax1_ticks[0] - ax1_min) / ax1_tick_range
        top_margin_ratio = (ax1_max - ax1_ticks[-1]) / ax1_tick_range
    else:
        bottom_margin_ratio = top_margin_ratio = 0.1
    
    ax2_min = ax2_ticks[0] - bottom_margin_ratio * ax2_tick_range
    ax2_max = ax2_ticks[-1] + top_margin_ratio * ax2_tick_range
    ax2.set_ylim(ax2_min, ax2_max)
    
    fig.tight_layout()

    return fig


def plot_volatility_rolling(df, figsize=(8, 6)):
    """
    Volatilité glissante long terme
    
    Args:
        df: DataFrame avec les données de prix
        figsize: Tuple (width, height) pour la taille de la figure
    
    Returns:
        Figure matplotlib
    """
    logger.info("  → Calcul de la volatilité glissante...")
    
    fonds_col = df.columns[1]
    benchmark_col = df.columns[0]
    
    ret_fonds = df[fonds_col].astype(float).pct_change()
    ret_bench = df[benchmark_col].astype(float).pct_change()
    
    vol_fonds = ret_fonds.rolling(WINDOW).std() * np.sqrt(ppw)
    vol_bench = ret_bench.rolling(WINDOW).std() * np.sqrt(ppw)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(vol_fonds.index, vol_fonds*100, label=f"{fonds_col}", color="blue")
    ax.plot(vol_bench.index, vol_bench*100, label=f"{benchmark_col}", 
            color="grey", linestyle="--")
    ax.set_ylabel("Volatilité annualisée (%)")
    ax.set_title(f"Volatilité glissante ({WINDOW} semaines)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig


def plot_tracking_error_rolling(df, figsize=(8, 6)):
    """
    Tracking Error glissant long terme
    
    Args:
        df: DataFrame avec les données de prix
        figsize: Tuple (width, height) pour la taille de la figure
    
    Returns:
        Figure matplotlib
    """
    logger.info("  → Calcul du tracking error glissant...")
    
    fonds_col = df.columns[1]
    benchmark_col = df.columns[0]
    
    ret_fonds = df[fonds_col].astype(float).pct_change()
    ret_bench = df[benchmark_col].astype(float).pct_change()
    
    tracking_err = (ret_fonds - ret_bench).rolling(WINDOW).std() * np.sqrt(ppw)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(tracking_err.index, tracking_err*100, color="red", label="Tracking Error")
    ax.set_ylabel("Tracking Error annualisée (%)")
    ax.set_title(f"Tracking Error glissante ({WINDOW} semaines)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig


def plot_alpha_beta_rolling(df, figsize=(8, 6)):
    """
    Alpha et Beta glissants
    
    Args:
        df: DataFrame avec les données de prix
        figsize: Tuple (width, height) pour la taille de la figure
    
    Returns:
        Figure matplotlib
    """
    logger.info("  → Calcul des alpha/beta glissants...")
    
    fonds_col = df.columns[1]
    
    ret_fonds = df[fonds_col].astype(float).pct_change()
    ret_bench = df[df.columns[0]].astype(float).pct_change()
    
    alpha_beta_df = rolling_alpha_beta(ret_fonds, ret_bench, WINDOW)
    
    fig, ax1 = plt.subplots(figsize=figsize)
    
    ax1.plot(alpha_beta_df.index, alpha_beta_df["Alpha"]*WINDOW, 
             color="blue", label="Alpha (annualisé)")
    ax1.set_ylabel("Alpha annualisé (en %)", color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.plot(alpha_beta_df.index, alpha_beta_df["Beta"], color="red", label="Beta")
    ax2.set_ylabel("Beta", color="red")
    ax2.tick_params(axis='y', labelcolor="red")
    ax1.set_title(f"Alpha et Beta glissants (fenêtre {WINDOW} semaines) - {fonds_col}")
    ax1.xaxis.set_minor_locator(AutoMinorLocator(0.25))
    ax1.xaxis.minorticks_on()
    
    # Alignement des ticks
    # 1. Forcer matplotlib à finaliser les ticks de ax1
    fig.canvas.draw()
    
    # 2. Récupérer les positions des ticks de ax1 (en coordonnées de données)
    ax1_ticks = ax1.get_yticks()
    
    # 3. Calculer la plage de perf_relative
    beta_min = alpha_beta_df["Beta"].min()
    beta_max = alpha_beta_df["Beta"].max()

    # 4. Calculer le nombre de ticks voulus (même nombre que ax1)
    n_ticks = len(ax1_ticks)
    
    # 5. Créer des ticks "ronds" pour ax2
    # Trouver un pas rond qui donne environ n_ticks valeurs
    beta_range = beta_max - beta_min
    rough_step = beta_range / (n_ticks - 1) if n_ticks > 1 else beta_range
    
    # Arrondir le pas à une valeur "ronde" (0.5, 1, 2, 5, 10, 20, 50, etc.)
    magnitude = 10 ** np.floor(np.log10(rough_step))
    normalized = rough_step / magnitude
    if normalized < 1.5:
        nice_step = 1 * magnitude
    elif normalized < 3.5:
        nice_step = 2 * magnitude
    elif normalized < 7.5:
        nice_step = 5 * magnitude
    else:
        nice_step = 10 * magnitude

    # 5. Créer exactement n_ticks valeurs rondes centrées sur les données
    # Trouver une valeur de départ ronde proche du milieu
    mid_value = (beta_min + beta_max) / 2
    mid_tick = np.round(mid_value / nice_step) * nice_step
    
    # Créer les ticks symétriquement autour du milieu
    half_ticks = (n_ticks - 1) // 2
    ax2_ticks = np.array([mid_tick + (i - half_ticks) * nice_step for i in range(n_ticks)])
    
    # 6. Appliquer les ticks à ax2
    ax2.set_yticks(ax2_ticks)

    # 7. Ajuster les limites de ax2 pour aligner avec ax1
    # Les ticks doivent être à la même position verticale sur les deux axes
    ax1_min, ax1_max = ax1.get_ylim()
    ax1_tick_range = ax1_ticks[-1] - ax1_ticks[0]
    ax2_tick_range = ax2_ticks[-1] - ax2_ticks[0]
    
    # Calculer les marges proportionnelles
    if ax1_tick_range != 0:
        bottom_margin_ratio = (ax1_ticks[0] - ax1_min) / ax1_tick_range
        top_margin_ratio = (ax1_max - ax1_ticks[-1]) / ax1_tick_range
    else:
        bottom_margin_ratio = top_margin_ratio = 0.1
    
    ax2_min = ax2_ticks[0] - bottom_margin_ratio * ax2_tick_range
    ax2_max = ax2_ticks[-1] + top_margin_ratio * ax2_tick_range
    ax2.set_ylim(ax2_min, ax2_max)

    fig.tight_layout()

    return fig



def plot_volatility(df, figsize=(8, 6)):
    """
    Volatilité
    
    Args:
        df: DataFrame avec les données de prix
        figsize: Tuple (width, height) pour la taille de la figure
    
    Returns:
        Figure matplotlib
    """
    logger.info("  → Calcul de la volatilité...")
    
    fonds_col = df.columns[1]
    benchmark_col = df.columns[0]
    
    ret_fonds = df[fonds_col].astype(float).pct_change()
    ret_bench = df[benchmark_col].astype(float).pct_change()
    
    vol_long_fonds = ret_fonds.rolling(WINDOW).std() * np.sqrt(ppw)
    vol_long_bench = ret_bench.rolling(WINDOW).std() * np.sqrt(ppw)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(vol_long_fonds.index, vol_long_fonds*100, label=f"{fonds_col}", color="blue")
    ax.plot(vol_long_bench.index, vol_long_bench*100, label=f"{benchmark_col}", 
            color="grey", linestyle="--")
    ax.set_ylabel("Volatilité annualisée (%)")
    ax.set_title(f"Volatilité ({WINDOW} semaines)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig



def plot_tracking_error(df, figsize=(8, 6)):
    """
    Tracking Error
    
    Args:
        df: DataFrame avec les données de prix
        figsize: Tuple (width, height) pour la taille de la figure
    
    Returns:
        Figure matplotlib
    """
    logger.info("  → Calcul du tracking error...")
    
    fonds_col = df.columns[1]
    benchmark_col = df.columns[0]
    
    ret_fonds = df[fonds_col].astype(float).pct_change()
    ret_bench = df[benchmark_col].astype(float).pct_change()

    vol_long_fonds = ret_fonds.rolling(WINDOW).std() * np.sqrt(ppw)
    vol_long_bench = ret_bench.rolling(WINDOW).std() * np.sqrt(ppw)
    diff=abs(vol_long_fonds-vol_long_bench)

    te_long = (ret_fonds - ret_bench).rolling(WINDOW).std() * np.sqrt(ppw)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(te_long.index, te_long*100, color="darkred", label="TE")
    ax.plot(te_long.index,diff*100, color='darkblue', label="Différence vol absolue")
    ax.set_ylabel("Tracking Error (%)")
    ax.set_title(f"Tracking Error ({WINDOW} semaines)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig


def plot_drawdown(df, figsize=(8, 6)):
    """
    Drawdown Fonds vs Benchmark
    
    Args:
        df: DataFrame avec les données de prix
        figsize: Tuple (width, height) pour la taille de la figure
    
    Returns:
        Figure matplotlib
    """
    logger.info("  → Calcul des drawdowns...")
    
    fonds_col = df.columns[1]
    benchmark_col = df.columns[0]
    
    dd_fonds = compute_drawdown(df[fonds_col])
    dd_bench = compute_drawdown(df[benchmark_col])
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(dd_fonds.index, dd_fonds, label=f"{fonds_col}", color="blue")
    ax.plot(dd_bench.index, dd_bench, label=f"{benchmark_col}", 
            color="grey", linestyle="--")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Drawdown - Fonds vs Benchmark")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig


def plot_statistics_table(df, figsize=(8, 6)):
    """
    Tableau récapitulatif des statistiques
    
    Args:
        df: DataFrame avec les données de prix
        figsize: Tuple (width, height) pour la taille de la figure
    
    Returns:
        Figure matplotlib
    """
    logger.info("  → Génération du tableau statistique...")
    
    fonds_col = df.columns[1]
    benchmark_col = df.columns[0]
    
    ret_fonds = df[fonds_col].astype(float).pct_change()
    ret_bench = df[benchmark_col].astype(float).pct_change()
    
    stats = {}
    
    # Performances annualisées
    cutoff_3y = df.index.max() - pd.DateOffset(weeks=156)
    cutoff_5y = df.index.max() - pd.DateOffset(weeks=260)
    stats["Perf 3 ans ann."] = [
        perf_annualized(df.loc[df.index >= cutoff_3y, fonds_col], ppw),
        perf_annualized(df.loc[df.index >= cutoff_3y, benchmark_col], ppw)
    ]
    stats["Perf 5 ans ann."] = [
        perf_annualized(df.loc[df.index >= cutoff_5y, fonds_col], ppw),
        perf_annualized(df.loc[df.index >= cutoff_5y, benchmark_col], ppw)
    ]
    
    # Performance par année
    for year in [2025, 2024, 2023, 2022]:
        year_str = str(year)
        subset = df.loc[df.index.year == year]
        stats[f"Perf {year_str}"] = [
            subset[fonds_col].iloc[-1] / subset[fonds_col].iloc[0] - 1 if not subset.empty else np.nan,
            subset[benchmark_col].iloc[-1] / subset[benchmark_col].iloc[0] - 1 if not subset.empty else np.nan
        ]
    
    # Statistiques 36 mois
    cutoff_36m = df.index.max() - pd.DateOffset(weeks=156)
    rf36_fonds = ret_fonds.loc[ret_fonds.index >= cutoff_36m].dropna()
    rf36_bench = ret_bench.loc[ret_bench.index >= cutoff_36m].dropna()
    
    stats["Vol ann. 36m"] = [
        volatility_annualized(rf36_fonds, ppw),
        volatility_annualized(rf36_bench, ppw)
    ]
    
    stats["Corrélation 36m"] = [
        correlation_rolling(rf36_fonds, rf36_bench),
        1.0
    ]
    
    stats["TE ann. 36m"] = [
        tracking_error(rf36_fonds, rf36_bench, ppw),
        np.nan
    ]
    
    stats["Ratio info 36m"] = [
        information_ratio(rf36_fonds, rf36_bench, ppw),
        np.nan
    ]
    
    stats["Sharpe 36m"] = [
        sharpe_ratio(rf36_fonds, ppw),
        sharpe_ratio(rf36_bench, ppw)
    ]
    
    alpha_f, beta_f = alpha_beta(rf36_fonds, rf36_bench, ppw)
    stats["Beta 36m"] = [beta_f, 1.0]
    stats["Alpha ann. 36m"] = [alpha_f, np.nan]
    
    # Hit Ratio 36m (surperformance mensuelle)
    cutoff_36m_monthly = df.index.max() - pd.DateOffset(months=36)
    monthly_df = df.resample('ME').last().dropna(how='all')
    monthly_df_36 = monthly_df.loc[monthly_df.index >= cutoff_36m_monthly]
    
    if not monthly_df_36.empty:
        ret_fonds_m = monthly_df_36[fonds_col].astype(float).pct_change().dropna()
        ret_bench_m = monthly_df_36[benchmark_col].astype(float).pct_change().dropna()
        common_index = ret_fonds_m.index.intersection(ret_bench_m.index)
        ret_fonds_m = ret_fonds_m.loc[common_index]
        ret_bench_m = ret_bench_m.loc[common_index]
        num_months = len(ret_fonds_m)
        if num_months > 0:
            surperf_ratio = (ret_fonds_m > ret_bench_m).sum() / num_months
        else:
            surperf_ratio = np.nan
    else:
        surperf_ratio = np.nan
    
    stats["Hit Ratio 36m"] = [surperf_ratio, np.nan]

    # DataFrame final
    summary = pd.DataFrame(stats, index=[fonds_col, benchmark_col]).T
    summary = summary.map(lambda x: f"{x*100:.2f}%" if isinstance(x, float) and pd.notna(x) else "")
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    tbl = ax.table(
        cellText=summary.values,
        colLabels=summary.columns,
        rowLabels=summary.index,
        loc='center'
    )
    tbl.auto_set_font_size(True)
    tbl.set_fontsize(14)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_height(cell.get_height() * 1.6) #Ajuste la hauteur des cellules pour que le texte soit adapté manuellement
    ax.set_title("Tableau récapitulatif des statistiques",size=17)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_position([0, 0, 1, 1])

    return fig


def plot_alpha_tests(df, figsize=(8, 6)):
    """
    Tests statistiques (KPSS et alpha > 0)
    
    Args:
        df: DataFrame avec les données de prix
        figsize: Tuple (width, height) pour la taille de la figure
    
    Returns:
        Figure matplotlib
    """
    logger.info("  → Tests statistiques (KPSS et alpha > 0)...")
    
    fonds_col = df.columns[1]
    benchmark_col = df.columns[0]
    
    ret_fonds = df[fonds_col].astype(float).pct_change()
    ret_bench = df[benchmark_col].astype(float).pct_change()
    
    Alpha = ret_fonds - ret_bench
    Alpha_clean = Alpha.dropna()
    
    # Test KPSS (suppress InterpolationWarning)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=InterpolationWarning)
        statistic, p_value, lags, critical_values = kpss(Alpha_clean, regression='c')
    
    # Test alpha > 0
    desc_stats = DescrStatsW(Alpha_clean)
    t_stat, p_value2, _ = desc_stats.ttest_mean(0, alternative='larger')
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(Alpha.index, Alpha, label=f"{fonds_col}", color="blue")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Alpha (%)")
    ax.set_title("Test KPSS et Stationnarité de l'Alpha")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Annotations des résultats
    ymax = ax.get_ylim()[1]
    x_position = Alpha.index[int(len(Alpha) * 0.7)]
    
    if p_value2 < 0.05:
        ax.text(x_position, ymax * 0.85, '✓ Alpha > 0 (significatif)', 
                color='green', fontsize=11, fontweight='bold')
    else:
        ax.text(x_position, ymax * 0.85, '✗ Alpha non significatif', 
                color='red', fontsize=11, fontweight='bold')
    
    if p_value < 0.01:
        ax.text(x_position, ymax * 0.7, '✗ Série non stationnaire', 
                color='red', fontsize=11, fontweight='bold')
    else:
        ax.text(x_position, ymax * 0.7, '✓ Série stationnaire', 
                color='green', fontsize=11, fontweight='bold')
    
    fig.tight_layout()
    return fig


def plot_returns_distribution(df, figsize=(8, 6)):
    """
    Distribution des rendements avec densité de probabilité (KDE)
    
    Args:
        df: DataFrame avec les données de prix
        figsize: Tuple (width, height) pour la taille de la figure
    
    Returns:
        Figure matplotlib
    """
    logger.info("  → Calcul de la distribution des rendements...")
    
    fonds_col = df.columns[1]
    benchmark_col = df.columns[0]
    
    ret_fonds = df[fonds_col].astype(float).pct_change()
    ret_bench = df[benchmark_col].astype(float).pct_change()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Conversion en pourcentage et suppression des NaN
    ret_fonds_pct = ret_fonds.dropna() * 100
    ret_bench_pct = ret_bench.dropna() * 100
    
    # Tracer les densités de probabilité avec KDE gaussien
    ret_fonds_pct.plot.kde(ax=ax, label=f"{fonds_col}", color='blue', linewidth=2, bw_method='scott')
    ret_bench_pct.plot.kde(ax=ax, label=f"{benchmark_col}", color='red', linewidth=2, alpha=0.7, bw_method='scott')
    
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel("Rendements (%)")
    ax.set_ylabel("Densité de probabilité")
    ax.set_title("Distribution des rendements (KDE)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig


def plot_relative_performance_multifund(df, figsize=(8, 6)):
    """
    Performance relative cumulée multi-fonds
    
    Args:
        df: DataFrame avec les données de prix
        figsize: Tuple (width, height) pour la taille de la figure
    
    Returns:
        Figure matplotlib
    """
    logger.info("  → Calcul des performances relatives multi-fonds...")
    
    benchmark_col = df.columns[0]
    fonds_col=df.columns[1]
    
    # Création du mapping de couleurs consistant
    color_funds = plt.cm.tab10.colors
    color_map = {}
    color_map[benchmark_col] = "red"
    color_map[fonds_col] = "black"
    other_cols = [col for col in df.columns if col not in [benchmark_col, fonds_col]]
    for i, col in enumerate(other_cols):
        color_map[col] = color_funds[i % len(color_funds)]

    fig, ax = plt.subplots(figsize=figsize)
    
    for col in df.columns:
        if col == benchmark_col:
            continue
        
        # Série fonds et benchmark sans zéros
        s_f = df[col].astype(float).replace(0, np.nan)
        s_b = df[benchmark_col].astype(float).replace(0, np.nan)
        
        first_valid_idx = s_f.first_valid_index()
        if first_valid_idx is None:
            continue
        
        base_f = s_f.loc[first_valid_idx]
        base_b = s_b.loc[first_valid_idx]
        
        perf_fonds = s_f / base_f
        perf_bench = s_b / base_b
        perf_relative = (perf_fonds / perf_bench - 1.0) * 100.0
        
        if col == fonds_col:
            z = 100
        else:
            z = 1

        line, = ax.plot(df.index[df.index >= first_valid_idx],
                        perf_relative[df.index >= first_valid_idx],
                        label=col, linewidth=3, zorder=z,
                        color=color_map[col])
        
        if col == fonds_col:
            line.set_color('black')
        
        # Ajouter le label au dernier point valide
        last_valid_idx = perf_relative.last_valid_index()
        if last_valid_idx is not None:
            last_x = last_valid_idx
            last_y = perf_relative.loc[last_valid_idx]
            ax.text(last_x, last_y, f' {col}', va='center', ha='left', color=line.get_color(),fontweight="bold")
    
    # Étendre la limite x pour laisser de l'espace aux labels
    x_min, x_max = ax.get_xlim()
    ax.set_xlim(x_min, x_max + (x_max - x_min) * 0.05)
    
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Performance relative vs Benchmark (%)")
    ax.set_title("Performance relative cumulée des fonds vs Benchmark")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df, figsize=(8, 6)):
    """
    Heatmap corrélation des performances relatives
    
    Args:
        df: DataFrame avec les données de prix
        figsize: Tuple (width, height) pour la taille de la figure
    
    Returns:
        Figure matplotlib
    """
    logger.info("  → Calcul de la matrice de corrélation...")
    
    benchmark_col = df.columns[0]
    returns = df.astype(float).pct_change()
    
    perf_relatives = pd.DataFrame()
    for col in df.columns:
        if col != benchmark_col:
            perf_relatives[col] = returns[col] - returns[benchmark_col]
    
    corr_matrix = perf_relatives.corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, 
                fmt=".2f", ax=ax, cbar_kws={'label': 'Corrélation'})
    ax.set_title("Corrélation des performances relatives vs Benchmark")
    
    fig.tight_layout()
    return fig


def plot_volatility_vs_performance(df, figsize=(8, 6)):
    """
    Scatter Volatilité vs Performance (36 mois)
    
    Args:
        df: DataFrame avec les données de prix
        figsize: Tuple (width, height) pour la taille de la figure
    
    Returns:
        Figure matplotlib
    """
    logger.info("  → Scatter volatilité/performance...")
    
    benchmark_col = df.columns[0]
    fonds_col = df.columns[1]
    cutoff_36m = df.index.max() - pd.DateOffset(weeks=156)
    x_vol = []
    y_perf = []
    labels = []
    colors = []
    color_funds = plt.cm.tab10.colors
    # Création du mapping de couleurs consistant
    color_map = {}
    color_map[benchmark_col] = "red"
    color_map[fonds_col] = "black"
    other_cols = [col for col in df.columns if col not in [benchmark_col, fonds_col]]
    for i, col in enumerate(other_cols):
        color_map[col] = color_funds[i % len(color_funds)]
    
    for col in df.columns:
        subset = df.loc[df.index >= cutoff_36m, col].astype(float)
        returns_ = subset.pct_change().dropna()
        
        if len(subset) > 0:
            vol = returns_.std() * np.sqrt(ppw)
            perf_total = subset.iloc[-1] / subset.iloc[0] - 1
            years = len(subset) / ppw
            perf_ann = (1 + perf_total) ** (1/years) - 1
            
            x_vol.append(vol * 100)
            y_perf.append(perf_ann * 100)
            labels.append(col)
            colors.append(color_map[col])  # Utilisation du mapping
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.scatter(x_vol, y_perf, c=colors, s=100, edgecolors="black", linewidths=1)
    
    for i, label in enumerate(labels):
        ax.text(x_vol[i] + 0.1, y_perf[i], label, fontsize=9)
    
    ax.set_xlabel("Volatilité 36 mois annualisée (%)")
    ax.set_ylabel("Performance 36 mois annualisée (%)")
    ax.set_title("Volatilité vs Performance (36 mois)")
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig


def plot_alpha_ir_table(df, figsize=(8, 6)):
    """
    Table alpha annualisé & information ratio sur 36 mois
    
    Args:
        df: DataFrame avec les données de prix
        figsize: Tuple (width, height) pour la taille de la figure
    
    Returns:
        Figure matplotlib
    """
    logger.info("  → Calcul alpha et IR pour tous les fonds...")
    
    benchmark_col = df.columns[0]
    cutoff_36m = df.index.max() - pd.DateOffset(weeks=156)
    
    summary_alpha_ir = pd.DataFrame(columns=["Alpha ann. 36m", "Information Ratio 36m"])
    
    for col in df.columns:
        if col != benchmark_col:
            subset_fonds = df.loc[df.index >= cutoff_36m, col].astype(float).pct_change().dropna()
            subset_bench = df.loc[df.index >= cutoff_36m, benchmark_col].astype(float).pct_change().dropna()
            
            model = LinearRegression()
            X = subset_bench.values.reshape(-1, 1)
            y = subset_fonds.values.reshape(-1, 1)
            model.fit(X, y)
            
            alpha_weekly = model.intercept_[0]
            alpha_ann = (1 + alpha_weekly) ** ppw - 1
            
            diff = subset_fonds - subset_bench
            ir = diff.mean() / diff.std() * np.sqrt(ppw)
            
            summary_alpha_ir.loc[col] = [alpha_ann*100, ir]
    
    summary_alpha_ir = summary_alpha_ir.round(2)
    summary_alpha_ir["Alpha ann. 36m"] = summary_alpha_ir["Alpha ann. 36m"].astype(str) + " %"
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    tbl = table(ax, summary_alpha_ir, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    ax.set_title("Alpha annualisé & Information Ratio (36 mois)")
    
    fig.tight_layout()
    return fig