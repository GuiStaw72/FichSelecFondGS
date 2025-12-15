"""
Fonctions utilitaires pour les calculs financiers
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import logging

logger = logging.getLogger(__name__)

def first_valid_nonzero(series):
    '''Utilisé pour calculer des performances en cas de séries qui commencent pas des 0 ou NaN'''
    valid = series[(series.notna()) & (series != 0)]
    return valid.iloc[0] if not valid.empty else np.nan


def perf_annualized(series, periods_per_year):
    """
    Calcule la performance annualisée à partir d'une série de prix
    
    Args:
        series: Série pandas de prix
        periods_per_year: Nombre de périodes par an
    
    Returns:
        float: Performance annualisée
    """
    if len(series) < 2:
        return np.nan
    
    total_return = series.iloc[-1] / series.iloc[0] - 1
    years = len(series) / periods_per_year
    return (1 + total_return)**(1/years) - 1


def volatility_annualized(returns, periods_per_year):
    """
    Calcule la volatilité annualisée
    
    Args:
        returns: Série pandas de rendements
        periods_per_year: Nombre de périodes par an
    
    Returns:
        float: Volatilité annualisée
    """
    return returns.std() * np.sqrt(periods_per_year)


def tracking_error(returns_a, returns_b, periods_per_year):
    """
    Calcule le tracking error annualisé
    
    Args:
        returns_a: Rendements du fonds
        returns_b: Rendements du benchmark
        periods_per_year: Nombre de périodes par an
    
    Returns:
        float: Tracking error annualisé
    """
    return (returns_a - returns_b).std() * np.sqrt(periods_per_year)


def information_ratio(returns_a, returns_b, periods_per_year):
    """
    Calcule le ratio d'information
    
    Args:
        returns_a: Rendements du fonds
        returns_b: Rendements du benchmark
        periods_per_year: Nombre de périodes par an
    
    Returns:
        float: Information ratio
    """
    diff = returns_a - returns_b
    return diff.mean() / diff.std() * np.sqrt(periods_per_year)


def sharpe_ratio(returns, periods_per_year, rf=0):
    """
    Calcule le ratio de Sharpe
    
    Args:
        returns: Série de rendements
        periods_per_year: Nombre de périodes par an
        rf: Taux sans risque (par défaut 0)
    
    Returns:
        float: Ratio de Sharpe
    """
    excess = returns - rf/periods_per_year
    return excess.mean() / returns.std() * np.sqrt(periods_per_year)


def alpha_beta(returns_a, returns_b, periods_per_year):
    """
    Calcule l'alpha annualisé et le beta via régression linéaire
    
    Args:
        returns_a: Rendements du fonds
        returns_b: Rendements du benchmark
        periods_per_year: Nombre de périodes par an
    
    Returns:
        tuple: (alpha_annualisé, beta)
    """
    model = LinearRegression()
    X = returns_b.values.reshape(-1, 1)
    y = returns_a.values.reshape(-1, 1)
    model.fit(X, y)
    
    beta = model.coef_[0][0]
    alpha_weekly = model.intercept_[0]
    alpha_annualized = (1 + alpha_weekly)**periods_per_year - 1
    
    return alpha_annualized, beta


def rolling_alpha_beta(y, x, window):
    """
    Calcule l'alpha et le beta sur une fenêtre glissante
    
    Args:
        y: Rendements du fonds
        x: Rendements du benchmark
        window: Taille de la fenêtre
    
    Returns:
        DataFrame: Colonnes 'Alpha' et 'Beta'
    """
    alphas = []
    betas = []
    idx = []
    model = LinearRegression()
    
    for i in range(window, len(y)):
        y_win = y.iloc[i-window:i].values.reshape(-1, 1)
        x_win = x.iloc[i-window:i].values.reshape(-1, 1)
        
        if np.any(np.isnan(y_win)) or np.any(np.isnan(x_win)):
            alphas.append(np.nan)
            betas.append(np.nan)
        else:
            model.fit(x_win, y_win)
            betas.append(model.coef_[0][0])
            alphas.append(model.intercept_[0])
        
        idx.append(y.index[i])
    
    return pd.DataFrame({"Alpha": alphas, "Beta": betas}, index=idx)


def compute_drawdown(series):
    """
    Calcule le drawdown d'une série de prix
    
    Args:
        series: Série pandas de prix
    
    Returns:
        Series: Drawdown en pourcentage
    """
    cum_max = series.cummax()
    drawdown = (series / cum_max - 1) * 100
    return drawdown


def correlation_rolling(returns_a, returns_b):
    """
    Calcule la corrélation entre deux séries de rendements
    
    Args:
        returns_a: Première série de rendements
        returns_b: Deuxième série de rendements
    
    Returns:
        float: Coefficient de corrélation
    """
    return returns_a.corr(returns_b)
