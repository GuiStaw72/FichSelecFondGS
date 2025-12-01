# Import des librairies
import pandas as pd
def correlation_glissante_df(df: pd.DataFrame, col_x: str, col_y: str, window: int) -> pd.Series:
    """
    Calcule la covariance glissante entre deux colonnes d'un DataFrame sur une fenêtre donnée.
    
    :param df: DataFrame contenant les données
    :param col_x: nom de la colonne X
    :param col_y: nom de la colonne Y
    :param window: taille de la fenêtre glissante
    :return: Série pandas de la covariance glissante
    """
    x = df[col_x]
    y = df[col_y]
    x_mean = x.rolling(window).mean()
    y_mean = y.rolling(window).mean()
    cov = ((x - x_mean) * (y - y_mean)).rolling(window).mean()
    
    # Écarts-types glissants
    std_actif = ((x - x_mean)**2).rolling(window).mean().pow(0.5)
    std_marche = ((y - y_mean)**2).rolling(window).mean().pow(0.5)

     # Corrélation glissante
    correlation = cov / (std_actif * std_marche)

    return correlation