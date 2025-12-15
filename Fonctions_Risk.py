# Import des librairies
import pandas as pd

#  correlation glissante

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

#  beta glissant


def beta_glissant_vectorise(df, window=52):
    actif_mean = df['Actif'].rolling(window).mean()
    marche_mean = df['Marche'].rolling(window).mean()
    # Covariance glissante
    cov = ((df['Actif'] - actif_mean) * (df['Marche'] - marche_mean)).rolling(window).mean()
     # Variance glissante du marché
    var_marche = ((df['Marche'] - marche_mean)**2).rolling(window).mean()
     # Bêta glissant
    beta = cov / var_marche
    return beta.dropna()


#   def detect_drawdowns(date, perf):

def detect_drawdowns(date, perf):
    """
    Détecte les drawdowns d'une série de performances.
    Retourne un DataFrame avec :
    - Date de début
    - Amplitude (%)
    - Durée jusqu’au minimum (jours)
    - Durée totale (jours) (jusqu'à recovery)
    """
    df = pd.DataFrame({'DATE': pd.to_datetime(date), 'FONDS': perf})
    df = df.sort_values('DATE').reset_index(drop=True)

    peak = df['FONDS'][0]
    peak_date = df['DATE'][0]
    drawdowns = []

    in_drawdown = False
    start_date = None
    min_value = None
    min_date = None

    for i in range(1, len(df)):
        value = df.loc[i, 'FONDS']
        date = df.loc[i, 'DATE']

        if value > peak:
            if in_drawdown:
                # Fin du drawdown
                recovery_date = date
                duration_to_min = (min_date - start_date).days
                total_duration = (recovery_date - start_date).days
                amplitude = (peak - min_value) / peak * 100
                drawdowns.append({
                    'Date de début': start_date,
                    'Amplitude (%)': round(amplitude, 2),
                    'Durée jusqu’au minimum (jours)': duration_to_min,
                    'Durée totale (jours)': total_duration
                })
                in_drawdown = False

            # Nouveau sommet
            peak = value
            peak_date = date
        else:
            if not in_drawdown:
                # Début d’un nouveau drawdown
                in_drawdown = True
                start_date = peak_date
                min_value = value
                min_date = date
            elif value < min_value:
                # Mise à jour du minimum
                min_value = value
                min_date = date

    return pd.DataFrame(drawdowns)
