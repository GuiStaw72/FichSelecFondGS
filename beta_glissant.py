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