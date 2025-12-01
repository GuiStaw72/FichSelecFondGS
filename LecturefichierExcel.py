# Import des librairies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#import fonction norm pour distribution de probabilite
from scipy.stats import norm

#import fonction statistique pour calcul regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#import des fonctions librairie matplotlib
from matplotlib.ticker import PercentFormatter
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter

#import des fonction des autres fichiers 
from corr_glissante import correlation_glissante_df
from beta_glissant import beta_glissant_vectorise
from drawdown import detect_drawdowns


# Charger le fichier Excel
fichier_excel = "C:/Users/a67447/PYTHON/LectureDNCAARCHER.xlsx"
df = pd.read_excel(fichier_excel,sheet_name='Sheet1', engine='openpyxl')


# Compte le nombre de colonnes contenant au moins une valeur non vide
colonnes_remplies = df.apply(lambda col: col.notna().any()).sum()
print(f"Nombre de colonnes remplies avec des séries de données : {colonnes_remplies}")


# Lire deux colonnes spécifiques (par exemple 'Colonne1' et 'Colonne2')
date = pd.Series(df.iloc[0:, 0].values)
fond = pd.Series(df.iloc[0:, 1].values)
bmk_fond = pd.Series(df.iloc[0:, 2].values)

# Afficher les données
print("date :")
print(date)

print("\nfond :")
print(fond)

print("\nbmk_fond :")
print(bmk_fond)

7
# Conversion des series journalières en serie hebdo


df = pd.DataFrame({
    'date': pd.to_datetime(date),
    'fond': fond,
    'bmk_fond': bmk_fond
})



# Définir la priorité des jours de la semaine
# Lundi=0, Mardi=1, Mercredi=2, Jeudi=3, Vendredi=4
weekday_priority = {2: 0, 3: 1, 1: 2, 4: 3, 0: 4}  # mercredi > jeudi > mardi > vendredi > lundi

# Ajouter les colonnes nécessaires
df['weekday'] = df['date'].dt.weekday
df['priority'] = df['weekday'].map(weekday_priority)
df['year'] = df['date'].dt.isocalendar().year
df['week'] = df['date'].dt.isocalendar().week

# Trier par priorité et sélectionner la première valeur par semaine
df_sorted = df.sort_values(by=['year', 'week', 'priority'])
df_weekly = df_sorted.groupby(['year', 'week']).first().reset_index()

# Extraire les séries hebdomadaires
fond_weekly = df_weekly['fond']
bmk_fond_weekly = df_weekly['bmk_fond']
date_weekly = df_weekly['date']

# Affichage des résultats
#print("Dates retenues par semaine:")
#print(date_weekly)
#print("\nSérie fond hebdomadaire:")
#print(fond_weekly)
#print("\nSérie bmk_fond hebdomadaire:")
#print(bmk_fond_weekly)


# Calcul de la volatilite 52 s
# Calcul de la volatilité



# Calcul des rendements hebdomadaires
rdt_fond_weekly = fond_weekly.pct_change().dropna()
rdt_bmk_fond_weekly = bmk_fond_weekly.pct_change().dropna()
ecart_rdt_fond_weekly = rdt_fond_weekly - rdt_bmk_fond_weekly 



# Volatilité sur 52 semaines glissantes
vol_fond_weekly_52s = rdt_fond_weekly.rolling(window=52).std()* np.sqrt(52)
vol_bmk_fond_weekly_52s = rdt_bmk_fond_weekly.rolling(window=52).std()* np.sqrt(52)
TE_fond_weekly_52s = ecart_rdt_fond_weekly.rolling(window=52).std()* np.sqrt(52)

#################################
# Calcul de la performance 
################################
perf_fond_weekly = np.concatenate([[100], (1 + rdt_fond_weekly).cumprod() * 100])
perf_bmk_fond_weekly= np.concatenate([[100], (1 + rdt_bmk_fond_weekly).cumprod() * 100])

data_weekly = pd.DataFrame({
    'date_weekly': date_weekly,
    'rdt_fond_weekly': rdt_fond_weekly,
    'rdt_bmk_fond_weekly': rdt_bmk_fond_weekly,
    'ecart_rdt_fond_weekly': ecart_rdt_fond_weekly,
    'perf_fond_weekly' : perf_fond_weekly,
    'perf_bmk_fond_weekly' : perf_bmk_fond_weekly
})

#print("Dates retenues par semaine:")
#print(date_weekly)
#print("\nPerfromancd fond hebdomadaire:")
#print(perf_fond_weekly)
#print("\nPerformance bmk_fond hebdomadaire:")
#print(perf_bmk_fond_weekly)


#################################
# Calcul régression linéaire 
################################

Reg_X = rdt_fond_weekly.values.reshape(-1, 1)
Reg_Y = rdt_bmk_fond_weekly.values

# Régression linéaire
model = LinearRegression()
model.fit(Reg_X, Reg_Y)

a = model.coef_[0]   # pente
b = model.intercept_ # ordonnée à l'origine


# Prédictions et calcul du R²
predictions = model.predict(Reg_X)
r2 = r2_score(Reg_Y, predictions)


# Affichage de l'équation sur le graphique
equation = f"y = {a:.4f}x + {b:.4f}"





print(f"Taille originale date : {len(date_weekly )}")
print(f"Taille originale fond : {len(vol_fond_weekly_52s )}")
print(f"Taille originale bmk_fond : {len(vol_bmk_fond_weekly_52s )}")


#################################
# Calcul Information Ratio
################################

perf_relat_52s = []

# On s'assure de ne pas dépasser la longueur de la série
for i in range(len(perf_fond_weekly) - 55):
    start = perf_fond_weekly[4 + i]
    end = perf_fond_weekly[55 + i]
    if start != 0:
        perf_relat_52s.append((end - start) / start)
    else:
        perf_relat_52s.append(None)  # ou np.nan pour éviter division par zéro

# Convertir en série pandas si besoin
perf_relat_52s_series = pd.Series(perf_relat_52s)

IR_52s_ratio = perf_relat_52s_series /TE_fond_weekly_52s 
#################################
# Calcul Beta
################################

#Créer un DataFrame avec les dates
df_beta = pd.DataFrame({
    'Date': date_weekly,
    'Actif': rdt_fond_weekly,
    'Marche': rdt_bmk_fond_weekly
})


# Définir la date comme index
df_beta.set_index('Date', inplace=True)






# Calcul du bêta glissant
beta_glissant = beta_glissant_vectorise(df_beta)

# Affichage des 10 premiers résultats
print("Les 10 premiers bêta glissants (vectorisé) sur 52 semaines :")
print(beta_glissant.head(10))



#################################
# Calcul Correlation fond/Indice
################################
df_corr = pd.DataFrame({
    'Date': date_weekly,
    'Actif': rdt_fond_weekly,
    'Marche': rdt_bmk_fond_weekly
})

# Définir la date comme index
df_corr.set_index('Date', inplace=True)



# Calcul de la covariance glissante
corr_glissante = correlation_glissante_df(df_corr, 'Actif', 'Marche', window=52)

# Affichage
print("Les 10 premiers coefficients de correlation/indice glissants (vectorisé) sur 52 semaines :")
print(corr_glissante.dropna().head(10))

#######################################################
##### comparaison des distributions de probabilité
##########################################################
# Création de l'histogramme comparatif
from scipy.stats import norm  



mean_rdt_fond_weekly = np.mean(rdt_fond_weekly)
std_rdt_fond_weekly = np.std(rdt_fond_weekly)

# Définir les bornes de l'histogramme : [moyenne - 3*écart-type, moyenne + 3*écart-type]
lower_bound = mean_rdt_fond_weekly  - 3 * std_rdt_fond_weekly 
upper_bound = mean_rdt_fond_weekly  + 3 * std_rdt_fond_weekly 

# Créer 21 intervalles (donc 42 points de séparation)
bins = np.linspace(lower_bound, upper_bound, 22)

# Tracer la courbe de la loi normale
x = np.linspace(lower_bound, upper_bound, 500)
gaussian_curve = norm.pdf(x, mean_rdt_fond_weekly , std_rdt_fond_weekly)





#########################################################################
# Calcul des drawdowns absolus du fond 
#########################################################################

# Appel de la fonction drawdown
drawdown_abs = detect_drawdowns(date_weekly, perf_fond_weekly)

# Affichage du résultat
print("Les drawdowns du fonds sont  :")
print(drawdown_abs)


# fichier excel de sauvegarde

# label donnes entree

label_datajour=pd.DataFrame([["Donnees_jour"]])
label_dataweek=pd.DataFrame([["Donnees_week"]])

label_drawdownabs=pd.DataFrame([["Drawdown absolu du fond"]])

label_dataweekly=pd.DataFrame([["Ensemble de donnees hebdo"]])

with pd.ExcelWriter('donnees_int.xlsx', engine='openpyxl') as writer:
    label_datajour.to_excel(writer, sheet_name="donnees_jour", index=False, header=False, startrow=0, startcol=0)
    df.to_excel(writer, sheet_name='donnees_jour', index=False,startrow=1,startcol=0)

    label_dataweek.to_excel(writer, sheet_name="donnees_jour", index=False, header=False, startrow=0, startcol=10)
    df_weekly.to_excel(writer, sheet_name='donnees_jour', index=False, startrow=1,startcol=10)

    corr_glissante.to_excel(writer, sheet_name='Corrglissante', index=False)

    df_beta.to_excel(writer, sheet_name='Beta', index=False,startrow=1,startcol=0)
    beta_glissant.to_excel(writer, sheet_name='Beta', index=False, startrow=1,startcol=10)
    
    label_drawdownabs.to_excel(writer, sheet_name="Drawdown", index=False, header=False, startrow=0, startcol=0)
    drawdown_abs.to_excel(writer, sheet_name='Drawdown', index=False, startrow=1,startcol=0)
 
    label_dataweekly.to_excel(writer, sheet_name="Dataweek", index=False, header=False, startrow=0, startcol=0)
    data_weekly.to_excel(writer, sheet_name='Dataweek', index=False, startrow=1,startcol=0)



#########################################################################
# Generation des graphiques dans un PDF 
#########################################################################


# Formateur pour afficher en pourcentage
formatter = FuncFormatter(lambda y, _: f"{y * 100:.0f}%")
formatter1 = FuncFormatter(lambda y, _: f"{y * 100:.1f}%")



# Créer un fichier PDF

with PdfPages("graphiques_selection_fond.pdf") as pdf:

    
    
    ################################################
    # Graphique : Regression lineaire Indices/ Fonds 
    ################################################

    plt.figure(figsize=(10, 6))
    plt.scatter(rdt_fond_weekly, rdt_bmk_fond_weekly, color='blue', label='Données')
    plt.plot(rdt_fond_weekly, predictions, color='red', label='Régression linéaire')
    plt.xlabel('Rendements fond')
    plt.ylabel('Rendements becnhmark du fond')
    plt.title('Rendements hebdomadaires et régression linéaire')
    plt.text(
        Reg_X.min(), Reg_Y.max()-0.01,  # Position (tu peux ajuster)
        equation,
        fontsize=12, color='red'
    )
    plt.legend()
    plt.text(min(rdt_fond_weekly), max(rdt_bmk_fond_weekly), f"R² = {r2:.4f}", fontsize=12, color='green')
    plt.grid(True)
    plt.tight_layout()
    pdf.savefig()  # Sauvegarde le graphique dans le PDF
    plt.close()

    ###########################################
    # Graphique : Performance 
    ############################################
    plt.figure(figsize=(10, 6))
    plt.plot(date_weekly,perf_fond_weekly , label="Perf du fond")
    plt.plot(date_weekly,perf_bmk_fond_weekly , label="Perf du benchmark du fond")

    # Ajouter les étiquettes et le titre
    plt.xlabel("date")
    plt.ylabel("Performance base 100")
    plt.title('Performance fond et benchmark')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    pdf.savefig()  # Sauvegarde le graphique dans le PDF
    plt.close()

     ###########################################
    # Graphique : Information Ratio 
    ############################################
    plt.figure(figsize=(10, 6))
    plt.plot(date_weekly,IR_52s_ratio , label="IR du fond")
    

    # Ajouter les étiquettes et le titre
    plt.xlabel("date")
    plt.ylabel("IR")
    plt.title('Information Ratio du  fond')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    pdf.savefig()  # Sauvegarde le graphique dans le PDF
    plt.close()

    ###########################################
    # Graphique : Beta
    ############################################
    # Tracer le graphique avec les dates en abscisses
    plt.figure(figsize=(10, 6))
    plt.plot(beta_glissant.index, beta_glissant.values, label='Bêta glissant (52 semaines)', color='blue')
    plt.title('Évolution du bêta glissant sur 52 semaines')
    plt.xlabel('Date')
    plt.ylabel('Bêta')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    pdf.savefig()  # Sauvegarde le graphique dans le PDF
    plt.close()

 ###########################################
    # Graphique : Correlation fond/indice 
    ############################################
    # Tracer le graphique avec les dates en abscisses
    plt.figure(figsize=(10, 6))
    plt.plot(corr_glissante.index, corr_glissante.values, label='Correlation fond/indice glissant (52 semaines)', color='blue')
    plt.title('Évolution du coefficient correlation glissant sur 52 semaines')
    plt.xlabel('Date')
    plt.ylabel('Coefficient Correlation fond indice')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    pdf.savefig()  # Sauvegarde le graphique dans le PDF
    plt.close()
    ###########################################
    # Graphique : volatilité fond et benchmark 
    ############################################
    plt.figure(figsize=(10, 6))
    plt.plot(date_weekly.iloc[1:], vol_fond_weekly_52s, label="Volatilité du fond")
    plt.plot(date_weekly.iloc[1:], vol_bmk_fond_weekly_52s , label="Volatilité du benchmark du fond")

    # Ajouter les étiquettes et le titre
    plt.xlabel("Abscisse")
    plt.ylabel("Volatilité (%)")
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.title('Volatilité fond et benchmark')
    plt.legend()
    plt.grid(True)
    pdf.savefig()  # Sauvegarde le graphique dans le PDF
    plt.close()
    ###########################################
    # Graphique : Tracking Error
    ############################################
    plt.figure(figsize=(10, 6))
    plt.plot(date_weekly.iloc[1:], TE_fond_weekly_52s, label="TE du fond")
    # Ajouter les étiquettes et le titre
    plt.xlabel("Abscisse")
    plt.ylabel("TE (%)")
    plt.gca().yaxis.set_major_formatter(formatter1)
    plt.title('TE fond vs benchmark du fond')
    plt.legend()
    plt.grid(True)
    pdf.savefig()  # Sauvegarde le graphique dans le PDF
    plt.close()

    ###########################################
    # Graphique : distribution de probabilité
    ############################################
    plt.figure(figsize=(10, 6))
    plt.title('Comparaison des distributions de probabilité et courbe gaussienne du fond')
    plt.xlabel('Rendement hebdomadaire')
    #plt.hist(rdt_fond_weekly, bins=bins, density=True, alpha=0.6, color='blue', label='fond')
    #plt.hist(rdt_bmk_fond_weekly, bins=bins, density=True, alpha=0.6, color='orange', label='bmk_fond')
    #plt.plot(x, gaussian_curve, color='red', linewidth=2, label='Gaussienne (fond)')
    # Calcul des histogrammes
    fonds_hist, _ = np.histogram(rdt_fond_weekly, bins=bins, density=True)
    bmk_fond_hist, _ = np.histogram(rdt_bmk_fond_weekly, bins=bins, density=True)

    # Centres des classes
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bar_width = (bins[1] - bins[0]) * 0.4  # largeur réduite pour séparer les barres
    plt.bar(bin_centers - bar_width/2, fonds_hist, width=bar_width, label='FONDS', color='blue')
    plt.bar(bin_centers + bar_width/2, bmk_fond_hist, width=bar_width, label='Indice Fonds Mid Cap', color='orange')
    plt.plot(x, gaussian_curve, color='red', linewidth=2, label='Gaussienne (fond)')
    plt.legend()
    pdf.savefig()  # Sauvegarde le graphique dans le PDF
    plt.close()



# Sauvegarde
plt.savefig("distribution_rendements_cote_a_cote.png")



