"""
Module de chargement et préparation des données Excel
"""
import pandas as pd
import logging
import time
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from rich.logging import RichHandler
from rich import box
from config import console

logger = logging.getLogger(__name__)

def load_data(filename="Data.xlsx"):
    """
    Charge et prépare les données depuis un fichier Excel
    
    Args:
        filename: Nom du fichier Excel à charger
    
    Returns:
        DataFrame pandas avec les données préparées
    """
    start = time.time()
    logger.info(f"Chargement du fichier '{filename}'")
    
    try:
        xls = pd.ExcelFile(filename)
    except FileNotFoundError:
        logger.error(f"Fichier '{filename}' introuvable")
        return None
    except Exception as e:
        logger.error(f"Erreur lors de l'ouverture du fichier : {e}")
        return None
    
    # Sélection de l'onglet de données
    sheet_data = "NAV" if "NAV" in xls.sheet_names else xls.sheet_names[0]
    console.print(f"[green]✓[/green] Onglet sélectionné : [green]'{sheet_data}'[/green]")
    logger.info(f"Onglet sélectionné : '{sheet_data}'")
    
    # Chargement des données
    df = pd.read_excel(xls, sheet_name=sheet_data, index_col=0)
    df = df.iloc[1:]  # Suppression de la première ligne
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.sort_index()
    
    elapsed = time.time() - start
    console.print(f"[green]✓[/green] Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes [dim]({elapsed:.2f}s)[/dim]")
    logger.info(f"Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes ({elapsed:.2f}s)")
    
    # Renommage des colonnes
    df = rename_columns(df, xls)
    
    return df


def rename_columns(df, xls):
    """
    Renomme les colonnes du DataFrame
    
    Args:
        df: DataFrame à renommer
        xls: Objet ExcelFile pour accéder aux onglets
    
    Returns:
        DataFrame avec colonnes renommées
    """
    logger.info("Renommage des colonnes...")
    
    # Génération de noms par défaut
    cols = df.columns.tolist()
    default_names = ['Benchmark'] + [f'Fonds {i}' for i in range(1, len(cols))]
    df.columns = default_names
    
    # Tentative de chargement depuis l'onglet "Name"
    if "Name" in xls.sheet_names:
        try:
            df_names = pd.read_excel(xls, sheet_name="Name", header=None)
            name_map = df_names.iloc[0].tolist()
            
            if len(name_map) == len(df.columns):
                df.columns = name_map
                
                # Affichage des noms chargés
                table = Table(title="Colonnes détectées", title_style="bold", box=box.ROUNDED, border_style="steel_blue3",header_style="bold")
                table.add_column("Position", justify="center", style="steel_blue3")
                table.add_column("Nom", style="white")
                
                for i, name in enumerate(name_map):
                    table.add_row(str(i+1), name)
                
                console.print()
                console.print(table)
                console.print("[green]✓[/green] Colonnes renommées automatiquement depuis l'onglet 'Name'")
                logger.info("Colonnes renommées depuis l'onglet 'Name'")
                return df
            else:
                logger.warning(f"Nombre de colonnes incompatible (attendu: {len(df.columns)}, trouvé: {len(name_map)})")
        except Exception as e:
            logger.warning(f"Erreur lors de la lecture de l'onglet 'Name' : {e}")
    
    # Mode manuel si nécessaire
    console.print("\n[bold cyan]🔧 Mode de renommage manuel activé[/bold cyan]")
    console.print("[dim]Appuyez sur Entrée pour conserver le nom par défaut[/dim]\n")
    
    new_names = []
    for col in df.columns:
        new_name = Prompt.ask(
            f"Nouveau nom pour [bold cyan]{col}[/bold cyan]",
            default=col
        ).strip()
        new_names.append(new_name if new_name else col)
    
    df.columns = new_names
    
    # Affichage du résumé
    table = Table(title="Colonnes finales", box=box.ROUNDED, border_style="steel_blue3")
    table.add_column("Position", justify="center", style="steel_blue3")
    table.add_column("Nom", style="white")
    
    for i, name in enumerate(new_names):
        table.add_row(str(i+1), name)
    
    console.print()
    console.print(table)
    logger.info(f"✓ Colonnes renommées : {', '.join(new_names)}")
    
    return df


def get_benchmark_and_funds(df):
    """
    Identifie la colonne benchmark et les colonnes de fonds
    
    Args:
        df: DataFrame contenant les données
    
    Returns:
        tuple: (nom_colonne_benchmark, liste_colonnes_fonds)
    """
    benchmark_col = df.columns[0]
    fonds_cols = df.columns[1:].tolist()
    return benchmark_col, fonds_cols

def load_comm(filename="Data.xlsx"):
    """"
    Charge les commentaires pour la matrice d'analyse force/faiblesse...
    """

    start = time.time()
    logger.info(f"Chargement du fichier '{filename}'")

    try:
        xls = pd.ExcelFile(filename)
    except FileNotFoundError:
        logger.error(f"Fichier '{filename}' introuvable")
        return None
    except Exception as e:
        logger.error(f"Erreur lors de l'ouverture du fichier : {e}")
        return None
    
    sheet_data = "Comm"
    console.print(f"[green]✓[/green] Onglet sélectionné : [green]'{sheet_data}'[/green]")
    logger.info(f"Onglet sélectionné : '{sheet_data}'")

    df = pd.read_excel(xls, sheet_name=sheet_data,usecols="A:D",nrows=5)
    elapsed = time.time() - start
    console.print(f"[green]✓[/green] Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes [dim]({elapsed:.2f}s)[/dim]")
    logger.info(f"Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes ({elapsed:.2f}s)")

    return df

def load_synth(filename="Data.xlsx"):
    """"
    Charge la synthese...
    """

    start = time.time()
    logger.info(f"Chargement du fichier '{filename}'")

    try:
        xls = pd.ExcelFile(filename)
    except FileNotFoundError:
        logger.error(f"Fichier '{filename}' introuvable")
        return None
    except Exception as e:
        logger.error(f"Erreur lors de l'ouverture du fichier : {e}")
        return None
    
    sheet_data = "Comm"
    console.print(f"[green]✓[/green] Onglet sélectionné : [green]'{sheet_data}'[/green]")
    logger.info(f"Onglet sélectionné : '{sheet_data}'")

    df = pd.read_excel(xls, sheet_name=sheet_data,usecols="A",skiprows=6, nrows=2)
    elapsed = time.time() - start
    console.print(f"[green]✓[/green] Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes [dim]({elapsed:.2f}s)[/dim]")
    logger.info(f"Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes ({elapsed:.2f}s)")

    return "Synthèse : " + df['Synthese'][0]

def load_info(filename="Data.xlsx"):

    start = time.time()
    logger.info(f"Chargement du fichier '{filename}'")

    try:
        xls = pd.ExcelFile(filename)
    except FileNotFoundError:
        logger.error(f"Fichier '{filename}' introuvable")
        return None
    except Exception as e:
        logger.error(f"Erreur lors de l'ouverture du fichier : {e}")
        return None
    
    sheet_data = "InfoFonds"
    console.print(f"[green]✓[/green] Onglet sélectionné : [green]'{sheet_data}'[/green]")
    logger.info(f"Onglet sélectionné : '{sheet_data}'")

    df = pd.read_excel(xls, sheet_name=sheet_data, usecols="A:B", skiprows=3,nrows=44)
    elapsed = time.time() - start
    console.print(f"[green]✓[/green] Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes [dim]({elapsed:.2f}s)[/dim]")
    logger.info(f"Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes ({elapsed:.2f}s)")

    return df