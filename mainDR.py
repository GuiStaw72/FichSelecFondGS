"""
Script principal d'analyse de fonds d'investissement
Génère un rapport PDF personnalisé basé sur les choix de l'utilisateur
"""
import time
import logging
from matplotlib.backends.backend_pdf import PdfPages

from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.text import Text
from rich import box, inspect

from data_loader import load_data
from analysis_modules import (
    performance_analysis,
    volatility_tracking_analysis,
    drawdown_statistics_analysis,
    multi_fund_comparison_analysis
)
from config import setup_logging, console

logger = logging.getLogger(__name__)


# Définition des analyses disponibles
AVAILABLE_ANALYSES = {
    1: {
        'name': 'Performance & Volatilité',
        'description': 'Performance, volatilité glissante, tracking error, alpha/beta',
        'function': performance_analysis
    },
    2: {
        'name': 'Volatilité & Tracking Error CT/LT',
        'description': 'Analyse approfondie de la volatilité et du tracking error',
        'function': volatility_tracking_analysis
    },
    3: {
        'name': 'Statistiques Avancées',
        'description': 'Drawdown, récapitulatif statistiques, test significativité',
        'function': drawdown_statistics_analysis
    },
    4: {
        'name': 'Comparaison Univers',
        'description': 'Performance relative, corrélations, scatter vol/perf',
        'function': multi_fund_comparison_analysis
    }
}

def print_header():
    """Affiche l'en-tête stylisé du programme"""
    title = Text()
    title.append("📊 ", style="bold steel_blue3")
    title.append("FICHE RISQUE", style="bold steel_blue3")
    title.append(" 📈", style="bold steel_blue3")
    
    console.print()
    console.print(Panel(
        title,
        box=box.DOUBLE,
        border_style="steel_blue3",
        padding=(1, 2),
        expand=False
    ))
    console.print()


def select_analyses():
    """Permet à l'utilisateur de sélectionner les analyses à inclure"""
    
    # Création du tableau des analyses disponibles
    table = Table(
        title="Analyses Disponibles",
        title_style="bold",
        box=box.ROUNDED,
        border_style="steel_blue3",
        show_header=True,
        header_style="bold"
    )
    
    table.add_column("N°", justify="center", style="steel_blue3", width=5)
    table.add_column("Nom de l'analyse", style="white")
    table.add_column("Description", style="dim white", width=50)
    
    for key, info in AVAILABLE_ANALYSES.items():
        table.add_row(
            str(key),
            f"{info['name']}",
            info['description']
        )
    
    console.print(table)
    console.print()
    
    # Instructions
    instructions = Table.grid(padding=(0, 2))
    instructions.add_column(style="white")
    instructions.add_column(style="white")
    
    instructions.add_row("💡", "Entrez les numéros séparés par des virgules (ex: 1,2,4)")
    instructions.add_row("✨", "Tapez 'all' pour sélectionner toutes les analyses (par défaut)")
    instructions.add_row("🚪", "Tapez 'quit' pour annuler")
    
    console.print(Panel(instructions, title="[bold steel_blue3]Options[/bold steel_blue3]", border_style="steel_blue3",expand=False))
    console.print()
    
    while True:
        choice = Prompt.ask(
            "[sea_green3]Votre choix[/sea_green3]",
            default="all"
        ).strip().lower()
        
        if choice == 'quit':
            console.print("[yellow]⚠️  Opération annulée[/yellow]")
            return None
        
        if choice == 'all':
            console.print("[green]✓[/green] Toutes les analyses sélectionnées!")
            return list(AVAILABLE_ANALYSES.keys())
        
        try:
            selected = [int(x.strip()) for x in choice.split(',')]
            if all(s in AVAILABLE_ANALYSES for s in selected):
                console.print(f"[green]✓[/green] {len(selected)} analyse(s) sélectionnée(s)!")
                return selected
            else:
                console.print("[red]❌ Numéros invalides. Réessayez.[/red]")
        except ValueError:
            console.print("[red]❌ Format invalide. Utilisez des numéros séparés par des virgules.[/red]")


def generate_report(df, selected_analyses, output_file="reporting_fonds.pdf"):
    """Génère le rapport PDF avec les analyses sélectionnées"""
    
    console.print()
    console.print(Panel(
        f"[bold steel_blue3]Génération du rapport avec {len(selected_analyses)} analyse(s)[/bold steel_blue3]",
        border_style="steel_blue3",expand=False
    ))
    console.print()
    
    figures = []
    
    # Table de résultats
    results_table = Table(
        title="Analyses Réalisées",
        title_style="bold",
        box=box.ROUNDED,
        border_style="steel_blue3",
        show_header=True,
        header_style="bold"
    )
    results_table.add_column("Analyse", style="white", width=35)
    results_table.add_column("Statut", justify="center", width=15)
    results_table.add_column("Temps", justify="center", style="white", width=15)
    
    for analysis_id in selected_analyses:
        analysis_info = AVAILABLE_ANALYSES[analysis_id]
        with console.status(f"[steel_blue3]📊 {analysis_info['name']} en cours..."):
            start_time = time.time()
            try:
                fig = analysis_info['function'](df)
                figures.append(fig)
                elapsed = time.time() - start_time
                logger.info(f"✓ {analysis_info['name']} terminée en {elapsed:.2f}s")
                results_table.add_row(
                    analysis_info['name'],
                    "[green]✓[/green] Réussi",
                    f"{elapsed:.2f}s"
                )
            except Exception as e:
                logger.error(f"✗ Erreur dans {analysis_info['name']}: {e}", exc_info=True)
                results_table.add_row(
                    analysis_info['name'],
                    "[red]✗[/red] Échec",
                    "-"
                )
    
    console.print()
    console.print(results_table)
    console.print()
        
    # Sauvegarde dans le PDF
    if figures:
        with console.status(f"💾 Sauvegarde dans {output_file}...") as status:
            with PdfPages(output_file) as pdf:
                for fig in figures:
                    pdf.savefig(fig)
        
        console.print(f"[green]✓[/green] Rapport sauvegardé : {output_file}")
    else:
        console.print("[yellow]⚠️  Aucune figure à sauvegarder[/yellow]")
    
    return len(figures)


def main():
    """Point d'entrée principal"""
    setup_logging()
    start_time = time.time()
    
    print_header()
    
    try:
        # Chargement des données
        console.print(Panel(
        "[bold steel_blue3]Chargement des données[/bold steel_blue3]",
        border_style="steel_blue3",expand=False
        ))
        console.print()
        with console.status("[bold steel_blue3]📂 Chargement des données...\n"):
            df = load_data("Data.xlsx")
        
        if df is None:
            logger.error("[red]❌ Impossible de charger les données. Arrêt du programme.[/red]")
            return
        
        console.print("[green]✓[/green] Données chargées avec succès!")
        console.print()
        
        # Sélection des analyses
        selected = select_analyses()
        
        if selected is None:
            console.print("[yellow]👋 Au revoir![/yellow]")
            return
        
        # Génération du rapport
        num_analyses = generate_report(df, selected)
        
        # Résumé final
        total_time = time.time() - start_time
        
        summary = Table.grid(padding=(0, 2))
        summary.add_column(style="bold steel_blue3", justify="right")
        summary.add_column(style="white")
        
        summary.add_row("Analyses générées:", f"[bold white]{num_analyses}[/bold white]")
        summary.add_row("Temps total:", f"[bold white]{total_time:.2f}s[/bold white]")
        summary.add_row("Fichier PDF:", "[bold white]reporting_fonds.pdf[/bold white]")
        
        console.print()
        console.print(Panel(
            summary,
            title="[bold steel_blue3]✨ Terminé avec succès! ✨[/bold steel_blue3]",
            border_style="steel_blue3",
            box=box.DOUBLE,
            expand=False
        ))
        console.print()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Interruption par l'utilisateur[/yellow]")
    except Exception as e:
        logger.error(f"Erreur critique : {e}", exc_info=True)


if __name__ == "__main__":
    main()