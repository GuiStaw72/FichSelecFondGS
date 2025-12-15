"""
Configuration globale du projet
Contient uniquement les paramètres
"""
import logging
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# Paramètres de calcul
# WINDOW_SHORT = 30  # Fenêtre courte en semaines A supprimer ?
WINDOW = 52   # Fenêtre longue en semaines pour TE et Volatilité
PERIODS_PER_YEAR = 52  # Nombre de périodes par an (hebdomadaire)

# Configuration du logging
LOG_FORMAT = "%(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"
LOG_LEVEL = logging.WARNING

custom_theme = Theme({
    "log.time": "dim white",            # Color for time column
    "log.message": "dodger_blue1",      # Default message color
    "repr.number":"none",
    "repr.ellipsis":"none"
})

console=Console(theme=custom_theme)

def setup_logging():
    handler = RichHandler(
        markup=True,
        omit_repeated_times=False,
        console=console,
    )
    logging.basicConfig(
        level=LOG_LEVEL,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=[handler],
    )