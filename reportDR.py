from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from utils_report import draw_fig_keep_ratio, get_best_font_size
import tempfile
import time
import logging
import os
import pandas as pd
from rich.console import Console
from rich.prompt import Prompt
from rich.logging import RichHandler
from rich import box
from pathlib import Path
from PIL import ImageGrab
from io import BytesIO
from data_loader import load_data, load_comm, load_info, load_synth
from config import setup_logging, console
from analysis_modules import plot_performance_absolute_relative,\
    plot_alpha_beta_rolling, plot_volatility, plot_tracking_error,\
    plot_statistics_table, plot_drawdown,plot_returns_distribution,\
    plot_alpha_tests, plot_relative_performance_multifund,\
    plot_volatility_vs_performance,plot_correlation_heatmap

logger = logging.getLogger(__name__)

def create_report(df, df2, df3, s4, logo_path, output_path="fiche_fonds.pdf"):
    """
    Fonction principale pour générer le pdf avec ReportLab
    Prend 3 dataframes en entrées :
    Le premier dataframe df1 contient les NAV de tous les fonds et benchmarks
    Le second dataframe df2 contient les infos principalement données par Bloomberg
    Le troisieme dataframe df3 contient les commentaires de la matrice force/faiblesse/opportunité
    S4 est simplement un paragraphe qui contient la synthese a insérer

    logo_path est le chemin pour le logo EDF

    output_path est le chemin nom du rapport de sortie
    """
    start = time.time()
    logger.info(f"Création du pdf '{output_path}'")

    console.print("Initiation du pdf")
    c = canvas.Canvas(output_path, pagesize=A4) #Création du canvas principal
    W, H = A4  # 595 x 842 points approx.
    BleuEDF=16/256, 87/256, 200/256 #Couleurs custom d'EDF
    BleuEDFdark=0/256, 26/256, 112/256
    
    #Premiere Page
    #Logo EDF
    c.drawImage(logo_path,x=25,y=H-128, width=140, height=140, preserveAspectRatio=True)

    # Titre de la fiche
    Funds_name=df.columns[1]
    c.setFont("Helvetica-Bold", 16)
    c.setFillColorRGB(*BleuEDF)
    c.drawString(150, H - 50, "EDF Gestion - Fiche Risque")
    c.setFont("Helvetica-Bold", 20)
    c.drawString(150, H - 80, Funds_name)

    #Tableau Excel / Copie les informations formatées du tableau excel avec les requetes BLG
    styles = getSampleStyleSheet()

    colWidths = [120,120]
    rowHeights = [8.5]*44

    #style pour les cases normales
    ist = styles['BodyText'].clone('infos')
    ist.fontName = "Helvetica"
    ist.fontSize= 4
    ist.textColor=colors.black

    #style pour les case en-tetes
    ist2 = styles['BodyText'].clone('infos2')
    ist2.fontName = "Helvetica-Bold"
    ist2.fontSize= 4
    ist2.textColor = colors.white

    #style pour les cases en gras
    list_gras= [25,30,36,40]
    ist3 = styles['BodyText'].clone('infos3')
    ist3.fontName = "Helvetica-Bold"
    ist3.fontSize = 4
    ist3.textColor = BleuEDF

    data = [[Paragraph(str(col), ist2) for col in df2.columns]]
    for i, row in enumerate(df2.itertuples(index=False),start=0):
        row_cells = []
        for j, cell in enumerate(row):
            if pd.isna(cell):
                text=""
                if j==0:
                    rowHeights[i+1]=2
            else:
                text=str(cell)
            if i in list_gras:
                row_cells.append(Paragraph(text, ist3))
            else:
                row_cells.append(Paragraph(text, ist))
        data.append(row_cells)

    table = Table(data, colWidths=colWidths, rowHeights=rowHeights)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), BleuEDF),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING',(0,0),(-1,-1),1.5),
        ('SPAN', (0,0), (1,0)),
        ('SPAN', (0,26), (1,26)),
        ('SPAN', (0,31), (1,31)),
        ('SPAN', (0,37), (1,37)),
        ('SPAN', (0,41), (1,41)),
    ]))

    table.wrapOn(c, 0, 0)
    table.drawOn(c, 40, 380)
    elapsed = time.time() - start
    console.print(f"[green]✓[/green] Tableau chargé et formaté [dim]({elapsed:.2f}s)[/dim]")

    # Figures matplotlib
    start = time.time()
    fig1 = plot_performance_absolute_relative(df,figsize=(7.5,4.5))
    elapsed = time.time() - start
    console.print(f"[green]✓[/green] Graphique performance [dim]({elapsed:.2f}s)[/dim]")

    start = time.time()
    fig2 = plot_alpha_beta_rolling(df,figsize=(7.5,4.5))
    elapsed = time.time() - start
    console.print(f"[green]✓[/green] Graphique alpha [dim]({elapsed:.2f}s)[/dim]")

    start = time.time()
    fig3 = plot_volatility(df,figsize=(6,4.5))
    elapsed = time.time() - start
    console.print(f"[green]✓[/green] Graphique volatilité [dim]({elapsed:.2f}s)[/dim]")

    start = time.time()
    fig4 = plot_tracking_error(df,figsize=(6,4.5))
    elapsed = time.time() - start
    console.print(f"[green]✓[/green] Graphique Tracking-error [dim]({elapsed:.2f}s)[/dim]")

    start = time.time()
    fig5 = plot_statistics_table(df,figsize=(7,5.5))
    elapsed = time.time() - start
    console.print(f"[green]✓[/green] Table statistiques [dim]({elapsed:.2f}s)[/dim]")

    start = time.time()
    fig6 = plot_drawdown(df,figsize=(6,4.5))
    elapsed = time.time() - start
    console.print(f"[green]✓[/green] Graphique Drawdown [dim]({elapsed:.2f}s)[/dim]")

    start = time.time()
    fig7 = plot_returns_distribution(df,figsize=(6,4.5))
    elapsed = time.time() - start
    console.print(f"[green]✓[/green] Graphique Distribution [dim]({elapsed:.2f}s)[/dim]")

    start = time.time()
    fig8 = plot_alpha_tests(df,figsize=(6,4.5))
    elapsed = time.time() - start
    console.print(f"[green]✓[/green] Calcul KPSS [dim]({elapsed:.2f}s)[/dim]")

    # Placement absolu figures
    draw_fig_keep_ratio(c, fig1, x=320, y=580, width_pt=250)
    draw_fig_keep_ratio(c, fig2, x=320, y=400, width_pt=250)
    draw_fig_keep_ratio(c, fig3, x=35, y=230, width_pt=180)
    draw_fig_keep_ratio(c, fig4, x=225, y=230, width_pt=180)
    draw_fig_keep_ratio(c, fig5, x=405, y=230, width_pt=180)
    draw_fig_keep_ratio(c, fig6, x=35, y=40, width_pt=180)
    draw_fig_keep_ratio(c, fig7, x=225, y=40, width_pt=180)
    draw_fig_keep_ratio(c, fig8, x=415, y=40, width_pt=180)

    c.showPage()

    #2nde page

    #Tableau Commentaires
    start = time.time()
    styles = getSampleStyleSheet()

    colWidths = [75,150,150,150]
    rowHeights = [25,60,60,60,60]

    data = []
    headers = df3.columns.tolist()

    # Ajout des headers
    header_row = []
    for j, col in enumerate(headers):
        if j == 0:
            header_row.append(Paragraph("", styles['BodyText']))  # Case (0,0) vide
        else:
            hd = styles['BodyText'].clone('headers')
            hd.fontName = "Helvetica"
            hd.textColor=colors.white
            header_row.append(Paragraph(str(col), hd))
    data.append(header_row)

    for i, row in enumerate(df3.itertuples(index=False), start=1):
        row_cells = []
        for j, cell in enumerate(row):
            text = str(cell).replace("\n", "<br/>")

            if j == 0:
                st = styles['BodyText'].clone('firstcol')
                st.fontName = "Helvetica"
                st.textColor=colors.white
                row_cells.append(Paragraph(text, st))
                continue
            # Calcul de taille police optimale
            font_size = get_best_font_size(text,"Helvetica",colWidths[j],
                rowHeights[i] if i < len(rowHeights) else rowHeights[-1]
            )

            st = styles['BodyText'].clone('cellstyle')
            st.fontSize = font_size
            st.fontName = "Helvetica"
            row_cells.append(Paragraph(text, st))
        data.append(row_cells)

    table = Table(data, colWidths=colWidths, rowHeights=rowHeights)

    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), BleuEDF),
        ('BACKGROUND', (0, 0), (-1, 0), BleuEDFdark),
        ('VALIGN', (0, 0), (-1,-1), 'MIDDLE'),
        ('VALIGN', (1, 1), (4, 3), 'TOP'),
        ('GRID', (0, 0), (-1,-1), 1, colors.black),
    ]))

    table.wrapOn(c, 0, 0)
    table.drawOn(c, 30, 560)
    elapsed = time.time() - start
    console.print(f"[green]✓[/green] Tableau de commentaires [dim]({elapsed:.2f}s)[/dim]")

    # Figure matplotlib
    start = time.time()
    fig9 = plot_relative_performance_multifund(df,figsize=(22,10))
    elapsed = time.time() - start
    console.print(f"[green]✓[/green] Graphique multi-fonds [dim]({elapsed:.2f}s)[/dim]")

    start = time.time()
    fig10 = plot_volatility_vs_performance(df,figsize=(10,20/3))
    elapsed = time.time() - start
    console.print(f"[green]✓[/green] Graphique perf/vol [dim]({elapsed:.2f}s)[/dim]")

    start = time.time()
    fig11 = plot_correlation_heatmap(df,figsize=(8,6))
    elapsed = time.time() - start
    console.print(f"[green]✓[/green] Corrélation perfs relatives [dim]({elapsed:.2f}s)[/dim]")

    # Placement absolu figures
    draw_fig_keep_ratio(c, fig9, x=35, y=310, width_pt=525)
    draw_fig_keep_ratio(c, fig10, x=35, y=100, width_pt=300)
    draw_fig_keep_ratio(c, fig11, x=350, y=100, width_pt=220)

    # Placement de la synthese
    stsy = styles["BodyText"].clone('synth')
    p = Paragraph(s4, stsy)

    t = Table([[p]], colWidths=[550])
    t.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.5, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    t.wrapOn(c, 0, 0)
    t.drawOn(c, 30, 30)

    c.showPage()

    c.save()

    console.print("Fiche générée")

if __name__ == "__main__":
    df = load_data("Data.xlsx")
    df2 = load_info("Data.xlsx")
    df3 = load_comm("Data.xlsx")
    s4=load_synth("Data.xlsx")
    lp=os.path.join("ressources","logo_EDF_RVB_2025.png")
    create_report(df, df2, df3, s4, logo_path=lp, output_path="rapport_test.pdf",)
    # input_xlsx =Path("Infos.xlsx")
    # sheet = "InfoFonds"           # ou nom exact de la feuille
    # rng = "A4:B47"             # plage à copier
    # output_png = Path("test.png")

    # saved = excel_range_to_png_via_chart(input_xlsx, sheet, rng, output_png, visible=False)
    # print("Image sauvegardée :", saved)