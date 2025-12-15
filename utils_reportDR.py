from io import BytesIO
import matplotlib.pyplot as plt
from pathlib import Path
from win32com.client import gencache
from win32com.client import constants
import time
from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth

def fig_to_img_buffer(fig):
    """
    Copie une figure de matplotlib dans le buffer de sauvegarde,
    permettant ensuite de directement l'intégré dans un pdf sans la sauvegarder
    """
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer

def excel_range_to_png(xlsx_path: str, sheet_name: str, excel_range: str, out_path: str,
                       visible: bool = False, wait_after_paste: float = 0.2,
                       upscale_factor: float = 2.0):
    """
    Inutile a priori, utilisé précédemment pour faire des copies d'écran des fichiers excel
    """
    xlsx_path = str(Path(xlsx_path).resolve())
    out_path = Path(out_path).resolve()
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    excel = gencache.EnsureDispatch("Excel.Application")
    excel.Visible = bool(visible)
    excel.DisplayAlerts = False

    wb = excel.Workbooks.Open(xlsx_path)

    try:
        try:
            ws = wb.Worksheets(sheet_name)
        except Exception:
            ws = wb.ActiveSheet

        rng = ws.Range(excel_range)

        try:
            XL_SCREEN = constants.xlScreen
            XL_PICTURE = constants.xlPicture
        except Exception:
            XL_SCREEN = 1
            XL_PICTURE = -4147

        rng.CopyPicture(Appearance=XL_SCREEN, Format=XL_PICTURE)

        left = float(rng.Left)
        top = float(rng.Top)
        width = float(rng.Width)
        height = float(rng.Height)

        if width <= 0:
            width = 800
        if height <= 0:
            height = 600

        chart_obj = ws.ChartObjects().Add(left, top, width, height)
        chart = chart_obj.Chart

        try:
            chart.Paste()
        except Exception:
            chart_obj.Chart.Paste()

        time.sleep(wait_after_paste)

        tmp_export = str(out_path)  # export direct
        exported = chart.Export(tmp_export)
        if not exported:
            raise RuntimeError("Export Excel échoué")

    finally:
        try:
            chart_obj.Delete()
        except Exception:
            pass
        try:
            wb.Close(False)
        except Exception:
            pass
        try:
            excel.Quit()
        except Exception:
            pass

    # Upscale via Pillow (haute qualité)
    if upscale_factor != 1.0:
        img = Image.open(tmp_export)
        new_size = (int(img.width * upscale_factor), int(img.height * upscale_factor))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        img.save(out_path)

    return out_path


def draw_fig_keep_ratio(canvas, fig, x, y, width_pt):
    """
    Dessine une figure matplotlib sur un canvas ReportLab
    en conservant le ratio largeur/hauteur d'origine.

    Args:
        canvas: reportlab.pdfgen.canvas.Canvas
        fig: matplotlib.figure.Figure
        x, y: coordonnées bas-gauche en points
        width_pt: largeur souhaitée en points (1pt = 1/72 inch)
    """
    # Convertir la figure en buffer PNG
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    fig_width_px, fig_height_px = fig.get_size_inches() * fig.dpi
    buf.seek(0)

    # Calcul du ratio
    aspect = fig_height_px / fig_width_px
    height_pt = width_pt * aspect

    # ImageReader puis drawImage
    img = ImageReader(buf)
    canvas.drawImage(img, x, y, width=width_pt, height=height_pt)

def get_best_font_size(text, font_name, max_width, max_height, max_font=12, min_font=3):
    """Retourne la taille de police maximale qui rentre dans la cellule"""
    for size in range(max_font, min_font, -1):
        line_height = size * 1.2
        # Estimation pour textes multi-lignes, on vérifie la largeur d'une ligne
        if "<br/>" in text:
            lines = text.split("<br/>")
        else:
            lines = [text]

        too_wide = any(stringWidth(line, font_name, size) > (max_width - 4) for line in lines)
        too_high = (len(lines) * line_height) > (max_height - 4)

        if not too_wide and not too_high:
            return size
    return min_font