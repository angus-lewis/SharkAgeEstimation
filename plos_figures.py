"""Helpers for saving figures that meet PLOS ONE figure requirements.

PLOS ONE: max 7.5 in (19.05 cm) wide by 8.75 in (22.23 cm) tall, 300-600 dpi,
RGB TIFF with LZW compression, Arial/Times/Symbol fonts at 8-12 pt.
"""
from io import BytesIO

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image
from plotnine import element_text, theme

PLOS_MAX_WIDTH = 7.5
PLOS_MAX_HEIGHT = 8.75
PLOS_DPI = 300

plos_font = theme(text=element_text(family="Arial"))
matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType fonts in the PDF
matplotlib.rcParams["font.family"] = "Arial"

def save_plos(p, filename_stem):
    """Save a ggplot, plot composition or matplotlib Figure as {filename_stem}.pdf and {filename_stem}.tiff."""
    fig = p if isinstance(p, Figure) else p.draw()
    fig.savefig(f"{filename_stem}.pdf", facecolor="white")
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=PLOS_DPI, facecolor="white")
    if fig is not p:
        plt.close(fig)
    buf.seek(0)
    Image.open(buf).convert("RGB").save(
        f"{filename_stem}.tiff", compression="tiff_lzw", dpi=(PLOS_DPI, PLOS_DPI)
    )
