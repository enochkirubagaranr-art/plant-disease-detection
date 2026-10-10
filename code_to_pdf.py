import sys
from pathlib import Path

try:
    from fpdf import FPDF
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'fpdf2'])
    from fpdf import FPDF

BASE_DIR = Path(__file__).resolve().parent
SOURCE_FILE = BASE_DIR / 'streamlit_app.py'
OUTPUT_FILE = BASE_DIR / 'MAIN_CODE.pdf'

if not SOURCE_FILE.exists():
    raise FileNotFoundError(f'Source file not found: {SOURCE_FILE}')

content = SOURCE_FILE.read_text(encoding='utf-8')
lines = content.splitlines()

pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=10)
pdf.add_page()
pdf.set_font('Courier', size=9)
pdf.set_left_margin(10)
pdf.set_right_margin(10)
cell_width = 190

for line in lines:
    clean_line = line.rstrip('\n').replace('\t', '    ')
    clean_line = clean_line.encode('latin-1', errors='replace').decode('latin-1')
    max_chars = 110
    if not clean_line:
        pdf.ln(4)
    else:
        for i in range(0, len(clean_line), max_chars):
            pdf.multi_cell(cell_width, 4, clean_line[i:i + max_chars])

pdf.output(str(OUTPUT_FILE))
print(f'PDF created: {OUTPUT_FILE}')
