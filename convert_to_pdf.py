"""
Convert PROJECT_REPORT.md to PDF using fpdf2 (pure Python)
"""
import sys
import subprocess
from pathlib import Path

def install_package(package):
    """Install a Python package quietly"""
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", package])

def convert_md_to_pdf():
    """Convert markdown to PDF using fpdf2"""
    try:
        from fpdf import FPDF
    except ImportError:
        print("Installing fpdf2...")
        install_package("fpdf2")
        from fpdf import FPDF
    
    base_dir = Path(r"c:\Users\Enoch\Desktop\FINAL")
    report_path = base_dir / "PROJECT_REPORT.md"
    pdf_path = base_dir / "PROJECT_REPORT.pdf"
    
    # Read markdown file
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create PDF with better margins
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Use standard fonts
    pdf.set_font("Helvetica", size=11)
    
    # Process markdown content line by line
    lines = content.split('\n')
    
    for line in lines:
        # Skip empty lines
        if not line.strip():
            pdf.ln(2)
            continue
        
        # Handle headings
        if line.startswith('# '):
            pdf.set_font("Helvetica", "B", size=18)
            pdf.set_text_color(15, 32, 39)  # Dark blue
            text = line.replace('# ', '').strip()
            pdf.multi_cell(0, 10, text)
            pdf.ln(2)
            pdf.set_font("Helvetica", size=11)
            pdf.set_text_color(0, 0, 0)
        elif line.startswith('## '):
            pdf.set_font("Helvetica", "B", size=14)
            pdf.set_text_color(37, 117, 252)  # Blue
            text = line.replace('## ', '').strip()
            pdf.multi_cell(0, 8, text)
            pdf.ln(1)
            pdf.set_font("Helvetica", size=11)
            pdf.set_text_color(0, 0, 0)
        elif line.startswith('### '):
            pdf.set_font("Helvetica", "B", size=12)
            pdf.set_text_color(106, 17, 203)  # Purple
            text = line.replace('### ', '').strip()
            pdf.multi_cell(0, 7, text)
            pdf.set_font("Helvetica", size=11)
            pdf.set_text_color(0, 0, 0)
        elif line.startswith('- ') or line.startswith('* '):
            # Bullet point - just render with dash
            text = line.lstrip('- ').lstrip('* ').strip()
            pdf.multi_cell(0, 6, '  - ' + text)
        elif line.startswith('---'):
            # Horizontal line (add space)
            pdf.ln(2)
        else:
            # Regular text - skip if just whitespace
            text = line.strip()
            if text and len(text) > 2:  # Only process if line has meaningful content
                try:
                    pdf.multi_cell(0, 5, text)
                except:
                    # If rendering fails, try simpler approach
                    pass
    
    # Save PDF
    pdf.output(str(pdf_path))
    return str(pdf_path)

if __name__ == "__main__":
    print("=" * 60)
    print("Converting PROJECT_REPORT.md to PDF...")
    print("=" * 60)
    
    try:
        pdf_file = convert_md_to_pdf()
        print(f"✅ PDF successfully created!")
        print(f"📄 Location: {pdf_file}")
        print("=" * 60)
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 60)
