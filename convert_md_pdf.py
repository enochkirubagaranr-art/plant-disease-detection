"""
Convert PROJECT_REPORT.md to PDF using ReportLab
"""
import sys
import subprocess
from pathlib import Path

def install_package(package):
    """Install a Python package quietly"""
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", package])

def convert_md_to_pdf():
    """Convert markdown to PDF using ReportLab"""
    import re
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
        from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
    except ImportError:
        print("Installing reportlab...")
        install_package("reportlab")
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
        from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
    
    base_dir = Path(r"c:\Users\Enoch\Desktop\FINAL")
    report_path = base_dir / "PROJECT_REPORT.md"
    pdf_path = base_dir / "PROJECT_REPORT.pdf"
    
    # Read markdown file
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create PDF
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4,
                          rightMargin=0.5*inch, leftMargin=0.5*inch,
                          topMargin=0.75*inch, bottomMargin=0.75*inch)
    
    # Create style sheet
    styles = getSampleStyleSheet()
    story = []
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor='#0f2027',
        spaceAfter=12,
        alignment=TA_CENTER
    )
    
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=13,
        textColor='#2575fc',
        spaceAfter=8,
        spaceBefore=8
    )
    
    heading3_style = ParagraphStyle(
        'CustomHeading3',
        parent=styles['Heading3'],
        fontSize=11,
        textColor='#6a11cb',
        spaceAfter=6,
        spaceBefore=6
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=6
    )
    
    bullet_style = ParagraphStyle(
        'BulletStyle',
        parent=styles['BodyText'],
        fontSize=10,
        leftIndent=20,
        spaceAfter=4
    )
    
    def escape_and_format(text):
        """Escape XML special characters and apply basic formatting"""
        # Escape special characters
        text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        # Convert **text** to <b>text</b> using regex
        text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
        # Convert *text* to <i>text</i> using regex
        text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
        return text
    
    # Process markdown
    lines = content.split('\n')
    
    for line in lines:
        # Skip empty lines
        if not line.strip():
            story.append(Spacer(1, 0.1*inch))
            continue
        
        # Handle headings
        if line.startswith('# '):
            text = line.replace('# ', '').strip()
            text = escape_and_format(text)
            story.append(Paragraph(text, title_style))
        elif line.startswith('## '):
            text = line.replace('## ', '').strip()
            text = escape_and_format(text)
            story.append(Paragraph(text, heading2_style))
        elif line.startswith('### '):
            text = line.replace('### ', '').strip()
            text = escape_and_format(text)
            story.append(Paragraph(text, heading3_style))
        elif line.startswith('- ') or line.startswith('* '):
            # Bullet point
            text = line.lstrip('- *').strip()
            text = escape_and_format(text)
            story.append(Paragraph('- ' + text, bullet_style))
        elif line.startswith('---'):
            # Horizontal line
            story.append(Spacer(1, 0.15*inch))
        else:
            # Regular text
            text = line.strip()
            if text and len(text) > 2:
                text = escape_and_format(text)
                story.append(Paragraph(text, body_style))
    
    # Build PDF
    doc.build(story)
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
