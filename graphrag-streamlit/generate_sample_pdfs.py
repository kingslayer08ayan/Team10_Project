"""
Generates sample PDFs into database/ so the GraphRAG pipeline has something
to ingest out of the box. Replace these with your own PDFs any time --
the pipeline only assumes "some PDFs live in database/".
"""
import json
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

with open(".cache/papers.json") as f:
    papers = json.load(f)

styles = getSampleStyleSheet()
title_style = ParagraphStyle("TitleX", parent=styles["Title"], fontSize=15)
h2 = ParagraphStyle("H2", parent=styles["Heading2"], spaceBefore=10)
body = styles["BodyText"]

for p in papers:
    path = f"database/{p['id']}_{p['title'][:40].replace(' ', '_').replace('/', '-')}.pdf"
    doc = SimpleDocTemplate(path, pagesize=LETTER,
                            leftMargin=0.9 * inch, rightMargin=0.9 * inch,
                            topMargin=0.9 * inch, bottomMargin=0.9 * inch)
    story = [
        Paragraph(p["title"], title_style),
        Paragraph(f"({p['year']})", body),
        Spacer(1, 12),
        Paragraph("Problem", h2), Paragraph(p["problem"], body),
        Paragraph("Method", h2), Paragraph(p["method"], body),
        Paragraph("Dataset", h2), Paragraph(p["dataset"], body),
        Paragraph("Evaluation", h2), Paragraph(p["evaluation"], body),
        Paragraph("Results", h2), Paragraph(p["results"], body),
        Paragraph("Limitations", h2), Paragraph(p["limitation"], body),
    ]
    doc.build(story)
    print("wrote", path)

print(f"\n{len(papers)} sample PDFs written to database/")
