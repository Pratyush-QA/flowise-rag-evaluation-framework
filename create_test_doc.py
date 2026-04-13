"""
Creates a PDF knowledge base document for Flowise RAG pipeline testing.
Content is aligned with the test queries in the RAGAS evaluation framework.
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib import colors
import os

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "Sample_Docs", "scientists_knowledge_base.pdf")

def create_pdf():
    doc = SimpleDocTemplate(
        OUTPUT_PATH,
        pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=20, spaceAfter=20, alignment=TA_CENTER)
    h1_style = ParagraphStyle('H1', parent=styles['Heading1'], fontSize=14, spaceBefore=16, spaceAfter=8, textColor=colors.HexColor('#1a1a2e'))
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=11, spaceAfter=8, leading=16)

    content = []

    content.append(Paragraph("Famous Scientists — Knowledge Base", title_style))
    content.append(Paragraph("A reference document for RAG pipeline evaluation", styles['Normal']))
    content.append(Spacer(1, 0.5*cm))

    # ---- Albert Einstein ----
    content.append(Paragraph("Albert Einstein and the Theory of Relativity", h1_style))
    content.append(Paragraph(
        "Albert Einstein was a German-born theoretical physicist who is widely regarded as one of the greatest "
        "scientists of all time. Born on March 14, 1879, in Ulm, Germany, Einstein proposed the theory of "
        "relativity, which fundamentally transformed our understanding of time, space, and gravity.",
        body_style))
    content.append(Paragraph(
        "Einstein's special theory of relativity, published in 1905, introduced the famous equation E=mc², "
        "establishing that energy and mass are interchangeable. His general theory of relativity, published in "
        "1915, described gravity not as a force but as a curvature of spacetime caused by mass and energy.",
        body_style))
    content.append(Paragraph(
        "In 1921, Albert Einstein was awarded the Nobel Prize in Physics for his discovery of the law of the "
        "photoelectric effect, which was pivotal in establishing quantum theory. Einstein's contributions to "
        "physics revolutionized modern science and laid the foundation for technologies such as GPS systems, "
        "nuclear energy, and lasers.",
        body_style))
    content.append(Spacer(1, 0.3*cm))

    # ---- Marie Curie ----
    content.append(Paragraph("Marie Curie and Radioactivity Research", h1_style))
    content.append(Paragraph(
        "Marie Curie, born Maria Sklodowska on November 7, 1867, in Warsaw, Poland, was a pioneering physicist "
        "and chemist who conducted groundbreaking research on radioactivity. She was the first woman to win a "
        "Nobel Prize, and she remains the only person in history to have won Nobel Prizes in two different "
        "scientific disciplines.",
        body_style))
    content.append(Paragraph(
        "Marie Curie won the Nobel Prize in Physics in 1903, which she shared with her husband Pierre Curie and "
        "Henri Becquerel, for their research on spontaneous radiation. She later won the Nobel Prize in Chemistry "
        "in 1911 for her discovery of the elements polonium and radium.",
        body_style))
    content.append(Paragraph(
        "Curie developed techniques for isolating radioactive isotopes and was the first female professor at "
        "the University of Paris. Her pioneering research laid the groundwork for modern nuclear physics and "
        "cancer treatment using radiation therapy. She died on July 4, 1934, due to aplastic anemia caused by "
        "prolonged exposure to radiation.",
        body_style))
    content.append(Spacer(1, 0.3*cm))

    # ---- Isaac Newton ----
    content.append(Paragraph("Isaac Newton and Classical Mechanics", h1_style))
    content.append(Paragraph(
        "Sir Isaac Newton was an English mathematician, physicist, and astronomer born on January 4, 1643, in "
        "Woolsthorpe, England. Newton formulated the laws of motion and universal gravitation, which laid the "
        "foundation for classical mechanics and dominated the scientific view of the physical universe for over "
        "three centuries.",
        body_style))
    content.append(Paragraph(
        "Newton's three laws of motion describe the relationship between a body and the forces acting upon it, "
        "and how motion changes in response to those forces. His law of universal gravitation states that every "
        "particle of matter in the universe attracts every other particle with a force proportional to the "
        "product of their masses and inversely proportional to the square of the distance between them.",
        body_style))
    content.append(Paragraph(
        "In 1687, Newton published his landmark work Philosophiae Naturalis Principia Mathematica (Mathematical "
        "Principles of Natural Philosophy), commonly known as the Principia. This work is considered one of the "
        "most important books in the history of science. Newton also made major contributions to optics and "
        "shares credit with Gottfried Wilhelm Leibniz for developing calculus.",
        body_style))
    content.append(Spacer(1, 0.3*cm))

    # ---- Charles Darwin ----
    content.append(Paragraph("Charles Darwin and the Theory of Evolution", h1_style))
    content.append(Paragraph(
        "Charles Robert Darwin was an English naturalist and biologist born on February 12, 1809, in Shrewsbury, "
        "England. Darwin introduced the theory of evolution by natural selection, which is the unifying theory "
        "of the life sciences and explains the diversity of living organisms on Earth.",
        body_style))
    content.append(Paragraph(
        "Darwin's theory of evolution by natural selection, published in his landmark 1859 book On the Origin "
        "of Species, proposed that all species of life have descended from common ancestors. Natural selection "
        "is the process by which organisms with favorable traits are more likely to survive and reproduce, "
        "passing those traits on to the next generation.",
        body_style))
    content.append(Paragraph(
        "Darwin's voyage on HMS Beagle from 1831 to 1836 was transformative for his scientific development. "
        "During the voyage, he observed the unique wildlife on the Galapagos Islands, particularly the variation "
        "among finches, which contributed to his development of evolutionary theory. His work revolutionized "
        "our understanding of biology and the relationship between all living beings.",
        body_style))
    content.append(Spacer(1, 0.3*cm))

    # ---- Ada Lovelace ----
    content.append(Paragraph("Ada Lovelace — The First Computer Programmer", h1_style))
    content.append(Paragraph(
        "Augusta Ada King, Countess of Lovelace, commonly known as Ada Lovelace, was an English mathematician "
        "born on December 10, 1815, in London. She is widely regarded as the first computer programmer for her "
        "work on Charles Babbage's early mechanical general-purpose computer, the Analytical Engine.",
        body_style))
    content.append(Paragraph(
        "In 1843, Ada Lovelace translated an article about the Analytical Engine written by Italian mathematician "
        "Luigi Menabrea, and supplemented it with her own extensive notes. These notes, which were three times "
        "longer than the original article, contained what is recognized as the first algorithm intended to be "
        "processed by a machine, making her the world's first computer programmer.",
        body_style))
    content.append(Paragraph(
        "Lovelace was the daughter of the poet Lord Byron and mathematician Anne Isabella Milbanke. She had a "
        "remarkable vision for computing machines, suggesting they could be used for more than just pure "
        "calculation, including composing music. The programming language Ada, developed by the U.S. Department "
        "of Defense in the 1980s, was named in her honor.",
        body_style))
    content.append(Spacer(1, 0.3*cm))

    # ---- Eiffel Tower (for RubricScore test) ----
    content.append(Paragraph("The Eiffel Tower — Paris, France", h1_style))
    content.append(Paragraph(
        "The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars in Paris, France. "
        "It was designed and built by Gustave Eiffel's engineering company between 1887 and 1889 as the "
        "entrance arch for the 1889 World's Fair. The tower stands 330 metres tall and was the tallest "
        "man-made structure in the world for 41 years.",
        body_style))
    content.append(Paragraph(
        "The Eiffel Tower is one of the most recognizable structures in the world and France's most-visited "
        "monument, attracting nearly 7 million visitors annually. It is located in the 7th arrondissement of "
        "Paris, near the Seine River, and is visible from many parts of the city.",
        body_style))

    doc.build(content)
    print(f"PDF created successfully: {OUTPUT_PATH}")
    return OUTPUT_PATH

if __name__ == "__main__":
    create_pdf()
