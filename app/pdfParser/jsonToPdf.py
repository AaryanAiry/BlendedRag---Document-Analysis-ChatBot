import json
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.lib.colors import Color

def reconstruct_pdf_from_json(json_path, output_pdf_path="recreated.pdf"):
    with open(json_path, "r", encoding="utf-8") as f:
        pdf_data = json.load(f)

    c = canvas.Canvas(output_pdf_path, pagesize=letter)
    page_width, page_height = letter

    allowed_fonts = ["Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Times-Roman", "Courier"]
    base_dir = os.path.dirname(json_path)

    for page in pdf_data.get("pages", []):
        page_w = page.get("width", page_width)
        page_h = page.get("height", page_height)
        c.setPageSize((page_w, page_h))

        for element in page.get("elements", []):
            etype = element.get("type")
            pos = element.get("position", {})

            if etype == "textbox":
                font_name = element.get("font", {}).get("name", "Helvetica")
                font_size = element.get("font", {}).get("size", 12) or 12
                bold = element.get("font", {}).get("bold", False)
                italic = element.get("font", {}).get("italic", False)
                color = element.get("font", {}).get("color", [0, 0, 0])

                # Normalize font selection
                if "Bold" in font_name or bold:
                    font_name = "Helvetica-Bold"
                elif "Oblique" in font_name or italic:
                    font_name = "Helvetica-Oblique"
                elif font_name not in allowed_fonts:
                    font_name = "Helvetica"

                try:
                    c.setFont(font_name, font_size)
                except Exception:
                    c.setFont("Helvetica", 12)

                # Normalize color input
                try:
                    r, g, b = color[:3]
                    c.setFillColor(Color(r, g, b))
                except Exception:
                    c.setFillColor(Color(0, 0, 0))

                x = pos.get("x", 0)
                y = page_h - pos.get("y", 0) - pos.get("height", 0)
                text = element.get("content", "")
                c.drawString(x, y, text)

            elif etype == "image":
                img_path = element.get("src")
                if img_path:
                    if not os.path.isabs(img_path):
                        img_path = os.path.join(base_dir, img_path)
                    if os.path.exists(img_path):
                        try:
                            img = ImageReader(img_path)
                            x = pos.get("x", 0)
                            y = page_h - pos.get("y", 0) - pos.get("height", 0)
                            width = pos.get("width", 100)
                            height = pos.get("height", 100)
                            c.drawImage(img, x, y, width=width, height=height)
                        except Exception as e:
                            print(f"⚠️ Failed to draw image {img_path}: {e}")
                    else:
                        print(f"⚠️ Image not found: {img_path}")

            elif etype == "table":
                x0 = pos.get("x", 0)
                y0 = page_h - pos.get("y", 0) - pos.get("height", 0)
                table_data = element.get("content", [])

                if table_data:
                    num_rows = len(table_data)
                    num_cols = len(table_data[0]) if num_rows > 0 else 0
                    if num_rows > 0 and num_cols > 0:
                        cell_w = pos.get("width", 100) / num_cols
                        cell_h = pos.get("height", 100) / num_rows

                        # Borders
                        for i in range(num_rows + 1):
                            c.line(x0, y0 + i * cell_h, x0 + pos.get("width", 100), y0 + i * cell_h)
                        for j in range(num_cols + 1):
                            c.line(x0 + j * cell_w, y0, x0 + j * cell_w, y0 + pos.get("height", 100))

                        # Text inside cells
                        for i, row in enumerate(table_data):
                            for j, cell_text in enumerate(row):
                                text_x = x0 + j * cell_w + 2
                                text_y = y0 + pos.get("height", 100) - (i + 1) * cell_h + 2
                                c.setFont("Helvetica", 10)
                                c.setFillColor(Color(0, 0, 0))
                                c.drawString(text_x, text_y, str(cell_text))

            elif etype == "shape" and element.get("shape") == "rect":
                color = element.get("color", [0, 0, 0])
                try:
                    r, g, b = color[:3]
                    c.setStrokeColor(Color(r, g, b))
                except Exception:
                    c.setStrokeColor(Color(0, 0, 0))

                c.rect(
                    pos.get("x", 0),
                    page_h - pos.get("y", 0) - pos.get("height", 0),
                    pos.get("width", 100),
                    pos.get("height", 100),
                    stroke=1,
                    fill=0
                )

        c.showPage()

    c.save()
    print(f"✅ Recreated PDF saved at: {output_pdf_path}")


if __name__ == "__main__":
    reconstruct_pdf_from_json("output_json/sample.json", "recreated_sample.pdf")
