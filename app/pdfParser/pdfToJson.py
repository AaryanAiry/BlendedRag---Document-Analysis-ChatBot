import fitz  # PyMuPDF
import pdfplumber
import json
import os

def extract_pdf_layout(pdf_path, output_dir="output_json"):
    # Create output folder for images
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Open with PyMuPDF for text/images
    doc = fitz.open(pdf_path)

    # Also open with pdfplumber for table detection
    plumber_doc = pdfplumber.open(pdf_path)

    pdf_data = {"document": os.path.basename(pdf_path), "pages": []}

    for page_num in range(len(doc)):
        page = doc[page_num]
        plumber_page = plumber_doc.pages[page_num]

        width, height = page.rect.width, page.rect.height
        page_dict = {
            "page_number": page_num + 1,
            "width": width,
            "height": height,
            "elements": []
        }

        # === Extract text with font & positions ===
        text_dict = page.get_text("dict")
        for block in text_dict["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    element = {
                        "type": "textbox",
                        "position": {
                            "x": span["bbox"][0],
                            "y": span["bbox"][1],
                            "width": span["bbox"][2] - span["bbox"][0],
                            "height": span["bbox"][3] - span["bbox"][1],
                        },
                        "font": {
                            "name": span.get("font", "Unknown"),
                            "size": span.get("size", 0),
                            "bold": "Bold" in span.get("font", ""),
                            "italic": "Italic" in span.get("font", ""),
                        },
                        "content": span["text"]
                    }
                    page_dict["elements"].append(element)

        # === Extract images ===
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list, start=1):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)

            image_filename = f"page{page_num+1}_img{img_index}.png"
            image_path = os.path.join(images_dir, image_filename)

            if pix.n < 5:  # RGB or grayscale
                pix.save(image_path)
            else:  # CMYK
                pix = fitz.Pixmap(fitz.csRGB, pix)
                pix.save(image_path)

            # Get position(s) where image is drawn
            rects = page.get_image_rects(xref)
            for rect in rects:
                element = {
                    "type": "image",
                    "position": {
                        "x": rect.x0,
                        "y": rect.y0,
                        "width": rect.width,
                        "height": rect.height,
                    },
                    "src": os.path.join("images", image_filename)
                }
                page_dict["elements"].append(element)

        # === Extract tables (basic with pdfplumber) ===
        try:
            tables = plumber_page.find_tables()
            for t_index, table in enumerate(tables, start=1):
                element = {
                    "type": "table",
                    "position": {
                        "x": table.bbox[0],
                        "y": table.bbox[1],
                        "width": table.bbox[2] - table.bbox[0],
                        "height": table.bbox[3] - table.bbox[1],
                    },
                    "content": table.extract(),  # raw 2D array of cell texts
                }
                page_dict["elements"].append(element)
        except Exception as e:
            print(f"No tables found on page {page_num+1}: {e}")

        # Add page dict
        pdf_data["pages"].append(page_dict)

    plumber_doc.close()

    # Save JSON
    json_path = os.path.join(output_dir, os.path.splitext(os.path.basename(pdf_path))[0] + ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(pdf_data, f, indent=4, ensure_ascii=False)

    print(f"✅ Layout JSON saved at: {json_path}")
    print(f"✅ Extracted images saved at: {images_dir}")


if __name__ == "__main__":
    # Example usage
    extract_pdf_layout("sample.pdf")
