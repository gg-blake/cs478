import PyPDF2
import pdfplumber

def extract_urls_with_text(pdf_path):
    urls_and_text = []

    # Use PyPDF2 to get URLs
    with open(pdf_path, 'rb') as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(reader.pages):
                # Extract annotations (URLs) if they exist
                if "/Annots" in page:
                    annotations = page["/Annots"]
                    
                    for annotation in annotations:
                        annot_obj = annotation.get_object()

                        # Check if the annotation has a URI (URL)
                        if annot_obj.get("/Subtype") == "/Link" and "/A" in annot_obj:
                            uri = annot_obj["/A"].get("/URI")
                            # Attempt to get /Contents directly, or set to None if not available
                            text = annot_obj.get("/Contents")
                            
                            # If /Contents isn't available, fall back to extracting surrounding text
                            if not text:
                                # Get coordinates for the link annotation (if available)
                                rect = annot_obj.get("/Rect")
                                if rect:
                                    x0, y0, x1, y1 = rect  # Coordinates of the link area
                                    
                                    # Use pdfplumber to extract text in this area
                                    pdf_page = pdf.pages[page_num]
                                    text = pdf_page.within_bbox((float(x0), float(y0), float(x1), float(y1))).extract_text() or "No text available"
                            
                            if uri:
                                urls_and_text.append((text, uri))
    
    return urls_and_text

# Example usage
pdf_path = "pdf/ahnaf_audit.pdf"
extracted_data = extract_urls_with_text(pdf_path)

for text, url in extracted_data:
    print(f"Text: {text} -> URL: {url}")
