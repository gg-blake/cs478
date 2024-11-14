from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import magic
from audit_parser import extract_as_txt_from_request_file, parse_audit_text

app = Flask(__name__)

@app.route('/parse_audit_report', methods=['POST'])
def parse_audit_report():
    if 'audit_pdf' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
        
    file = request.files['audit_pdf']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Check if file is PDF using python-magic
    file_content = file.read()
    mime = magic.Magic(mime=True)
    file_type = mime.from_buffer(file_content)
    
    if file_type != 'application/pdf':
        return jsonify({'error': 'File must be a PDF'}), 400

    # Reset file pointer after reading
    file.seek(0)
    
    # Process the PDF file
    pdf_text = extract_as_txt_from_request_file(file)

    parse_result = parse_audit_text(pdf_text)

    return jsonify({'message': 'PDF processed successfully', 'data': parse_result}), 200

if __name__ == '__main__':
    app.run(debug=True)
