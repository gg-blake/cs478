from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import magic
from flask_swagger_ui import get_swaggerui_blueprint
from audit_parser import extract_as_txt_from_request_file, parse_audit_text

app = Flask(__name__)

# Configure Swagger UI
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Audit Report Parser API",
        'spec_version': '2.0'  # Changed to Swagger 2.0
    }
)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

@app.route('/parse_audit_report', methods=['POST'])
def parse_audit_report():
    """
    Parse an audit report PDF file
    ---
    tags:
      - Audit Report
    consumes:
      - multipart/form-data
    parameters:
      - in: formData
        name: file
        type: file
        required: true
        description: The PDF file to parse
    responses:
      200:
        description: PDF processed successfully
        schema:
          properties:
            message:
              type: string
              example: PDF processed successfully
            data:
              type: object
              description: Parsed audit report data
      400:
        description: Invalid input
        schema:
          properties:
            error:
              type: string
              example: No file provided
    """
    if not request.files:
        return jsonify({'error': 'No file provided'}), 400
        
    file = next(iter(request.files.values()))
    
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
