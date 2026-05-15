import os
import tempfile
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Configure upload settings
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size
ALLOWED_EXTENSIONS = {'pdf'}

# Configure Gemini API
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/summarize', methods=['POST'])
def summarize_pdf():
    if not GEMINI_API_KEY:
        return jsonify({'error': 'GEMINI_API_KEY not set. Please add it to your .env file.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Only PDF files are allowed'}), 400

    # Get summary length preference
    summary_length = request.form.get('length', 'medium')
    length_instructions = {
        'short':  'Provide a concise summary in 3–5 bullet points.',
        'medium': 'Provide a well-structured summary with key points, main themes, and important details in about 150–250 words.',
        'long':   'Provide a comprehensive, detailed summary covering all major sections, arguments, data, and conclusions in 400–600 words.'
    }
    length_prompt = length_instructions.get(summary_length, length_instructions['medium'])

    try:
        # Save file to a temp location
        filename = secure_filename(file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        try:
            # Upload PDF to Gemini Files API
            uploaded_file = genai.upload_file(tmp_path, mime_type='application/pdf')

            # Generate summary using Gemini 2.5 Flash
            model = genai.GenerativeModel('gemini-2.5-flash')
            prompt = f"""You are an expert document analyst. Analyze the uploaded PDF and provide a clear, insightful summary.

{length_prompt}

Structure your response with:
- **Overview**: A one-sentence description of what this document is about
- **Key Points**: The most important information (use bullet points)
- **Main Themes**: Core topics or arguments covered
- **Conclusion**: What the document concludes or recommends (if applicable)

Be clear, accurate, and helpful. Use markdown formatting."""

            response = model.generate_content([uploaded_file, prompt])
            summary = response.text

            # Clean up uploaded file from Gemini
            genai.delete_file(uploaded_file.name)

        finally:
            # Always remove the local temp file
            os.unlink(tmp_path)

        return jsonify({
            'summary': summary,
            'filename': filename
        })

    except Exception as e:
        return jsonify({'error': f'Failed to process PDF: {str(e)}'}), 500


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'api_key_set': bool(GEMINI_API_KEY)})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
