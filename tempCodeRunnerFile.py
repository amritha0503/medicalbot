# app.py
from flask import Flask, request, jsonify, render_template, url_for, send_file, session
from openai import OpenAI  # Updated import
import os
import cv2
import pytesseract
import numpy as np
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import json
from datetime import datetime
import io

# Set Tesseract PATH
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'  # Update this path if necessary

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'pdf', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16MB upload
app.secret_key = os.urandom(24)  # Required for session

# Create upload directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Preprocess the image to improve OCR accuracy"""
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not read the image file")

        # Resize if image is too large
        max_dimension = 2000
        height, width = image.shape[:2]
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            image = cv2.resize(image, None, fx=scale, fy=scale)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(binary)
        
        # Save preprocessed image
        preprocessed_path = os.path.join(
            app.config['UPLOAD_FOLDER'], 
            "preprocessed_" + os.path.basename(image_path)
        )
        cv2.imwrite(preprocessed_path, denoised)
        
        return preprocessed_path
    
    except Exception as e:
        print(f"Image preprocessing failed: {str(e)}")
        raise

def extract_text_from_image(image_path):
    """Extract text from image using OCR"""
    try:
        # Verify image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError("Image file not found")

        # Preprocess image
        preprocessed_path = preprocess_image(image_path)
        
        # Verify Tesseract installation
        if not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
            raise RuntimeError("Tesseract not found. Please verify installation.")
        
        # Configure Tesseract with better parameters
        custom_config = r'--oem 3 --psm 1 -l eng --dpi 300'
        
        # Extract text
        image = cv2.imread(preprocessed_path)
        if image is None:
            raise ValueError("Could not read preprocessed image")
        
        text = pytesseract.image_to_string(image, config=custom_config)
        
        # Validate extracted text
        if not text or len(text.strip()) < 10:
            raise ValueError("Insufficient text extracted from image")
        
        # Clean text
        cleaned_text = ' '.join(text.split())
        return cleaned_text

    except Exception as e:
        print(f"OCR Error: {str(e)}")
        raise

def analyze_lab_report(text):
    """Analyze lab report text using OpenAI API"""
    try:
        cleaned_text = text.strip().replace('\n', ' ').replace('\r', '')
        
        prompt = f"""
        Analyze this lab report text and provide results in valid JSON format:

        {cleaned_text}

        Format your complete response as a valid JSON object with this exact structure:
        {{
            "general_analysis": {{
                "summary": "string",
                "key_findings": ["string"],
                "overall_health_status": "string"
            }},
            "abnormal_findings": [
                {{
                    "test_name": "string",
                    "measured_value": "string",
                    "reference_range": "string",
                    "severity": "string",
                    "clinical_significance": "string",
                    "immediate_concerns": "string"
                }}
            ],
            "follow_up_testing": [
                {{
                    "recommended_test": "string",
                    "urgency": "string",
                    "reason": "string"
                }}
            ],
            "specialist_recommendations": [
                {{
                    "specialty": "string",
                    "reason": "string",
                    "urgency": "string"
                }}
            ],
            "lifestyle_recommendations": [
                {{
                    "category": "string",
                    "recommendation": "string",
                    "importance": "string"
                }}
            ],
            "monitoring_plan": {{
                "frequency": "string",
                "specific_tests": ["string"],
                "duration": "string"
            }}
        }}
        """

        response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical analyst that always responds in valid JSON format. Never include any text outside the JSON structure."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=2000,
            response_format={"type": "json_object"}  # Force JSON response
        )

        if not response.choices:
            raise ValueError("No response received from API")

        # Get the response content
        analysis = response.choices[0].message.content

        # Validate JSON
        try:
            parsed_json = json.loads(analysis)
            return json.dumps(parsed_json, indent=2)  # Return formatted JSON
        except json.JSONDecodeError as json_error:
            print(f"JSON parsing error: {str(json_error)}")
            print(f"Raw response: {analysis}")
            raise ValueError("Invalid JSON response from API")

    except Exception as e:
        error_msg = f"Analysis failed: {str(e)}"
        print(f"Error in analyze_lab_report: {error_msg}")
        return json.dumps({
            "error": error_msg,
            "status": "error",
            "raw_text": text[:100] + "..." if len(text) > 100 else text
        })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        conversation_history = data.get('history', [])

        # Prepare context-aware medical chat prompt
        prompt = f"""
        As a medical assistant, help with this query about lab results or medical topics.
        Previous conversation: {json.dumps(conversation_history)}
        Current question: {user_message}

        Provide a clear, professional response focusing on:
        1. Direct answers to medical questions
        2. Explanation of medical terms if present
        3. General health guidance when appropriate
        4. Clarification if more information is needed
        """
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical assistant helping with lab results and health queries. Provide clear, accurate information while maintaining appropriate medical disclaimers."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.5,
            max_tokens=2000
        )
        if response and response.choices:
            reply = response.choices[0].message.content
            return jsonify({
                'reply': reply,
            })
    except Exception as e:
        return jsonify({
            'reply': "I apologize, but I encountered an error. Please try rephrasing your question.",
            'status': 'error',
            'error': str(e)
        })

@app.route('/api/upload-report', methods=['POST'])
def upload_report():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Please select a lab report image'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Save uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Extract text from image
            try:
                extracted_text = extract_text_from_image(file_path)
                if not extracted_text:
                    return jsonify({'error': 'No text could be extracted from the image. Please ensure the image is clear and contains text.'}), 400
                
                # Analyze the extracted text
                analysis_result = analyze_lab_report(extracted_text)
                analysis_json = json.loads(analysis_result)
                
                if 'error' in analysis_json:
                    return jsonify({'error': analysis_json['error']}), 400
                
                # Store analysis in session
                session['last_analysis'] = analysis_result
                
                # Return results
                return jsonify({
                    'status': 'success',
                    'extracted_text': extracted_text,
                    'analysis': analysis_json,
                    'download_url': url_for('download_analysis', filename=filename)
                })
                
            except Exception as e:
                return jsonify({
                    'error': f'Analysis failed: {str(e)}. Please ensure the image is a clear lab report.'
                }), 500
            
        return jsonify({'error': 'Invalid file type. Please upload an image file (PNG, JPG, JPEG).'}), 400
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/download-analysis/<filename>', methods=['GET'])
def download_analysis(filename):
    try:
        analysis_result = session.get('last_analysis', None)
        if not analysis_result:
            return jsonify({'error': 'No analysis found'}), 404
        
        analysis = json.loads(analysis_result)
        report_content = []
        
        # Header
        report_content.append("MEDICAL LABORATORY REPORT ANALYSIS")
        report_content.append("=" * 50 + "\n")
        
        # General Analysis
        report_content.append("GENERAL ANALYSIS")
        report_content.append("-" * 20)
        general = analysis.get('general_analysis', {})
        report_content.append(f"Summary: {general.get('summary', '')}")
        report_content.append("\nKey Findings:")
        for finding in general.get('key_findings', []):
            report_content.append(f"• {finding}")
        report_content.append(f"\nOverall Health Status: {general.get('overall_health_status', '')}\n")
        
        # Abnormal Findings
        if analysis.get('abnormal_findings'):
            report_content.append("\nABNORMAL FINDINGS")
            report_content.append("-" * 20)
            for finding in analysis['abnormal_findings']:
                report_content.append(f"\nTest: {finding.get('test_name')}")
                report_content.append(f"Value: {finding.get('measured_value')}")
                report_content.append(f"Reference Range: {finding.get('reference_range')}")
                report_content.append(f"Severity: {finding.get('severity')}")
                report_content.append(f"Clinical Significance: {finding.get('clinical_significance')}")
                report_content.append(f"Immediate Concerns: {finding.get('immediate_concerns')}")
        
        # Follow-up Testing
        if analysis.get('follow_up_testing'):
            report_content.append("\nRECOMMENDED FOLLOW-UP TESTS")
            report_content.append("-" * 20)
            for test in analysis['follow_up_testing']:
                report_content.append(f"\nTest: {test.get('recommended_test')}")
                report_content.append(f"Urgency: {test.get('urgency')}")
                report_content.append(f"Reason: {test.get('reason')}")
        
        # Specialist Recommendations
        if analysis.get('specialist_recommendations'):
            report_content.append("\nSPECIALIST RECOMMENDATIONS")
            report_content.append("-" * 20)
            for spec in analysis['specialist_recommendations']:
                report_content.append(f"\nSpecialist: {spec.get('specialty')}")
                report_content.append(f"Reason: {spec.get('reason')}")
                report_content.append(f"Urgency: {spec.get('urgency')}")
        
        # Lifestyle Recommendations
        if analysis.get('lifestyle_recommendations'):
            report_content.append("\nLIFESTYLE RECOMMENDATIONS")
            report_content.append("-" * 20)
            for rec in analysis['lifestyle_recommendations']:
                report_content.append(f"\n{rec.get('category', '').title()}")
                report_content.append(f"Recommendation: {rec.get('recommendation')}")
                report_content.append(f"Importance: {rec.get('importance')}")
        
        # Monitoring Plan
        if analysis.get('monitoring_plan'):
            report_content.append("\nMONITORING PLAN")
            report_content.append("-" * 20)
            plan = analysis['monitoring_plan']
            report_content.append(f"Frequency: {plan.get('frequency')}")
            report_content.append("\nTests to Monitor:")
            for test in plan.get('specific_tests', []):
                report_content.append(f"• {test}")
            report_content.append(f"\nDuration: {plan.get('duration')}")
        
        # Create downloadable file
        buffer = io.StringIO()
        buffer.write('\n'.join(report_content))
        buffer.seek(0)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        download_filename = f"medical_analysis_{timestamp}.txt"
        
        return send_file(
            io.BytesIO(buffer.getvalue().encode()),
            mimetype='text/plain',
            as_attachment=True,
            download_name=download_filename
        )
        
    except Exception as e:
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)