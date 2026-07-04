import os
import json
import uuid
import base64
import tempfile
import logging
import typing
import threading
import time
from io import BytesIO
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
# from threading import Lock, Thread          # only needed by the 360 spin section (commented out below)
# from datetime import datetime                # only needed by the 360 spin section (commented out below)
# import concurrent.futures                    # only needed by the 360 spin section (commented out below)

from flask import Flask, request, jsonify, Blueprint
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from PIL import Image as PIL_Image
from dotenv import load_dotenv

# Vertex AI (virtual try-on) SDK surface
from google import genai as vertex_genai
from google.genai.types import (
    Image,
    ProductImage,
    RecontextImageConfig,
    RecontextImageSource,
)
from google.oauth2 import service_account

# Gemini API (360 spin) SDK surface — only used by the commented-out section below
# from google import genai as gemini_genai
# from google.genai import types as gemini_types

# Load environment variables
load_dotenv()

# ─────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# App + global config
# ─────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

CORS(app,
     origins=['*'],
     methods=['GET', 'POST', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization'])

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
VALID_CATEGORIES = ['upper_body', 'lower_body', 'dresses']

# ─────────────────────────────────────────────────────────
# Shared-secret API key auth — protects /try-on* only.
# No login system exists, so this isn't per-user auth — it just proves
# the caller is our app (or holds our key), not a random script.
# ─────────────────────────────────────────────────────────
APP_API_KEY = os.getenv("APP_API_KEY", "")

if not APP_API_KEY:
    logger.error("APP_API_KEY not set — /try-on endpoints will reject all requests")


def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not APP_API_KEY:
            return jsonify({
                'success': False,
                'error': 'Server misconfigured',
                'message': 'Authentication is not configured'
            }), 503

        auth_header = request.headers.get('Authorization', '')
        expected = f'Bearer {APP_API_KEY}'

        if auth_header != expected:
            return jsonify({
                'success': False,
                'error': 'Unauthorized',
                'message': 'Missing or invalid Authorization header'
            }), 401

        return f(*args, **kwargs)
    return decorated_function


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — Virtual Try-On service (Vertex AI)
# ═══════════════════════════════════════════════════════════════

PROJECT_ID = os.getenv("PROJECT_ID", "poetic-chariot-471517-p8")
LOCATION = os.getenv("LOCATION", "us-central1")

executor = ThreadPoolExecutor(max_workers=3)
processing_results = {}
processing_lock = threading.Lock()

tryon_client = None
try:
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON"):
        service_account_info = json.loads(os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON"))
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        tryon_client = vertex_genai.Client(
            vertexai=True,
            project=PROJECT_ID,
            location=LOCATION,
            credentials=credentials
        )
        logger.info("✅ Virtual Try-On (Vertex AI) client initialized successfully")
    else:
        logger.error("❌ GOOGLE_APPLICATION_CREDENTIALS_JSON not found in environment")
except Exception as e:
    logger.error(f"❌ Failed to initialize Virtual Try-On client: {e}")
    tryon_client = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_image_file(file):
    if not file or not file.filename:
        return False, "No file provided"

    if not allowed_file(file.filename):
        return False, f"Invalid file format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"

    file.seek(0, 2)
    file_size = file.tell()
    file.seek(0)

    if file_size > app.config['MAX_CONTENT_LENGTH']:
        return False, "File too large. Maximum size is 16MB"

    if file_size == 0:
        return False, "Empty file provided"

    return True, "Valid file"


def save_uploaded_file(file):
    try:
        filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        logger.info(f"File saved to: {temp_path}")
        return temp_path
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        return None


def pil_image_to_base64(pil_image):
    try:
        buffer = BytesIO()
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        pil_image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error converting image to base64: {e}")
        raise


def cleanup_files(file_paths):
    for file_path in file_paths:
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up file: {file_path}")
        except Exception as e:
            logger.warning(f"Could not delete temp file {file_path}: {e}")


def require_tryon_client(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not tryon_client:
            return jsonify({
                'success': False,
                'error': 'Service unavailable - Virtual Try-On client not initialized',
                'message': 'The AI service is currently unavailable. Please try again later.'
            }), 503
        return f(*args, **kwargs)
    return decorated_function


def process_try_on_background(request_id, person_path, clothing_path, garment_description, category):
    try:
        logger.info(f"[{request_id}] Starting background try-on processing...")

        response = tryon_client.models.recontext_image(
            model="virtual-try-on-001",
            source=RecontextImageSource(
                person_image=Image.from_file(location=person_path),
                product_images=[ProductImage(product_image=Image.from_file(location=clothing_path))],
            ),
            config=RecontextImageConfig(
                output_mime_type="image/png",
                number_of_images=1,
                safety_filter_level="BLOCK_LOW_AND_ABOVE",
            ),
        )

        logger.info(f"[{request_id}] Virtual Try-On API call successful!")

        if not response.generated_images:
            with processing_lock:
                processing_results[request_id] = {
                    'status': 'failed',
                    'success': False,
                    'error': 'No image generated',
                    'message': 'The AI model did not generate any images. Please try again.'
                }
            logger.error(f"[{request_id}] No images generated by API")
            return

        result_image = typing.cast(PIL_Image.Image, response.generated_images[0].image._pil_image)
        output_base64 = pil_image_to_base64(result_image)

        with processing_lock:
            processing_results[request_id] = {
                'status': 'completed',
                'success': True,
                'message': 'Virtual try-on completed successfully',
                'results': {
                    'try_on_image': f"data:image/png;base64,{output_base64}",
                    'masked_image': None
                },
                'parameters': {
                    'garment_description': garment_description,
                    'category': category,
                    'model': 'virtual-try-on-001'
                },
                'completed_at': time.time()
            }

        logger.info(f"[{request_id}] Try-on processing completed successfully")

    except Exception as e:
        logger.error(f"[{request_id}] Error during try-on background processing: {str(e)}")
        with processing_lock:
            processing_results[request_id] = {
                'status': 'failed',
                'success': False,
                'error': 'Virtual try-on failed',
                'message': f'An error occurred during processing: {str(e)}'
            }
    finally:
        cleanup_files([person_path, clothing_path])


tryon_bp = Blueprint('tryon', __name__)


@tryon_bp.route('/try-on', methods=['POST'])
@require_tryon_client
@require_api_key
def virtual_try_on():
    person_path = None
    clothing_path = None
    request_id = str(uuid.uuid4())

    try:
        if 'person_image' not in request.files or 'clothing_image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'Missing required files',
                'message': 'Both person_image and clothing_image are required'
            }), 400

        person_image = request.files['person_image']
        clothing_image = request.files['clothing_image']
        garment_description = request.form.get('garment_description', 'stylish clothing')
        category = request.form.get('category', 'upper_body')

        if category not in VALID_CATEGORIES:
            return jsonify({
                'success': False,
                'error': 'Invalid category',
                'message': f'Category must be one of: {", ".join(VALID_CATEGORIES)}'
            }), 400

        is_valid, message = validate_image_file(person_image)
        if not is_valid:
            return jsonify({'success': False, 'error': 'Invalid person image', 'message': message}), 400

        is_valid, message = validate_image_file(clothing_image)
        if not is_valid:
            return jsonify({'success': False, 'error': 'Invalid clothing image', 'message': message}), 400

        person_path = save_uploaded_file(person_image)
        clothing_path = save_uploaded_file(clothing_image)

        if not person_path or not clothing_path:
            return jsonify({
                'success': False,
                'error': 'File processing error',
                'message': 'Failed to process uploaded files'
            }), 500

        logger.info(f"[{request_id}] Processing try-on with category: {category}, description: {garment_description}")

        with processing_lock:
            processing_results[request_id] = {
                'status': 'processing',
                'started_at': time.time()
            }

        executor.submit(
            process_try_on_background,
            request_id,
            person_path,
            clothing_path,
            garment_description,
            category
        )

        return jsonify({
            'success': True,
            'message': 'Processing started',
            'request_id': request_id,
            'status_url': f'/try-on/status/{request_id}',
            'note': 'Poll the status_url endpoint to get the result'
        }), 202

    except Exception as e:
        cleanup_files([person_path, clothing_path])
        logger.error(f"[{request_id}] Error during try-on request: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Request processing failed',
            'message': f'An error occurred: {str(e)}'
        }), 500


@tryon_bp.route('/try-on/status/<request_id>', methods=['GET'])
@require_api_key
def try_on_status(request_id):
    with processing_lock:
        if request_id not in processing_results:
            return jsonify({
                'success': False,
                'error': 'Request not found',
                'message': 'The specified request ID was not found or has expired'
            }), 404

        result = processing_results[request_id].copy()

    if result.get('status') == 'processing':
        elapsed = time.time() - result.get('started_at', time.time())
        return jsonify({
            'success': True,
            'status': 'processing',
            'message': f'Still processing ({elapsed:.1f}s elapsed)',
            'request_id': request_id
        }), 202

    if result.get('status') == 'completed':
    
        def delayed_cleanup():
            time.sleep(300)
            with processing_lock:
                processing_results.pop(request_id, None)
                logger.info(f"Cleaned up completed request: {request_id}")
        
        cleanup_thread = threading.Thread(target=delayed_cleanup, daemon=True)
        cleanup_thread.start()
        
        return jsonify(result), 200  

    if result.get('status') == 'failed':
        return jsonify(result), 500   

    return jsonify({'success': False, 'error': 'Unknown status', 'message': 'The request status is unknown'}), 500


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — 360° Spin service (Gemini API) — COMMENTED OUT
#
# Disabled per request. Nothing deleted — uncomment this whole block
# (and its matching imports near the top of the file, and the
# `app.register_blueprint(spin_bp)` line further down) to restore it.
# ═══════════════════════════════════════════════════════════════

# spin_job_store = {}
# spin_job_store_lock = Lock()
#
# _gemini_client = None
#
#
# def get_gemini_client():
#     global _gemini_client
#     if _gemini_client is None:
#         api_key = os.getenv("GEMINI_API_KEY")
#         if not api_key:
#             raise ValueError("GEMINI_API_KEY not found in environment variables")
#         _gemini_client = gemini_genai.Client(api_key=api_key)
#         logger.info("✅ 360° Spin (Gemini) client initialized successfully")
#     return _gemini_client
#
#
# def encode_image_to_base64(image_data):
#     if isinstance(image_data, bytes):
#         return base64.b64encode(image_data).decode()
#     buffer = BytesIO()
#     image_data.save(buffer, format='JPEG')
#     return base64.b64encode(buffer.getvalue()).decode()
#
#
# def generate_angle_image(client, image_data, angle_prompt, suffix, back_garment_image=None):
#     model = "gemini-3.1-flash-image-preview"
#
#     base64_image = encode_image_to_base64(image_data)
#
#     content_parts = [
#         gemini_types.Part.from_bytes(
#             mime_type="image/jpeg",
#             data=base64.b64decode(base64_image)
#         )
#     ]
#
#     if back_garment_image is not None and "back view" in angle_prompt.lower():
#         back_garment_base64 = encode_image_to_base64(back_garment_image)
#         content_parts.append(
#             gemini_types.Part.from_bytes(
#                 mime_type="image/jpeg",
#                 data=base64.b64decode(back_garment_base64)
#             )
#         )
#
#     content_parts.append(gemini_types.Part.from_text(text=angle_prompt))
#
#     contents = [gemini_types.Content(role="user", parts=content_parts)]
#
#     config = gemini_types.GenerateContentConfig(
#         response_modalities=["image", "text"],
#         safety_settings=[
#             gemini_types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_LOW_AND_ABOVE"),
#             gemini_types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_LOW_AND_ABOVE"),
#             gemini_types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_LOW_AND_ABOVE"),
#             gemini_types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_LOW_AND_ABOVE")
#         ],
#         response_mime_type="text/plain",
#     )
#
#     try:
#         for chunk in client.models.generate_content_stream(model=model, contents=contents, config=config):
#             if chunk.candidates and chunk.candidates[0].content.parts:
#                 for part in chunk.candidates[0].content.parts:
#                     if part.inline_data:
#                         return part.inline_data.data
#         return None
#     except Exception as e:
#         logger.error(f"Error generating angle image: {e}")
#         return None
#
#
# def generate_single_angle(args):
#     client, image_bytes, angle, description, index, back_garment_bytes = args
#
#     angle_prompt = f"""Generate {description} of the same person wearing identical clothing.
#
# CRITICAL REQUIREMENTS:
# 1. BACKGROUND: Maintain the EXACT SAME background from the original image. The background color, lighting, shadows, and environment MUST remain completely identical across all angles. This is STRICT and NON-NEGOTIABLE.
# 2. PERSON: Keep the exact same person appearance, face, body proportions, height, and features.
# 3. CLOTHING: Maintain identical clothing fit, colors, patterns, fabric texture, and draping style.
# 4. LIGHTING: Keep the same lighting direction, intensity, and shadows as the original image.
# 5. PERSPECTIVE: Show {angle}° rotation perspective while keeping everything else constant.
#
# The only change should be the rotation angle. Everything else including background MUST stay identical to the source image."""
#
#     garment_ref = back_garment_bytes if "back view" in description else None
#
#     image_data = generate_angle_image(
#         client,
#         image_bytes,
#         angle_prompt,
#         f"{index:02d}_{angle:03.0f}deg",
#         back_garment_image=garment_ref
#     )
#
#     if image_data:
#         return {
#             'angle': angle,
#             'description': description,
#             'image_data': base64.b64encode(image_data).decode(),
#             'index': index
#         }
#     return None
#
#
# def update_spin_job_status(request_id, status, message=None, progress=None, result=None, error=None):
#     with spin_job_store_lock:
#         if request_id in spin_job_store:
#             spin_job_store[request_id]['status'] = status
#             spin_job_store[request_id]['updated_at'] = datetime.now()
#
#             if message:
#                 spin_job_store[request_id]['message'] = message
#             if progress is not None:
#                 spin_job_store[request_id]['progress'] = progress
#             if result is not None:
#                 spin_job_store[request_id]['result'] = result
#             if error is not None:
#                 spin_job_store[request_id]['error'] = error
#
#
# def process_360_generation_async(request_id, image_bytes, back_garment_bytes, num_angles):
#     try:
#         logger.info(f"🔄 Starting async 360° generation for request: {request_id}")
#         update_spin_job_status(request_id, 'processing', 'Preparing images...', 0)
#
#         angles = [i * (360 / num_angles) for i in range(num_angles)]
#         angle_descriptions = [
#             "front view",
#             "front-right diagonal view",
#             "right side view",
#             "back-right diagonal view",
#             "back view",
#             "back-left diagonal view",
#             "left side view",
#             "front-left diagonal view"
#         ]
#
#         client = get_gemini_client()
#
#         tasks = [
#             (client, image_bytes, angle, desc, i, back_garment_bytes)
#             for i, (angle, desc) in enumerate(zip(angles, angle_descriptions))
#         ]
#
#         generated_images = []
#         completed = 0
#
#         update_spin_job_status(request_id, 'processing', 'Generating angles...', 10)
#
#         with concurrent.futures.ThreadPoolExecutor(max_workers=4) as spin_executor:
#             futures = {spin_executor.submit(generate_single_angle, task): i for i, task in enumerate(tasks)}
#
#             for future in concurrent.futures.as_completed(futures):
#                 result = future.result()
#                 if result:
#                     generated_images.append(result)
#                     completed += 1
#
#                     progress = 10 + int((completed / len(tasks)) * 85)
#                     update_spin_job_status(
#                         request_id,
#                         'processing',
#                         f'Generated {completed}/{len(tasks)} angles...',
#                         progress
#                     )
#                     logger.info(f"✅ Generated angle {completed}/{len(tasks)} for request {request_id}")
#
#         generated_images.sort(key=lambda x: x['index'])
#
#         if not generated_images:
#             update_spin_job_status(request_id, 'failed', error='Failed to generate any images')
#             logger.error(f"❌ No images generated for request {request_id}")
#             return
#
#         result = {
#             'success': True,
#             'images': generated_images,
#             'total_angles': len(generated_images)
#         }
#
#         update_spin_job_status(request_id, 'completed', 'Generation complete!', 100, result=result)
#         logger.info(f"✅ 360° generation completed successfully for request {request_id}")
#
#     except Exception as e:
#         logger.error(f"❌ Error in async 360° generation for request {request_id}: {e}")
#         update_spin_job_status(request_id, 'failed', error=str(e))
#
#
# def cleanup_old_spin_jobs():
#     """Remove 360° jobs older than 1 hour"""
#     with spin_job_store_lock:
#         now = datetime.now()
#         to_remove = []
#         for req_id, job in spin_job_store.items():
#             age = (now - job['updated_at']).total_seconds()
#             if age > 3600:
#                 to_remove.append(req_id)
#
#         for req_id in to_remove:
#             del spin_job_store[req_id]
#             logger.info(f"🧹 Cleaned up old 360° job: {req_id}")
#
#
# spin_bp = Blueprint('spin360', __name__)
#
#
# @spin_bp.route('/generate-360', methods=['POST'])
# def generate_360():
#     """
#     Generate 360° spin images (async with polling)
#     Expected form data:
#     - person_image: The try-on result image (required)
#     - back_garment_image: Back view reference (optional)
#     - num_angles: Number of angles to generate (default: 8)
#
#     Returns: 202 with request_id for polling
#     """
#     try:
#         if 'person_image' not in request.files:
#             return jsonify({'error': 'person_image is required'}), 400
#
#         person_file = request.files['person_image']
#         back_garment_file = request.files.get('back_garment_image')
#         num_angles = int(request.form.get('num_angles', 8))
#
#         request_id = str(uuid.uuid4())
#
#         person_image = PIL_Image.open(person_file)
#         img_buffer = BytesIO()
#         person_image.save(img_buffer, format='JPEG')
#         image_bytes = img_buffer.getvalue()
#
#         back_garment_bytes = None
#         if back_garment_file:
#             back_garment_pil = PIL_Image.open(back_garment_file)
#             back_garment_buffer = BytesIO()
#             back_garment_pil.save(back_garment_buffer, format='JPEG')
#             back_garment_bytes = back_garment_buffer.getvalue()
#
#         with spin_job_store_lock:
#             spin_job_store[request_id] = {
#                 'status': 'queued',
#                 'message': 'Request queued for processing',
#                 'progress': 0,
#                 'created_at': datetime.now(),
#                 'updated_at': datetime.now(),
#                 'result': None,
#                 'error': None
#             }
#
#         thread = Thread(
#             target=process_360_generation_async,
#             args=(request_id, image_bytes, back_garment_bytes, num_angles)
#         )
#         thread.daemon = True
#         thread.start()
#
#         logger.info(f"✅ 360° generation request accepted: {request_id}")
#
#         return jsonify({
#             'request_id': request_id,
#             'message': '360° generation started. Use /status/{request_id} to check progress.'
#         }), 202
#
#     except Exception as e:
#         logger.error(f"Error in generate_360: {e}")
#         return jsonify({'error': str(e)}), 500
#
#
# @spin_bp.route('/status/<request_id>', methods=['GET'])
# def check_spin_status(request_id):
#     """
#     Check the status of a 360° generation request
#     Returns:
#     - 202: Still processing
#     - 200: Completed successfully
#     - 404: Request not found
#     - 500: Processing failed
#     """
#     with spin_job_store_lock:
#         if request_id not in spin_job_store:
#             return jsonify({'error': 'Request not found'}), 404
#
#         job = spin_job_store[request_id]
#
#         if job['status'] in ['queued', 'processing']:
#             return jsonify({
#                 'status': job['status'],
#                 'message': job['message'],
#                 'progress': job['progress'],
#                 'request_id': request_id
#             }), 202
#
#         if job['status'] == 'completed':
#             result = job['result']
#             del spin_job_store[request_id]
#             return jsonify(result), 200
#
#         if job['status'] == 'failed':
#             error_msg = job.get('error', 'Unknown error occurred')
#             del spin_job_store[request_id]
#             return jsonify({'error': error_msg}), 500
#
#         return jsonify({'error': 'Unknown job status'}), 500


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — Shared routes, error handlers, startup
# ═══════════════════════════════════════════════════════════════

app.register_blueprint(tryon_bp)
# app.register_blueprint(spin_bp)   # disabled — 360 spin commented out


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    logger.warning("File upload too large")
    return jsonify({'success': False, 'error': 'File too large', 'message': 'Maximum file size is 16MB'}), 413


@app.errorhandler(404)
def handle_not_found(e):
    return jsonify({'success': False, 'error': 'Endpoint not found', 'message': 'The requested endpoint does not exist'}), 404


@app.errorhandler(500)
def handle_internal_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({'success': False, 'error': 'Internal server error', 'message': 'An unexpected error occurred. Please try again later.'}), 500


@app.errorhandler(405)
def handle_method_not_allowed(e):
    return jsonify({'success': False, 'error': 'Method not allowed', 'message': 'The requested method is not allowed for this endpoint'}), 405


@app.route('/', methods=['GET'])
def home():
    """Combined API documentation endpoint"""
    return jsonify({
        'message': 'TryFit API — Virtual Try-On',
        'version': '2.0.0',
        'status': {
            'try_on': 'healthy' if tryon_client else 'unhealthy',
            # 'spin_360': 'healthy' if os.getenv('GEMINI_API_KEY') else 'unhealthy',  # disabled — 360 spin commented out
        },
        'endpoints': {
            '/': {'method': 'GET', 'description': 'API documentation and status'},
            '/health': {'method': 'GET', 'description': 'Check API health status'},
            '/try-on': {
                'method': 'POST',
                'description': 'Upload person and clothing images for virtual try-on (requires Bearer API key)',
                'parameters': {
                    'person_image': 'Image file of person (required)',
                    'clothing_image': 'Image file of clothing (required)',
                    'garment_description': 'Description of the garment (optional)',
                    'category': 'upper_body | lower_body | dresses (optional)'
                },
                'accepted_formats': list(ALLOWED_EXTENSIONS),
                'max_file_size': '16MB'
            },
            '/try-on/status/<request_id>': {'method': 'GET', 'description': 'Poll virtual try-on result (requires Bearer API key)'},
            # '/generate-360': {...},        # disabled — 360 spin commented out
            # '/status/<request_id>': {...}, # disabled — 360 spin commented out
        },
        'models': {
            'try_on': 'virtual-try-on-001',
            # 'spin_360': 'gemini-3.1-flash-image-preview',  # disabled — 360 spin commented out
        }
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check"""
    tryon_healthy = bool(tryon_client)
    # spin_healthy = bool(os.getenv('GEMINI_API_KEY'))   # disabled — 360 spin commented out
    overall_healthy = tryon_healthy  # and spin_healthy

    return jsonify({
        'status': 'healthy' if overall_healthy else 'degraded',
        'services': {
            'try_on': {
                'status': 'healthy' if tryon_healthy else 'unhealthy',
                'message': 'Try-On client initialized' if tryon_healthy else 'Vertex AI client not initialized',
            },
            # 'spin_360': {...},  # disabled — 360 spin commented out
        },
        'timestamp': str(int(time.time())),
        'version': '2.0.0'
    }), 200 if overall_healthy else 503


@app.route('/favicon.ico')
def favicon():
    return '', 204


# Re-enable this to restore the 360 spin cleanup loop:
# def periodic_cleanup():
#     """Background loop clearing stale 360° jobs (try-on results are popped on read)"""
#     while True:
#         time.sleep(600)  # every 10 minutes
#         cleanup_old_spin_jobs()


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'

    # Re-enable this to restore the 360 spin cleanup loop alongside the server:
    # cleanup_thread = Thread(target=periodic_cleanup, daemon=True)
    # cleanup_thread.start()

    if not debug:
        logger.info("Starting production server (try-on)...")

    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        threaded=True
    )