import os
import logging
import uuid
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, session
from werkzeug.utils import secure_filename
from utils import (extract_text_from_pdf, chunk_texts, build_vectorstore,
                   retrieve_contexts, enhanced_retrieve_contexts, ask_ollama,
                   load_processed_files, save_processed_files, PROCESSED_FILES)

# Import chat upload routes
from upload_during_chat import register_upload_routes

# Setup logging
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", os.urandom(24).hex())

# Configuration
UPLOAD_FOLDER = os.path.join(os.getcwd(), "docs")
ALLOWED_EXTENSIONS = {'pdf', 'evtx', 'txt'}
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload size

# Memory-based document persistence - no disk writes for metadata
document_registry = []

# Global variables
vectorstore = None
has_documents = False

# Processing status tracking
processing_status = {
    'in_progress': False,
    'progress': 0,
    'total_docs': 0,
    'processed_docs': 0,
    'message': '',
    'complete': False,
    'error': None
}


def allowed_file(filename):
    return '.' in filename and filename.rsplit(
        '.', 1)[1].lower() in ALLOWED_EXTENSIONS


# In-memory document registry functions
def add_document_to_registry(name, file_path, doc_type):
    """Add document to in-memory registry"""
    global document_registry
    try:
        # Check if document already exists
        for doc in document_registry:
            if doc.get('file_path') == file_path:
                return False

        # Add new document
        size_bytes = os.path.getsize(file_path)
        document_registry.append({
            'id': len(document_registry) + 1,
            'name': name,
            'file_path': file_path,
            'doc_type': doc_type,
            'size_bytes': size_bytes,
            'created_at': datetime.utcnow().isoformat()
        })
        return True
    except Exception as e:
        logging.error(f"Error adding document to registry: {str(e)}")
        return False


def remove_document_from_registry(file_path):
    """Remove document from in-memory registry"""
    global document_registry
    try:
        # Filter out the document to remove
        new_registry = [
            doc for doc in document_registry
            if doc.get('file_path') != file_path
        ]

        # Check if anything was removed
        if len(new_registry) < len(document_registry):
            document_registry = new_registry
            return True

        return False
    except Exception as e:
        logging.error(f"Error removing document from registry: {str(e)}")
        return False


def verify_registry_documents():
    """Verify that all document files still exist and return valid ones"""
    global document_registry
    valid_docs = []

    for doc in document_registry:
        file_path = doc.get('file_path')
        if file_path and os.path.exists(file_path):
            valid_docs.append(doc)

    # If some documents are missing, update the registry
    if len(valid_docs) < len(document_registry):
        document_registry = valid_docs

    return valid_docs


def scan_for_documents():
    """Scan the upload folder for documents and add them to the registry"""
    global document_registry

    # Skip if registry already has documents
    if document_registry:
        return

    if os.path.exists(app.config['UPLOAD_FOLDER']):
        pdf_files = [
            f for f in os.listdir(app.config['UPLOAD_FOLDER'])
            if f.endswith('.pdf')
        ]
        evtx_files = [
            f for f in os.listdir(app.config['UPLOAD_FOLDER'])
            if f.endswith('.evtx')
        ]
        txt_files = [
            f for f in os.listdir(app.config['UPLOAD_FOLDER'])
            if f.endswith('.txt')
        ]

        # Register PDFs
        for pdf in pdf_files:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf)
            add_document_to_registry(pdf, file_path, 'pdf')

        # Register EVTXs
        for evtx in evtx_files:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], evtx)
            add_document_to_registry(evtx, file_path, 'evtx')

        # Register text files
        for txt in txt_files:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], txt)
            add_document_to_registry(txt, file_path, 'txt')


@app.route('/')
def index():
    # Check for documents and display them
    global has_documents, processing_status

    # Only scan if we haven't already processed
    if not has_documents:
        scan_for_documents()

    # Verify registry documents and get valid ones
    valid_docs = verify_registry_documents()
    has_documents = len(valid_docs) > 0

    # Get document list for display
    documents = []

    for doc in valid_docs:
        doc_size = doc['size_bytes'] / (1024 * 1024)  # Size in MB
        doc_type = doc['doc_type'].upper()
        documents.append({
            'name': doc['name'],
            'type': doc_type,
            'size': f"{doc_size:.2f} MB"
        })

    return render_template('index.html',
                           documents=documents,
                           has_documents=has_documents,
                           processing_status=processing_status
                           if processing_status['in_progress'] else None)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part', 'danger')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'danger')
        return redirect(request.url)

    # Only allow PDFs in the main upload form - EVTX only allowed in chat
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file and file_ext == '.pdf':
        filename = secure_filename(file.filename)
        # Add unique identifier to prevent overwriting
        if os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], filename)):
            name, ext = os.path.splitext(filename)
            filename = f"{name}_{uuid.uuid4().hex[:8]}{ext}"

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Add document to registry
        add_document_to_registry(filename, file_path, 'pdf')
        flash(f'File {filename} uploaded successfully!', 'success')

        # Process the document to update the vectorstore
        process_documents()

        return redirect(url_for('index'))
    else:
        flash(
            'Only PDF files are allowed in the document manager. Use the chat interface to upload EVTX files.',
            'warning')
        return redirect(request.url)


@app.route('/processing-status', methods=['GET'])
def get_processing_status():
    """Get the current document processing status for AJAX polling"""
    global processing_status

    return jsonify({
        'success': True,
        'in_progress': processing_status['in_progress'],
        'progress': processing_status['progress'],
        'message': processing_status['message'],
        'complete': processing_status['complete'],
        'error': processing_status['error']
    })


@app.route('/process', methods=['POST'])
def process_documents():
    global vectorstore, has_documents, processing_status

    # If this is an AJAX request, return a JSON response
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'

    # Check if processing is already in progress
    if processing_status['in_progress']:
        if is_ajax:
            return jsonify({
                'success':
                False,
                'error':
                'Document processing is already in progress'
            })
        else:
            flash('Document processing is already in progress', 'warning')
            return redirect(url_for('index'))

    try:
        # Reset processing status
        processing_status['in_progress'] = True
        processing_status['progress'] = 0
        processing_status['message'] = 'Starting document processing...'
        processing_status['complete'] = False
        processing_status['error'] = None

        # Get documents from registry
        valid_docs = verify_registry_documents()
        documents_to_process = []

        for doc in valid_docs:
            file_path = doc.get('file_path')
            if file_path.endswith('.pdf') or file_path.endswith('.txt'):
                documents_to_process.append(file_path)

        if not documents_to_process:
            processing_status['in_progress'] = False
            processing_status['error'] = 'No documents found to process'

            if is_ajax:
                return jsonify({
                    'success': False,
                    'error': 'No documents found to process'
                })
            else:
                flash('No documents found to process', 'warning')
                has_documents = False
                return redirect(url_for('index'))

        processing_status['total_docs'] = len(documents_to_process)
        processing_status['processed_docs'] = 0

        # If this is an AJAX request, return a JSON response and continue processing
        if is_ajax:
            from threading import Thread
            processing_thread = Thread(target=process_documents_background,
                                       args=(documents_to_process, ))
            processing_thread.daemon = True
            processing_thread.start()

            return jsonify({
                'success':
                True,
                'message':
                f'Processing {len(documents_to_process)} documents in background'
            })

        # For non-AJAX requests, process documents directly
        return process_documents_sync(documents_to_process)

    except Exception as e:
        logging.error(f"Error starting document processing: {str(e)}")
        processing_status['in_progress'] = False
        processing_status['error'] = str(e)

        if is_ajax:
            return jsonify({
                'success':
                False,
                'error':
                f'Error starting document processing: {str(e)}'
            })
        else:
            flash(f'Error processing documents: {str(e)}', 'danger')
            return redirect(url_for('index'))


def process_documents_background(documents_to_process):
    """Process documents in a background thread"""
    try:
        process_documents_sync(documents_to_process, is_background=True)
    except Exception as e:
        logging.error(f"Error in background document processing: {str(e)}")
        processing_status['in_progress'] = False
        processing_status['error'] = str(e)
        processing_status['complete'] = True


def process_documents_sync(documents_to_process, is_background=False):
    """Process documents synchronously"""
    global vectorstore, has_documents, processing_status, PROCESSED_FILES
    if not documents_to_process:
        processing_status['error'] = 'No documents to process'
        return

    # Filter out already processed files
    documents_to_process = [doc for doc in documents_to_process if doc not in PROCESSED_FILES]
    if not documents_to_process:
        processing_status['message'] = 'All documents already processed'
        processing_status['progress'] = 100
        processing_status['complete'] = True
        return

    # Initialize flag to track if processing was already started
    if not hasattr(process_documents_sync, 'processing_started'):
        process_documents_sync.processing_started = False

    try:
        # Extract text from PDFs and text files
        texts = []

        # Process documents
        for i, doc_path in enumerate(documents_to_process):
            # Update progress (text extraction phase - 50% of total)
            progress_value = int((i / len(documents_to_process)) * 50)
            processing_status['progress'] = progress_value
            processing_status[
                'message'] = f'Extracting text from document {i+1} of {len(documents_to_process)}...'

            if doc_path.endswith('.pdf'):
                text = extract_text_from_pdf(doc_path)
                if text:
                    texts.append(text)
                    logging.info(
                        f"Processed PDF: {os.path.basename(doc_path)}: {len(text)} characters"
                    )
            elif doc_path.endswith('.txt'):
                try:
                    with open(doc_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    if text:
                        texts.append(text)
                        logging.info(
                            f"Processed text file: {os.path.basename(doc_path)}: {len(text)} characters"
                        )
                except Exception as e:
                    logging.error(
                        f"Error reading text file {doc_path}: {str(e)}")

            processing_status['processed_docs'] = i + 1

        if not texts:
            processing_status['in_progress'] = False
            processing_status[
                'error'] = 'Could not extract text from any of the document files'
            processing_status['complete'] = True

            if not is_background:
                flash('Could not extract text from any of the document files',
                      'danger')
                return redirect(url_for('index'))

        # Update progress (chunking phase - 75%)
        processing_status['progress'] = 75
        processing_status['message'] = 'Splitting text into chunks...'

        # Create chunks
        chunks = chunk_texts(texts)

        # Update progress (vectorstore creation phase - 90%)
        processing_status['progress'] = 90
        processing_status['message'] = 'Building vector store...'

        # Always rebuild vectorstore with new chunks
        global vectorstore, has_documents
        vectorstore = build_vectorstore(chunks)
        has_documents = True

        # Add processed files to cache
        for doc in documents_to_process:
            PROCESSED_FILES.add(doc)
        try:
            save_processed_files()
            logging.info(f"Updated cache with {len(documents_to_process)} new files")
        except Exception as e:
            logging.error(f"Failed to save processed files cache: {e}")

        # Mark as complete
        processing_status['progress'] = 100
        processing_status[
            'message'] = f'Successfully processed {len(documents_to_process)} documents with {len(chunks)} chunks'
        processing_status['complete'] = True
        processing_status['in_progress'] = False

        if not is_background:
            flash(
                f'Successfully processed {len(documents_to_process)} documents',
                'success')
            return redirect(url_for('index'))

    except Exception as e:
        logging.error(f"Error processing documents: {str(e)}")
        processing_status['in_progress'] = False
        processing_status['error'] = str(e)
        processing_status['complete'] = True

        if not is_background:
            flash(f'Error processing documents: {str(e)}', 'danger')
            return redirect(url_for('index'))


@app.route('/ask', methods=['POST'])
def ask_question():
    global vectorstore, has_documents

    if not has_documents or vectorstore is None:
        return jsonify({
            'success': False,
            'error': 'No documents have been processed yet. Please upload and process documents first.'
        })

    question = request.json.get('question', '').strip()
    if not question:
        return jsonify({
            'success': False,
            'error': 'Question cannot be empty.'
        })

    try:
        # Use enhanced retrieval techniques
        retriever = vectorstore.as_retriever()

        # Get enhanced contexts using multiple RAG techniques
        retrieval_result = enhanced_retrieve_contexts(question, retriever)

        # Extract the results
        contexts = retrieval_result["contexts"]
        context_text = retrieval_result["context_text"]
        techniques_used = retrieval_result["techniques_used"]

        # Generate answer from Ollama with techniques information
        answer = ask_ollama(question, context_text, techniques_used)

        # Ensure the answer is a string, not a JSON structure
        if isinstance(answer, (dict, list)):
            answer = json.dumps(answer)

        response_data = {
            'success': True,
            'question': question,
            'answer': answer,
            'contexts': contexts,
            'techniques_used': techniques_used
        }

        # Ensure the entire response can be properly serialized
        try:
            # Test serialize
            json.dumps(response_data)
            return jsonify(response_data)
        except (TypeError, ValueError) as json_err:
            # If serialization fails, sanitize the response
            logging.error(f"JSON serialization error: {str(json_err)}")
            safe_contexts = [{'content': str(c.get('content', '')), 
                             'score': float(c.get('score', 0)), 
                             'method': str(c.get('method', 'unknown'))} 
                            for c in contexts]
            return jsonify({
                'success': True,
                'question': question,
                'answer': str(answer),
                'contexts': safe_contexts,
                'techniques_used': [str(t) for t in techniques_used]
            })

    except Exception as e:
        logging.error(f"Error answering question: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Error processing your question: {str(e)}'
        })

@app.route('/delete/<filename>', methods=['POST'])
def delete_document(filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'],
                                 secure_filename(filename))
        if os.path.exists(file_path):
            os.remove(file_path)

            # Remove from registry
            remove_document_from_registry(file_path)

            flash(f'Document {filename} deleted successfully', 'success')
            # Reprocess remaining documents
            process_documents()
        else:
            flash(f'Document {filename} not found', 'danger')
    except Exception as e:
        flash(f'Error deleting document: {str(e)}', 'danger')

    return redirect(url_for('index'))

# Register upload routes after process_documents is defined
register_upload_routes(app, process_documents)


def preprocess_existing_documents():
    """Pre-process all existing documents in the docs folder"""
    logging.info("Pre-processing existing documents...")
    with app.app_context():
        try:
            # Load processed files cache
            global PROCESSED_FILES
            PROCESSED_FILES = load_processed_files()

            # Scan for existing documents
            scan_for_documents()

            # Get unprocessed documents
            valid_docs = verify_registry_documents()
            documents_to_process = []

            for doc in valid_docs:
                file_path = doc.get('file_path')
                if file_path not in PROCESSED_FILES and (file_path.endswith('.pdf') or file_path.endswith('.txt')):
                    documents_to_process.append(file_path)

            if documents_to_process:
                process_documents_sync(documents_to_process)

            logging.info(f"Pre-processing complete. {len(PROCESSED_FILES)} files in cache.")
        except Exception as e:
            logging.error(f"Error during pre-processing: {str(e)}")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler('error.log'),
                  logging.StreamHandler()])

    # Pre-process existing documents
    preprocess_existing_documents()

    # Start the Flask app
    extra_files = [f for f in app.jinja_loader.list_templates()]
    extra_dirs = ['templates/', 'static/']
    extra_files = extra_files + extra_dirs
    app.run(host="0.0.0.0", port=8888, debug=True, extra_files=extra_files, use_reloader=True)