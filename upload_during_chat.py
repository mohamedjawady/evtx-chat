"""
Additional routes to support uploading during chat.
Import this file in app.py
"""
import os
import uuid
import logging
from flask import request, jsonify, url_for
from werkzeug.utils import secure_filename
import persistent_storage

def register_upload_routes(app, process_documents_func):
    """
    Register routes for document upload during chat
    
    Args:
        app: Flask application instance
        process_documents_func: Function to process documents after upload
    """
    
    ALLOWED_EXTENSIONS = {'pdf', 'evtx'}
    
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    @app.route('/upload-in-chat', methods=['POST'])
    def upload_in_chat():
        """Handle document uploads during chat session"""
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file part'
            })
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            })
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Add unique identifier to prevent overwriting
            if os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], filename)):
                name, ext = os.path.splitext(filename)
                filename = f"{name}_{uuid.uuid4().hex[:8]}{ext}"
                
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Store document metadata for persistence
            file_type = filename.rsplit('.', 1)[1].lower()
            persistent_storage.add_document(filename, file_path, file_type)
            
            # Handle EVTX files - convert to text
            if filename.lower().endswith('.evtx'):
                try:
                    from evtx_parser import evtx_to_text
                    
                    # Convert EVTX to text and save as TXT
                    logging.info(f"Processing EVTX file: {filename}")
                    text_content = evtx_to_text(file_path)
                    txt_filename = os.path.splitext(filename)[0] + '.txt'
                    txt_path = os.path.join(app.config['UPLOAD_FOLDER'], txt_filename)
                    
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(text_content)
                    
                    # Add the converted txt file to persistent storage
                    persistent_storage.add_document(txt_filename, txt_path, 'txt')
                    
                    # Process documents after upload
                    process_documents_func()
                    
                    return jsonify({
                        'success': True,
                        'message': f'EVTX file {filename} uploaded and converted to text successfully',
                        'document_name': filename
                    })
                except Exception as e:
                    logging.error(f"Error processing EVTX file {filename}: {str(e)}")
                    return jsonify({
                        'success': False,
                        'error': f'Error processing EVTX file: {str(e)}'
                    })
            else:
                # Process documents after upload
                process_documents_func()
                
                return jsonify({
                    'success': True,
                    'message': f'File {filename} uploaded successfully',
                    'document_name': filename
                })
        else:
            return jsonify({
                'success': False,
                'error': 'Only PDF and EVTX files are allowed'
            })
