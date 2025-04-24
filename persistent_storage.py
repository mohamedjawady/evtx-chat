"""
File-based persistent storage for document indexing.
This module provides functionality to maintain document persistence even after application restarts.
"""
import os
import json
import logging
from datetime import datetime

# File to store document metadata
DOCUMENT_METADATA_FILE = os.path.join(os.getcwd(), "docs", "document_metadata.json")

def ensure_metadata_file_exists():
    """Ensure that the document metadata file exists"""
    os.makedirs(os.path.dirname(DOCUMENT_METADATA_FILE), exist_ok=True)
    if not os.path.exists(DOCUMENT_METADATA_FILE):
        with open(DOCUMENT_METADATA_FILE, 'w') as f:
            json.dump([], f)

def load_document_metadata():
    """Load document metadata from file"""
    ensure_metadata_file_exists()
    try:
        with open(DOCUMENT_METADATA_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading document metadata: {str(e)}")
        return []

def save_document_metadata(metadata_list):
    """Save document metadata to file"""
    ensure_metadata_file_exists()
    try:
        with open(DOCUMENT_METADATA_FILE, 'w') as f:
            json.dump(metadata_list, f)
    except Exception as e:
        logging.error(f"Error saving document metadata: {str(e)}")

def add_document(name, file_path, doc_type):
    """Add document metadata to the persistent storage"""
    try:
        metadata_list = load_document_metadata()
        
        # Check if document already exists
        for doc in metadata_list:
            if doc.get('file_path') == file_path:
                return False
        
        # Add new document
        size_bytes = os.path.getsize(file_path)
        metadata_list.append({
            'id': len(metadata_list) + 1,
            'name': name,
            'file_path': file_path,
            'doc_type': doc_type,
            'size_bytes': size_bytes,
            'created_at': datetime.utcnow().isoformat()
        })
        
        save_document_metadata(metadata_list)
        return True
    except Exception as e:
        logging.error(f"Error adding document metadata: {str(e)}")
        return False

def remove_document(file_path):
    """Remove document metadata from the persistent storage"""
    try:
        metadata_list = load_document_metadata()
        
        # Filter out the document to remove
        new_metadata_list = [doc for doc in metadata_list if doc.get('file_path') != file_path]
        
        # Check if anything was removed
        if len(new_metadata_list) < len(metadata_list):
            save_document_metadata(new_metadata_list)
            return True
        
        return False
    except Exception as e:
        logging.error(f"Error removing document metadata: {str(e)}")
        return False

def get_all_documents():
    """Get all document metadata"""
    return load_document_metadata()

def verify_document_files():
    """Verify that all document files still exist"""
    metadata_list = load_document_metadata()
    valid_docs = []
    
    for doc in metadata_list:
        file_path = doc.get('file_path')
        if file_path and os.path.exists(file_path):
            valid_docs.append(doc)
    
    # If some documents are missing, update the metadata file
    if len(valid_docs) < len(metadata_list):
        save_document_metadata(valid_docs)
    
    return valid_docs
