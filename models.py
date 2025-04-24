from app import db
from datetime import datetime
import os
import json

class IndexedDocument(db.Model):
    """Representation of a document that has been indexed in the system"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(255), nullable=False)
    file_type = db.Column(db.String(50), nullable=False)  # PDF, EVTX, TXT
    size_bytes = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    chunks = db.relationship('DocumentChunk', backref='document', lazy=True, cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<IndexedDocument {self.name}>"
    
    def get_size_display(self):
        """Return human-readable file size"""
        size_mb = self.size_bytes / (1024 * 1024)
        return f"{size_mb:.2f} MB"
        
    @staticmethod
    def add_document(name, file_path, file_type):
        """Create and return a new document record"""
        try:
            size_bytes = os.path.getsize(file_path)
            doc = IndexedDocument(
                name=name,
                file_path=file_path,
                file_type=file_type,
                size_bytes=size_bytes
            )
            db.session.add(doc)
            db.session.commit()
            return doc
        except Exception as e:
            db.session.rollback()
            raise e

class DocumentChunk(db.Model):
    """Representation of a document chunk in the database"""
    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.Integer, db.ForeignKey('indexed_document.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    chunk_index = db.Column(db.Integer, nullable=False)
    metadata = db.Column(db.Text, nullable=True)  # JSON serialized metadata
    
    def __repr__(self):
        return f"<DocumentChunk {self.id} for document {self.document_id}>"
        
    def get_metadata_dict(self):
        """Return the metadata as a dictionary"""
        if not self.metadata:
            return {}
        try:
            return json.loads(self.metadata)
        except:
            return {}
            
    def set_metadata_dict(self, metadata_dict):
        """Set the metadata from a dictionary"""
        if metadata_dict:
            self.metadata = json.dumps(metadata_dict)
        else:
            self.metadata = None

# Keep for compatibility with utils.py Document class
class Document:
    """
    Compatibility class for the Document class used in utils.py
    """
    def __init__(self, id=None, name=None, path=None, size=None, chunks=None, page_content=None, metadata=None):
        self.id = id
        self.name = name
        self.path = path
        self.size = size
        self.chunks = chunks or []
        
        # For compatibility with utils.Document
        self.page_content = page_content
        self.metadata = metadata or {}

class Context:
    """
    Representation of a context chunk retrieved from the vector database.
    """
    def __init__(self, text, source=None, score=None):
        self.text = text
        self.source = source
        self.score = score
