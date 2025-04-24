import os
import re
import logging
import json
import math
import hashlib
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple, Union, Callable


# Simple implementation of LLM interfaces for compatibility
class OllamaLLM:
    """Simple wrapper for Ollama API calls"""

    def __init__(self, model="llama3", temperature=0.7):
        self.model = model
        self.temperature = temperature
        self.base_url = "http://localhost:11434/api"

    def invoke(self, prompt):
        """Send prompt to Ollama API and return response using urllib instead of requests"""

        # For compatibility with various prompt formats
        if hasattr(prompt, 'format'):
            # It's likely a template
            formatted_prompt = prompt
        elif isinstance(prompt, dict) and 'content' in prompt:
            # It might be a chat message format
            formatted_prompt = prompt['content']
        else:
            # Assume it's a string
            formatted_prompt = str(prompt)

        # Prepare the request data
        data = {
            "model": self.model,
            "prompt": formatted_prompt,
            "temperature": self.temperature,
            "stream": False
        }

        # Convert data to JSON string and encode as bytes
        data_json = json.dumps(data).encode('utf-8')

        # Create a POST request
        req = urllib.request.Request(
            f"{self.base_url}/generate",
            data=data_json,
            headers={'Content-Type': 'application/json'}
        )

        # Send the request
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                # Read and decode the response
                response_data = response.read().decode('utf-8')
                response_json = json.loads(response_data)
                return response_json.get("response", "No response generated")
        except urllib.error.HTTPError as e:
            logging.warning(f"Ollama API returned error: {e.code} - {e.reason}")
            return f"Error: Ollama API returned error: {e.code} - {e.reason}"
        except urllib.error.URLError as e:
            logging.error(f"Failed to connect to Ollama API: {str(e.reason)}")
            return "I apologize, but I'm unable to connect to the Ollama service. Please ensure Ollama is running locally with the appropriate model."

        # except Exception as e:
        #     logging.error(f"Error in Ollama LLM: {str(e)}")
        #     return f"Error generating response: {str(e)}"


class ChatPromptTemplate:
    """Simple template for chat prompts"""

    @classmethod
    def from_template(cls, template_string):
        """Create a template from a string"""
        return cls(template_string)

    def __init__(self, template_string):
        self.template = template_string

    def format(self, **kwargs):
        """Format the template with the given arguments"""
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            logging.error(f"Missing key in template formatting: {e}")
            return f"Error in template: missing key {e}"
        except Exception as e:
            logging.error(f"Error formatting template: {e}")
            return f"Error formatting template: {e}"

# Simple fallback for tokenizers
def simple_sent_tokenize(text):
    """Simple sentence tokenizer fallback"""
    # Simple regex-based sentence splitting
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def simple_word_tokenize(text):
    """Simple word tokenizer fallback"""
    # Simple regex-based word splitting
    return re.findall(r'\b\w+\b', text.lower())

# Common English stopwords
ENGLISH_STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
    'which', 'this', 'that', 'these', 'those', 'then', 'just', 'so', 'than',
    'such', 'both', 'through', 'about', 'for', 'is', 'of', 'while', 'during',
    'to', 'from', 'in', 'on', 'at', 'by', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'up',
    'down', 'with', 'without', 'be', 'am', 'is', 'are', 'was', 'were', 'been',
    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
    'can', 'could', 'would', 'should', 'will', 'shall', 'may', 'might', 'must',
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
    'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'hers',
    'ours', 'theirs'
}

# Constants
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5  # Number of retrieved chunks
MAX_SUB_QUERIES = 3  # Maximum number of sub-queries for decomposition
MULTI_QUERY_COUNT = 3  # Number of queries to generate in multi-query approach

# Cache for processed files
PROCESSED_FILES = set()
CACHE_FILE = os.path.join(os.getcwd(), "docs", "processed_files.json")

def load_processed_files():
    """Load the set of processed files from cache"""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return set(json.load(f))
    return set()

def save_processed_files():
    """Save the set of processed files to cache"""
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, 'w') as f:
        json.dump(list(PROCESSED_FILES), f)

@dataclass
class Document:
    """Simplified document class to replace langchain's Document"""
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Generate a document ID based on content hash
        self.id = hashlib.md5(self.page_content.encode()).hexdigest()

# Ollama API endpoint (default for local deployment)
OLLAMA_API = "http://localhost:11434/api"

def clean_text(text):
    """Cleans extracted text by removing extra newlines and page numbers."""
    text = re.sub(r'\n\s*\n', '\n', text)  # Remove excessive newlines
    text = re.sub(r'Page \d+', '', text)  # Remove page numbers
    return text.strip()

def extract_text_from_pdf(filepath):
    """
    Extracts text from a single PDF file using simple file reading.
    This is a fallback implementation when pdfplumber is not available.
    """
    try:
        # Simple text extraction - read file as binary and look for text
        with open(filepath, 'rb') as f:
            content = f.read()

        # Extract all text between relevant markers in PDF
        text_parts = []

        # Try to decode text directly
        try:
            # Common text block pattern in PDFs
            text = content.decode('utf-8', errors='ignore')

            # Clean up text
            text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
            text = re.sub(r'\s+', ' ', text)

            # Extract strings that look like sentences
            sentences = re.findall(r'[A-Z][^.!?]*[.!?]', text)
            if sentences:
                text_parts.extend(sentences)

        except Exception as decode_error:
            logging.error(f"Error decoding PDF text: {decode_error}")

        # If we got some text
        if text_parts:
            text = " ".join(text_parts)
            return clean_text(text)
        else:
            logging.warning(f"No text extracted from {os.path.basename(filepath)}")
            # Return placeholder text so processing can continue
            return f"[PDF content from {os.path.basename(filepath)} - Text extraction unavailable]"

    except Exception as e:
        logging.error(f"Error processing {os.path.basename(filepath)}: {e}")
        return f"[Error processing PDF: {os.path.basename(filepath)}]"

def load_pdfs(directory):
    """Loads and extracts text from all PDFs in the specified directory using parallel processing."""
    pdf_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".pdf")]

    if not pdf_files:
        logging.error(f"❌ No PDF files found in {directory}")
        return []

    texts = []
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(extract_text_from_pdf, pdf_files))

    for filename, text in zip(pdf_files, results):
        if text:
            texts.append(text)
            logging.info(f"✅ Processed: {os.path.basename(filename)} ({len(text)} characters)")

    return texts

def chunk_texts(texts):
    """Splits texts into overlapping chunks for better retrieval."""
    all_chunks = []

    for i, text in enumerate(texts):
        # Split text into sentences using simple method
        sentences = simple_sent_tokenize(text)

        # Group sentences into chunks with overlap
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            # If adding this sentence would exceed chunk size and we already have content,
            # store the current chunk and start a new one with overlap
            if current_size + sentence_size > CHUNK_SIZE and current_chunk:
                # Create document from current chunk
                chunk_text = " ".join(current_chunk)
                all_chunks.append(Document(
                    page_content=chunk_text,
                    metadata={"chunk_id": len(all_chunks), "source_idx": i}
                ))

                # Start new chunk with overlap (keep last few sentences)
                overlap_sentences = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk
                current_chunk = overlap_sentences
                current_size = sum(len(s) for s in overlap_sentences)

            # Add the current sentence to the chunk
            current_chunk.append(sentence)
            current_size += sentence_size

        # Don't forget the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            all_chunks.append(Document(
                page_content=chunk_text,
                metadata={"chunk_id": len(all_chunks), "source_idx": i}
            ))

    logging.info(f"🔹 Created {len(all_chunks)} chunks from {len(texts)} documents.")
    return all_chunks

def calculate_tfidf(documents):
    """Calculate TF-IDF scores for document terms."""
    # First, tokenize all documents
    tokenized_docs = []
    all_tokens = set()

    for doc in documents:
        # Tokenize, convert to lowercase, and remove stopwords and punctuation
        tokens = [t.lower() for t in simple_word_tokenize(doc.page_content) 
                 if t.lower() not in ENGLISH_STOPWORDS and t.isalnum()]
        tokenized_docs.append(tokens)
        all_tokens.update(tokens)

    # Calculate term frequencies for each document
    tf = []
    for tokens in tokenized_docs:
        doc_tf = {}
        for token in tokens:
            doc_tf[token] = doc_tf.get(token, 0) + 1
        # Normalize by document length
        doc_len = len(tokens)
        if doc_len > 0:
            for token in doc_tf:
                doc_tf[token] /= doc_len
        tf.append(doc_tf)

    # Calculate inverse document frequency
    idf = {}
    N = len(documents)
    for token in all_tokens:
        # Count documents containing the token
        doc_count = sum(1 for doc_tf in tf if token in doc_tf)
        # Add smoothing to avoid division by zero
        idf[token] = math.log(N / (1 + doc_count)) + 1

    # Calculate TF-IDF for each document
    tfidf = []
    for doc_tf in tf:
        doc_tfidf = {}
        for token, term_tf in doc_tf.items():
            doc_tfidf[token] = term_tf * idf[token]
        tfidf.append(doc_tfidf)

    return tfidf, all_tokens

class SimpleVectorStore:
    """A simple vector store implementation using TF-IDF."""

    def __init__(self, documents):
        self.documents = documents
        self.tfidf, self.vocabulary = calculate_tfidf(documents)

    def similarity(self, query_tfidf, doc_tfidf):
        """Calculate cosine similarity between query and document"""
        # Find common tokens
        common_tokens = set(query_tfidf.keys()) & set(doc_tfidf.keys())

        if not common_tokens:
            return 0.0

        # Calculate dot product
        dot_product = sum(query_tfidf[token] * doc_tfidf[token] for token in common_tokens)

        # Calculate magnitudes
        query_magnitude = math.sqrt(sum(v**2 for v in query_tfidf.values()))
        doc_magnitude = math.sqrt(sum(v**2 for v in doc_tfidf.values()))

        # Avoid division by zero
        if query_magnitude == 0 or doc_magnitude == 0:
            return 0.0

        return dot_product / (query_magnitude * doc_magnitude)

    def search(self, query, k=5):
        """Search for documents similar to the query"""
        # Tokenize query
        query_tokens = [t.lower() for t in simple_word_tokenize(query) 
                     if t.lower() not in ENGLISH_STOPWORDS and t.isalnum()]

        # Calculate query TF
        query_tf = {}
        for token in query_tokens:
            query_tf[token] = query_tf.get(token, 0) + 1

        # Normalize by query length
        query_len = len(query_tokens)
        if query_len > 0:
            for token in query_tf:
                query_tf[token] /= query_len

        # Calculate query TF-IDF
        query_tfidf = {}
        for token, tf in query_tf.items():
            if token in self.vocabulary:
                query_tfidf[token] = tf * (self.tfidf[0].get(token, 0) / tf if tf > 0 else 0)

        # Calculate similarity scores
        scores = []
        for i, doc_tfidf in enumerate(self.tfidf):
            score = self.similarity(query_tfidf, doc_tfidf)
            scores.append((i, score))

        # Sort by score descending and get top k
        top_results = sorted(scores, key=lambda x: x[1], reverse=True)[:k]

        # Return documents
        results = []
        for idx, score in top_results:
            doc = self.documents[idx]
            # Add score to metadata
            doc_with_score = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "score": score}
            )
            results.append(doc_with_score)

        return results

    def as_retriever(self):
        """Return self as a retriever object"""
        return self

    def get_relevant_documents(self, query, k=5):
        """Retrieve relevant documents for a query"""
        return self.search(query, k)

def build_vectorstore(chunks):
    """Builds a FAISS vector store from document chunks."""
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer
    import os
    import pickle

    # Initialize the encoder
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    # encoder = SentenceTransformer('all-MiniLM-L6-v2').to('cuda')

    # Convert documents to embeddings
    documents = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    # embeddings = encoder.encode(documents, batch_size=14, show_progress_bar=True)
    embeddings = encoder.encode(documents, batch_size=14, show_progress_bar=True, convert_to_numpy=True, convert_to_tensor=False, num_workers=4)

    # embeddings = encoder.encode(documents, device='cuda')

    # Normalize the vectors
    faiss.normalize_L2(embeddings)

    # Build the FAISS index
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # Save index and metadata
    if not os.path.exists('faiss_store'):
        os.makedirs('faiss_store')

    faiss.write_index(index, 'faiss_store/docs.index')
    with open('faiss_store/metadata.pkl', 'wb') as f:
        pickle.dump({'documents': documents, 'metadata': metadatas}, f)

    # Return a retriever-compatible wrapper
    class FaissRetriever:
        def __init__(self, index, documents, metadatas, encoder):
            self.index = index
            self.documents = documents
            self.metadatas = metadatas
            self.encoder = encoder

        def get_relevant_documents(self, query, k=5):
            # Encode and normalize the query
            query_vector = self.encoder.encode([query])
            faiss.normalize_L2(query_vector)

            # Search
            D, I = self.index.search(query_vector, k)

            # Convert to documents
            docs = []
            for i, (idx, score) in enumerate(zip(I[0], D[0])):
                doc = Document(
                    page_content=self.documents[idx],
                    metadata={
                        **self.metadatas[idx],
                        'score': float(score)  # Score is already a similarity
                    }
                )
                docs.append(doc)
            return docs

        def as_retriever(self):
            return self

    return FaissRetriever(index, documents, metadatas, encoder)

def retrieve_contexts(query, retriever):
    """Retrieves relevant document chunks for a given query."""
    results = retriever.get_relevant_documents(query, k=TOP_K)
    return "\n\n".join([doc.page_content for doc in results]) if results else "No relevant context found."

def generate_multi_queries(question, model=None):
    """
    Transforms a single user query into multiple diverse queries to broaden the search space.

    Args:
        question: The original user question
        model: Optional LLM model to use (if None, uses Ollama)

    Returns:
        List of diverse queries related to the original question
    """
    try:
        # If no model is provided, use Ollama
        if model is None:
            model = OllamaLLM(model="deepseek-r1:7b", temperature=0.7)  # Higher temperature for diversity

        prompt = ChatPromptTemplate.from_template(
            """You are a threat hunting assistant that helps users search for information effectively.

            Original Question: {question}

            Please generate {count} different search queries that could help find relevant information about this topic.
            Make the queries diverse to cover different aspects, terminology, and perspectives of the same question.
            Focus on threat hunting relevant information.

            Format your response as a JSON array of strings, with each string being a search query.
            Example: ["query 1", "query 2", "query 3"]

            Do not include any other explanation or text outside the JSON array.
            """
        )

        result = model.invoke(prompt.format(question=question, count=MULTI_QUERY_COUNT))

        # Parse the JSON response
        try:
            # Clean up the response to ensure it's valid JSON
            result = result.strip()
            if result.startswith("```json"):
                result = result[7:]
            if result.endswith("```"):
                result = result[:-3]
            result = result.strip()

            queries = json.loads(result)

            # Ensure we have at least one query (the original)
            if not queries or not isinstance(queries, list):
                return [question]

            # Add the original question if it's not already in the list
            if question not in queries:
                queries.append(question)

            # Return up to MULTI_QUERY_COUNT queries
            return queries[:MULTI_QUERY_COUNT]

        except json.JSONDecodeError:
            logging.warning(f"Could not parse multi-query response as JSON: {result}")
            return [question]  # Return original question on error

    except Exception as e:
        logging.error(f"Error generating multiple queries: {str(e)}")
        return [question]  # Return original question on error


def decompose_query(question, model=None):
    """
    Breaks down complex queries into simpler sub-queries.

    Args:
        question: The original complex question
        model: Optional LLM model to use (if None, uses Ollama)

    Returns:
        List of sub-queries
    """
    try:
        # If no model is provided, use Ollama
        if model is None:
            model = OllamaLLM(model="deepseek-r1:7b", temperature=0)

        prompt = ChatPromptTemplate.from_template(
            """You are a threat hunting assistant that helps break down complex questions.

            Original Question: {question}

            Please break down this question into {max_subqueries} or fewer simpler, more specific questions 
            that together would help answer the original question comprehensively.
            Focus on threat hunting relevant breakdown.

            Format your response as a JSON array of strings, with each string being a sub-question.
            Example: ["sub-question 1", "sub-question 2", "sub-question 3"]

            Do not include any other explanation or text outside the JSON array.
            """
        )

        result = model.invoke(prompt.format(question=question, max_subqueries=MAX_SUB_QUERIES))

        # Parse the JSON response
        try:
            # Clean up the response to ensure it's valid JSON
            result = result.strip()
            if result.startswith("```json"):
                result = result[7:]
            if result.endswith("```"):
                result = result[:-3]
            result = result.strip()

            sub_queries = json.loads(result)

            # Ensure we have at least one query (the original)
            if not sub_queries or not isinstance(sub_queries, list):
                return [question]

            # Return up to MAX_SUB_QUERIES sub-queries
            return sub_queries[:MAX_SUB_QUERIES]

        except json.JSONDecodeError:
            logging.warning(f"Could not parse query decomposition response as JSON: {result}")
            return [question]  # Return original question on error

    except Exception as e:
        logging.error(f"Error decomposing query: {str(e)}")
        return [question]  # Return original question on error


def step_back_retrieval(question, retriever, model=None):
    """
    Implements the 'step back' mechanism that first seeks to understand broader context.

    Args:
        question: The original user question
        retriever: The vector store retriever
        model: Optional LLM model to use (if None, uses Ollama)

    Returns:
        Combined context with both high-level and specific information
    """
    try:
        # If no model is provided, use Ollama
        if model is None:
            model = OllamaLLM(model="deepseek-r1:7b", temperature=0)

        # Step 1: Generate a "step back" question to get broader context
        prompt = ChatPromptTemplate.from_template(
            """You are a threat hunting assistant that helps users think more broadly about a question.

            Original Question: {question}

            Before answering this specific question, I need to understand the broader context.
            Please rewrite this question to ask about the general topic, category, or concept that this question falls under.
            This should be a broader, more general question that would give background knowledge needed to understand the original question.
            Make it threat hunting relevant.

            For example:
            - If asked "What are the indicators of compromise for Emotet malware?", the broader question might be "What is Emotet malware and how does it typically operate?"
            - If asked "How to detect pass-the-hash attacks in Windows logs?", the broader question might be "What are pass-the-hash attacks and what systems do they target?"

            Provide ONLY the rewritten broader question, with no other text.
            """
        )

        broader_question = model.invoke(prompt.format(question=question))
        broader_question = broader_question.strip()

        # Step 2: Retrieve context for both the broader question and the original question
        broader_contexts = retriever.get_relevant_documents(broader_question, k=3)  # Get fewer for broader context
        specific_contexts = retriever.get_relevant_documents(question, k=TOP_K - 2)  # Adjust to keep total at TOP_K

        # Step 3: Combine contexts, removing duplicates
        seen_texts = set()
        combined_contexts = []

        # Add broader contexts first with a prefix
        for doc in broader_contexts:
            if doc.page_content not in seen_texts:
                # Add a prefix to indicate this is broader context
                modified_doc = Document(
                    page_content="[BROADER CONTEXT] " + doc.page_content,
                    metadata=doc.metadata
                )
                combined_contexts.append(modified_doc)
                seen_texts.add(doc.page_content)

        # Add specific contexts
        for doc in specific_contexts:
            if doc.page_content not in seen_texts:
                combined_contexts.append(doc)
                seen_texts.add(doc.page_content)

        # Format the combined context
        return combined_contexts

    except Exception as e:
        logging.error(f"Error in step-back retrieval: {str(e)}")
        # Fall back to standard retrieval
        return retriever.get_relevant_documents(question, k=TOP_K)


def adaptive_retrieval(question, retriever, model=None):
    """
    Adaptively retrieves information based on query complexity.

    Args:
        question: The user's question
        retriever: The vector store retriever
        model: Optional LLM model to use

    Returns:
        List of retrieved documents with metadata
    """
    try:
        # Determine query complexity
        is_complex = len(question.split()) > 15 or "?" in question.split("?")[1:]

        # For complex queries, use multiple retrieval strategies
        if is_complex:
            # First try query decomposition and multi-retrieval
            sub_queries = decompose_query(question, model)

            # Retrieve for each sub-query with a smaller number of results
            results_per_query = max(1, TOP_K // len(sub_queries))

            all_docs = []
            for sub_q in sub_queries:
                docs = retriever.get_relevant_documents(sub_q, k=results_per_query)
                all_docs.extend(docs)

            # Remove duplicates while preserving order
            seen = set()
            unique_docs = []
            for doc in all_docs:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    unique_docs.append(doc)

            # Return a maximum of TOP_K documents
            return unique_docs[:TOP_K]
        else:
            # Try both standard retrieval and step-back retrieval if query is simple
            standard_docs = retriever.get_relevant_documents(question, k=TOP_K)

            # Check if we have enough good results
            if len(standard_docs) >= TOP_K // 2:
                return standard_docs
            else:
                # Fall back to step-back retrieval for better results
                return step_back_retrieval(question, retriever, model)

    except Exception as e:
        logging.error(f"Error in adaptive retrieval: {str(e)}")
        # Fall back to standard retrieval
        return retriever.get_relevant_documents(question, k=TOP_K)


def enhanced_retrieve_contexts(question, retriever, model=None, use_multi_query=True):
    """
    Enhanced context retrieval using multiple RAG techniques.

    Args:
        question: The user's question
        retriever: The vector store retriever
        model: Optional LLM model to use
        use_multi_query: Whether to use multi-query approach

    Returns:
        Dictionary with contexts and technique information
    """
    try:
        all_docs = []
        techniques_used = []

        # Step 1: Multi-query retrieval (optional)
        if use_multi_query:
            queries = generate_multi_queries(question, model)
            if len(queries) > 1:
                techniques_used.append("Multi-Query Retrieval")

                # Adjust results per query to keep total reasonable
                results_per_query = max(1, TOP_K // len(queries))

                multi_query_docs = []
                for q in queries:
                    docs = retriever.get_relevant_documents(q, k=results_per_query)
                    multi_query_docs.extend(docs)

                # Remove duplicates while preserving order
                seen = set()
                for doc in multi_query_docs:
                    if doc.page_content not in seen:
                        seen.add(doc.page_content)
                        doc.metadata["retrieval_method"] = "multi_query"
                        all_docs.append(doc)

        # Step 2: Adaptive retrieval based on query complexity
        adaptive_docs = adaptive_retrieval(question, retriever, model)
        # Mark the source of these documents
        for doc in adaptive_docs:
            if "retrieval_method" not in doc.metadata:
                doc.metadata["retrieval_method"] = "adaptive"

        # Add adaptive retrieval docs and note the technique
        techniques_used.append("Adaptive Retrieval")

        # Combine all unique docs from different methods
        seen = set(doc.page_content for doc in all_docs)
        for doc in adaptive_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                all_docs.append(doc)

        # Step 3: If we still don't have enough, try step-back retrieval as a backup
        if len(all_docs) < TOP_K - 1:
            step_back_docs = step_back_retrieval(question, retriever, model)
            techniques_used.append("Step-Back Retrieval")

            # Add unique step-back docs
            for doc in step_back_docs:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    if "retrieval_method" not in doc.metadata:
                        doc.metadata["retrieval_method"] = "step_back"
                    all_docs.append(doc)

        # Limit to TOP_K total documents
        final_docs = all_docs[:TOP_K]

        # Format the contexts for display
        formatted_contexts = []
        for i, doc in enumerate(final_docs):
            context_obj = {
                'content': doc.page_content,
                'score': round((1 - (i * 0.1)) * 100),  # Simple score based on position
                'method': doc.metadata.get("retrieval_method", "standard"),
                'source': doc.metadata.get("source_idx", "unknown"),
                'document': os.path.basename(doc.metadata.get("file_path", "unknown"))
            }
            formatted_contexts.append(context_obj)

        # Create context text for LLM
        context_text = "\n\n".join([doc.page_content for doc in final_docs])

        return {
            "contexts": formatted_contexts,
            "context_text": context_text,
            "techniques_used": techniques_used
        }

    except Exception as e:
        logging.error(f"Error in enhanced context retrieval: {str(e)}")
        # Fall back to standard retrieval
        standard_docs = retriever.get_relevant_documents(question, k=TOP_K)

        formatted_contexts = []
        for i, doc in enumerate(standard_docs):
            context_obj = {
                'content': doc.page_content,
                'score': round((1 - (i * 0.1)) * 100),
                'method': "standard",
                'source': doc.metadata.get("source_idx", "unknown"),
                'document': os.path.basename(doc.metadata.get("file_path", "unknown"))
            }
            formatted_contexts.append(context_obj)

        context_text = "\n\n".join([doc.page_content for doc in standard_docs])

        return {
            "contexts": formatted_contexts,
            "context_text": context_text,
            "techniques_used": ["Standard Retrieval (Fallback)"]
        }


def ask_ollama(question, context, techniques_used=None):
    """Generates an answer using the local Ollama model."""
    # Create threat hunting specific prompt with technique information
    template = """You are an advanced threat hunting assistant that answers questions based only on the provided context.

    Context:
    {context}

    Question: {question}

    {techniques_info}

    Provide a detailed and accurate answer based solely on the information in the context.
    If the context doesn't contain enough information to answer the question properly, 
    state that you don't have sufficient information rather than making up an answer.

    For threat hunting questions, highlight any potential:
    - Indicators of Compromise (IOCs)
    - Tactics, Techniques, and Procedures (TTPs)
    - Affected systems or software
    - Mitigation strategies"""

    techniques_info = ""
    if techniques_used:
        techniques_info = f"\n\nRetrieval Techniques Used: {', '.join(techniques_used)}"

    prompt = ChatPromptTemplate.from_template(template.format(context=context, question=question, techniques_info=techniques_info))
    model = OllamaLLM(model="deepseek-r1:7b", temperature=0)  # Use deepseek for better reasoning
    answer = model.invoke(prompt)
    return answer


# Placeholder for processing status
processing_status = {
    'in_progress': False,
    'complete': False,
    'progress': 0,
    'message': '',
    'error': ''
}

# Placeholder for vector store (will be initialized later)
vectorstore = None
has_documents = False


def process_documents_sync(documents_to_process, is_background=False):
    """Process documents synchronously"""
    global vectorstore, has_documents, processing_status, PROCESSED_FILES
    if not documents_to_process:
        processing_status['error'] = 'No documents to process'
        return

    # Load existing cache
    try:
        PROCESSED_FILES = load_processed_files()
    except Exception as e:
        logging.warning(f"Could not load processed files cache: {e}")
        PROCESSED_FILES = set()

    # Filter out already processed files
    documents_to_process = [doc for doc in documents_to_process if doc not in PROCESSED_FILES]

    if not documents_to_process:
        processing_status['message'] = 'All documents already processed'
        processing_status['progress'] = 100
        processing_status['complete'] = True
        processing_status['in_progress'] = False
        return

    # Create a set to track newly processed files
    newly_processed = set()

    try:
        # Start processing
        processing_status['in_progress'] = True
        processing_status['message'] = 'Processing documents...'
        processing_status['progress'] = 0

        # Process each document
        texts = load_pdfs(documents_to_process[0]) # Assuming documents_to_process is a directory path
        chunks = chunk_texts(texts)
        vectorstore = build_vectorstore(chunks)
        has_documents = True

        # Update cache with newly processed files
        PROCESSED_FILES.update(newly_processed)
        try:
            save_processed_files()
            logging.info(f"Updated cache with {len(newly_processed)} new files")
        except Exception as e:
            logging.error(f"Failed to save processed files cache: {e}")

        # Mark as complete
        processing_status['progress'] = 100
        processing_status['complete'] = True
        processing_status['in_progress'] = False
        
        if not is_background:
            flash('Documents processed successfully', 'success')
            return redirect(url_for('index'))
            
    except Exception as e:
        processing_status['error'] = f"Error processing documents: {str(e)}"
        processing_status['complete'] = True
        processing_status['in_progress'] = False
        
        if not is_background:
            flash(f'Error processing documents: {str(e)}', 'danger') 
            return redirect(url_for('index'))