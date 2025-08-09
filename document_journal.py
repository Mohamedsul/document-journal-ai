# %% [markdown]
# ## Building a Searchable Document Journal using LlamaIndex and ChromaDB
# A modular, production-ready Python class to parse, index, and query document collections (PDFs) 
# using LlamaIndex with ChromaDB vector store and Hugging Face embeddings.
# **Updated for clean UI display in Streamlit and other user-facing applications.**

# %% [markdown]
# ### Notebook Highlights
# - Modular, class-based design for scalability
# - ChromaDB integration for persistent vector storage
# - Automatic PDF ingestion and metadata tagging
# - Local Hugging Face embeddings for semantic search
# - **Minimal, relevant source information**
# - RAG-style question answering capabilities
# - General-purpose document indexing for any domain

# %%
# 1. Import and Setup the Environment

# Standard library imports
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
from collections import defaultdict

# LlamaIndex Core Imports
try:
    from llama_index.core import (
        VectorStoreIndex,
        Document,
        Settings,
        StorageContext,
    )
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.readers.file import PDFReader
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.openai import OpenAI
    
    # ChromaDB Integration - with error handling
    from llama_index.vector_stores.chroma import ChromaVectorStore
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    
    LLAMAINDEX_AVAILABLE = True
except ImportError as e:
    logging.warning(f"LlamaIndex imports failed: {e}")
    LLAMAINDEX_AVAILABLE = False

# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# %%
class DocumentJournalIndexer:
    """
    Document Journal Indexer with ChromaDB vector store for creating a searchable knowledge base
    from any collection of PDF documents including reports, manuals, invoices, research papers, and more.
    
    Optimized for clean UI display with user-friendly query results.
    """
    
    def __init__(self, 
                 documents_dir: str = "documents",
                 chroma_db_dir: str = "storage/chroma_db",
                 collection_name: str = "document_journal",
                 openai_api_key: Optional[str] = None,
                 custom_document_types: Optional[Dict[str, str]] = None):
        """
        Initialize the Document Journal Indexer with ChromaDB integration.
        
        Args:
            documents_dir: Directory containing PDF documents
            chroma_db_dir: Directory to store ChromaDB data
            collection_name: Name of the ChromaDB collection
            openai_api_key: OpenAI API key for LLM (optional, for query answering)
            custom_document_types: Custom mapping of keywords to document types
        """
        if not LLAMAINDEX_AVAILABLE:
            raise RuntimeError("LlamaIndex dependencies are not available. Please install required packages.")
            
        self.documents_dir = Path(documents_dir)
        self.chroma_db_dir = Path(chroma_db_dir)
        self.collection_name = collection_name
        self.openai_api_key = openai_api_key
        self.custom_document_types = custom_document_types or {}
        
        # Create directories if they don't exist
        self.documents_dir.mkdir(exist_ok=True)
        self.chroma_db_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client and collection
        self._setup_chromadb()
        
        # Setup LlamaIndex settings
        self._setup_settings()
        
        # Initialize PDF reader
        self.pdf_reader = PDFReader()
        
        # Initialize index as None
        self.index = None
    
    def _setup_chromadb(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Create ChromaDB client with persistent storage
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.chroma_db_dir),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            try:
                self.chroma_collection = self.chroma_client.get_collection(
                    name=self.collection_name
                )
                logger.info(f"Found existing ChromaDB collection: {self.collection_name}")
            except Exception:
                self.chroma_collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Document Journal Knowledge Base"}
                )
                logger.info(f"Created new ChromaDB collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Error setting up ChromaDB: {e}")
            raise
    
    def _setup_settings(self):
        """Configure LlamaIndex settings with ChromaDB vector store."""
        try:
            # Use OpenAI embeddings if API key provided, otherwise use local embeddings
            if self.openai_api_key:
                Settings.llm = OpenAI(
                    api_key=self.openai_api_key,
                    model="gpt-3.5-turbo",
                    temperature=0.1
                )
                logger.info("Using OpenAI embeddings and LLM")
            else:
                # Disable the global LLM to prevent fallback errors
                Settings.llm = None
                # Use local HuggingFace embeddings (free alternative)
                Settings.embed_model = HuggingFaceEmbedding(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                logger.info("Using local HuggingFace embeddings")
            
            # Configure text splitter for optimal chunking
            Settings.node_parser = SentenceSplitter(
                chunk_size=512,
                chunk_overlap=50,
                separator="\n"
            )
            
        except Exception as e:
            logger.error(f"Error setting up LlamaIndex settings: {e}")
            raise
    
    def load_documents(self) -> List[Document]:
        """
        Load and parse PDF documents from the documents directory.
        
        Returns:
            List of Document objects with metadata
        """
        documents = []
        pdf_files = list(self.documents_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.documents_dir}")
            return documents
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            try:
                logger.info(f"Processing {pdf_file.name}")
                
                # Read the PDF content
                pdf_documents = self.pdf_reader.load_data(file=pdf_file)
                
                # Add comprehensive metadata to each document
                for doc in pdf_documents:
                    doc_type = self._extract_document_type(pdf_file.name)
                    
                    doc.metadata.update({
                        "filename": pdf_file.name,
                        "document_type": doc_type,
                        "source": str(pdf_file),
                        "file_size": pdf_file.stat().st_size,
                        "doc_id": str(uuid.uuid4()),  # Unique identifier
                        "indexed_at": datetime.now().isoformat()
                    })
                    
                    documents.append(doc)
                
                logger.info(f"Successfully loaded {len(pdf_documents)} pages from {pdf_file.name}")
                
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {e}")
                continue
        
        logger.info(f"Total documents loaded: {len(documents)}")
        return documents
    
    def _extract_document_type(self, filename: str) -> str:
        """
        Extract document type from filename for better categorization.
        
        Args:
            filename: Name of the PDF file
            
        Returns:
            Document type category
        """
        filename_lower = filename.lower()
        
        # Default generic document type mapping
        default_type_mapping = {
            "report": "Report",
            "manual": "Manual",
            "guide": "Guide",
            "invoice": "Invoice",
            "receipt": "Receipt",
            "contract": "Contract",
            "agreement": "Agreement",
            "meeting": "Meeting Notes",
            "minutes": "Meeting Notes",
            "proposal": "Proposal",
            "presentation": "Presentation",
            "research": "Research Paper",
            "study": "Research Paper",
            "analysis": "Analysis",
            "summary": "Summary",
            "memo": "Memo",
            "letter": "Letter",
            "specification": "Specification",
            "spec": "Specification",
            "policy": "Policy",
            "procedure": "Procedure",
            "instruction": "Instructions",
            "tutorial": "Tutorial",
            "review": "Review",
            "plan": "Plan",
            "budget": "Budget",
            "financial": "Financial Document",
            "legal": "Legal Document",
            "technical": "Technical Document"
        }
        
        # Merge with custom document types if provided
        type_mapping = {**default_type_mapping, **self.custom_document_types}
        
        for keyword, doc_type in type_mapping.items():
            if keyword in filename_lower:
                return doc_type
        
        return "General Document"
    
    def create_index(self, documents: List[Document] = None) -> VectorStoreIndex:
        """
        Create a vector index from documents using ChromaDB.
        
        Args:
            documents: List of documents to index. If None, load from directory
            
        Returns:
            VectorStoreIndex object
        """
        if documents is None:
            documents = self.load_documents()
        
        if not documents:
            raise ValueError("No documents to index")
        
        logger.info("Creating vector index with ChromaDB...")
        
        try:
            # Create ChromaDB vector store
            vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
            
            # Create storage context with ChromaDB
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Create the index
            self.index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                show_progress=True
            )
            
            logger.info("Vector index created successfully with ChromaDB")
            return self.index
            
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise
    
    def load_index(self) -> VectorStoreIndex:
        """
        Load existing index from ChromaDB.
        
        Returns:
            VectorStoreIndex object
        """
        try:
            logger.info("Loading index from ChromaDB...")
            
            # Create vector store from existing collection
            vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
            
            # Create storage context
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Load the index
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=storage_context
            )
            
            logger.info("Index loaded successfully from ChromaDB")
            return self.index
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            raise
    
    def index_exists(self) -> bool:
        """
        Check if the ChromaDB collection contains any documents.
        
        Returns:
            True if index exists and has documents, False otherwise
        """
        try:
            count = self.chroma_collection.count()
            return count > 0
        except Exception as e:
            logger.error(f"Error checking index existence: {e}")
            return False
    
    def build_or_load_index(self, force_rebuild: bool = False) -> VectorStoreIndex:
        """
        Build a new index or load an existing one.
        
        Args:
            force_rebuild: If True, rebuild index even if it exists
            
        Returns:
            VectorStoreIndex object
        """
        if force_rebuild:
            logger.info("Force rebuilding index...")
            self.reset_collection()
            documents = self.load_documents()
            self.index = self.create_index(documents)
        elif self.index_exists():
            logger.info("Loading existing index...")
            self.index = self.load_index()
        else:
            logger.info("Building new index...")
            documents = self.load_documents()
            self.index = self.create_index(documents)
        
        return self.index
    
    def reset_collection(self):
        """Reset the ChromaDB collection (delete all documents)."""
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
            self.chroma_collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Document Journal Knowledge Base"}
            )
            logger.info("ChromaDB collection reset successfully")
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            raise
    
    def get_document_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the indexed documents.
        
        Returns:
            Dictionary with document statistics
        """
        pdf_files = list(self.documents_dir.glob("*.pdf"))
        
        stats = {
            "total_files": len(pdf_files),
            "files": [],
            "total_size_mb": 0,
            "indexed_documents": self.chroma_collection.count() if hasattr(self, 'chroma_collection') else 0,
            "document_types": {}
        }
        
        for pdf_file in pdf_files:
            file_size = pdf_file.stat().st_size
            doc_type = self._extract_document_type(pdf_file.name)
            
            file_info = {
                "name": pdf_file.name,
                "size_mb": round(file_size / (1024 * 1024), 2),
                "type": doc_type
            }
            
            stats["files"].append(file_info)
            stats["total_size_mb"] += file_size / (1024 * 1024)
            
            # Count document types
            if doc_type in stats["document_types"]:
                stats["document_types"][doc_type] += 1
            else:
                stats["document_types"][doc_type] = 1
        
        stats["total_size_mb"] = round(stats["total_size_mb"], 2)
        return stats
    
    def _format_filename(self, filename: str) -> str:
        """
        Format filename for display (remove .pdf extension and clean up).
        
        Args:
            filename: Original filename
            
        Returns:
            Cleaned filename for display
        """
        # Remove .pdf extension
        name = filename.replace('.pdf', '').replace('.PDF', '')
        
        # Replace underscores and hyphens with spaces
        name = name.replace('_', ' ').replace('-', ' ')
        
        # Capitalize first letter of each word
        name = ' '.join(word.capitalize() for word in name.split())
        
        return name
    
    def _clean_and_combine_text(self, text_chunks: List[str]) -> str:
        """
        Clean and intelligently combine multiple text chunks from the same document.
        
        Args:
            text_chunks: List of text chunks to combine
            
        Returns:
            Combined and cleaned text
        """
        if not text_chunks:
            return "No content available"
        
        # Remove duplicates while preserving order
        seen = set()
        unique_chunks = []
        for chunk in text_chunks:
            chunk_cleaned = chunk.strip()
            if chunk_cleaned and chunk_cleaned not in seen:
                seen.add(chunk_cleaned)
                unique_chunks.append(chunk_cleaned)
        
        if not unique_chunks:
            return "No content available"
        
        # If only one chunk, return it cleaned
        if len(unique_chunks) == 1:
            return self._clean_text(unique_chunks[0])
        
        # For multiple chunks, create a summary with bullet points
        combined_text = []
        for i, chunk in enumerate(unique_chunks[:3], 1):  # Limit to 3 most relevant
            cleaned_chunk = self._clean_text(chunk, max_length=150)
            if cleaned_chunk:
                combined_text.append(f"• {cleaned_chunk}")
        
        return '\n'.join(combined_text)
    
    def _clean_text(self, text: str, max_length: int = 300) -> str:
        """
        Clean text by removing excessive whitespace and truncating if needed.
        
        Args:
            text: Text to clean
            max_length: Maximum length before truncation
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace and newlines
        text = ' '.join(text.split())
        
        # Truncate if too long
        if len(text) > max_length:
            text = text[:max_length].strip()
            # Try to end at a sentence or word boundary
            last_period = text.rfind('.')
            last_space = text.rfind(' ')
            
            if last_period > max_length - 50:
                text = text[:last_period + 1]
            elif last_space > max_length - 20:
                text = text[:last_space] + "..."
            else:
                text = text + "..."
        
        return text.strip()
    
    def query(self, query_text: str, top_k: int = 8) -> Dict[str, Any]:
        """
        Query the document journal knowledge base with professional, user-friendly results.
        
        Args:
            query_text: The question to ask
            top_k: Number of relevant documents to retrieve
            
        Returns:
            Dictionary containing clean, professional results with sources
        """
        if self.index is None:
            raise ValueError("Index not loaded. Call build_or_load_index() first.")
        
        try:
            # Create query engine
            query_engine = self.index.as_query_engine(
                similarity_top_k=top_k,
                response_mode="compact"
            )
            
            # Execute query
            response = query_engine.query(query_text)
            
            # Group results by document filename
            document_groups = defaultdict(list)
            sources = []
            
            if hasattr(response, 'source_nodes') and response.source_nodes:
                for node in response.source_nodes:
                    metadata = getattr(node.node, 'metadata', {})
                    filename = metadata.get('filename', 'Unknown Document')
                    doc_type = metadata.get('document_type', '')
                    text = getattr(node.node, 'text', '')
                    score = getattr(node, 'score', 0.0)
                    
                    if text.strip():  # Only include non-empty text
                        document_groups[filename].append({
                            'text': text,
                            'doc_type': doc_type,
                            'score': score
                        })
                        
                        # Also create sources list for compatibility
                        sources.append({
                            'summary': self._clean_text(text, max_length=200),
                            'metadata': metadata,
                            'score': score
                        })
            
            # Create clean document summaries
            document_summaries = []
            for filename, chunks in document_groups.items():
                if chunks:
                    # Extract text chunks
                    text_chunks = [chunk['text'] for chunk in chunks]
                    doc_type = chunks[0]['doc_type']  # Get document type from first chunk
                    avg_score = sum(chunk['score'] for chunk in chunks) / len(chunks)
                    
                    # Combine and clean the text
                    combined_text = self._clean_and_combine_text(text_chunks)
                    
                    document_summaries.append({
                        'filename': self._format_filename(filename),
                        'doc_type': doc_type,
                        'summary': combined_text,
                        'score': avg_score
                    })
            
            # Create the result
            result = {
                "has_results": len(document_summaries) > 0,
                "total_documents": len(document_summaries),
                "documents": document_summaries,
                "answer": str(response) if response else "No answer generated",
                "sources": sources  # Add sources for app compatibility
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error during query: {e}")
            return {
                "has_results": False,
                "total_documents": 0,
                "documents": [],
                "answer": "Unable to search documents. Please try again.",
                "sources": [],
                "error": "Unable to search documents. Please try again."
            }
    
    def format_results_for_display(self, result: Dict[str, Any], query_text: str = "") -> str:
        """
        Format query results for professional display in Streamlit or other UI.
        
        Args:
            result: Result dictionary from query method
            query_text: Original query text (optional)
            
        Returns:
            Professionally formatted markdown string
        """
        if not result.get("has_results", False):
            if result.get("error"):
                return f"❌ **Error:** {result['error']}"
            else:
                return "📭 **No relevant information found** in your documents for this query."
        
        # Build the formatted response
        output_lines = []
        
        # Add header
        if query_text:
            output_lines.append(f"## 📄 Document Search Results")
            output_lines.append(f"**Query:** {query_text}")
        else:
            output_lines.append(f"## 📄 Document Summary")
        
        output_lines.append("")  # Empty line
        
        # Add document count
        doc_count = result.get("total_documents", 0)
        if doc_count == 1:
            output_lines.append("Found relevant information in **1 document**:")
        else:
            output_lines.append(f"Found relevant information in **{doc_count} documents**:")
        
        output_lines.append("")  # Empty line
        
        # Add each document summary
        for i, doc in enumerate(result.get("documents", []), 1):
            filename = doc.get("filename", "Unknown Document")
            doc_type = doc.get("doc_type", "")
            summary = doc.get("summary", "No content available")
            
            # Document header with type if available
            if doc_type and doc_type != "General Document":
                output_lines.append(f"### 📋 {filename}")
                output_lines.append(f"*{doc_type}*")
            else:
                output_lines.append(f"### 📋 {filename}")
            
            output_lines.append("")  # Empty line
            
            # Add the summary content
            output_lines.append(summary)
            output_lines.append("")  # Empty line
            
            # Add separator between documents (except for the last one)
            if i < doc_count:
                output_lines.append("---")
                output_lines.append("")  # Empty line
        
        return "\n".join(output_lines)
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the ChromaDB collection.
        
        Returns:
            Dictionary with collection information
        """
        try:
            return {
                "collection_name": self.collection_name,
                "document_count": self.chroma_collection.count(),
                "collection_metadata": self.chroma_collection.metadata,
                "chroma_db_path": str(self.chroma_db_dir)
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}
    
    def add_custom_document_types(self, custom_types: Dict[str, str]):
        """
        Add or update custom document type mappings.
        
        Args:
            custom_types: Dictionary mapping keywords to document types
        """
        self.custom_document_types.update(custom_types)
        logger.info(f"Updated custom document types: {custom_types}")

# %%
# Example usage and testing
if __name__ == "__main__":
    # Initialize the Document Journal Indexer
    doc_journal = DocumentJournalIndexer(
        documents_dir="documents",
        chroma_db_dir="storage/chroma_db",
        collection_name="document_journal",
        openai_api_key=None,  # Add your OpenAI key here if you want LLM capabilities
        custom_document_types={
            "quarterly": "Quarterly Report",
            "annual": "Annual Report",
            "whitepaper": "White Paper"
        }
    )
    
    # Get document statistics
    stats = doc_journal.get_document_stats()
    print(f"\n📊 Document Statistics:")
    print(f"Total Files: {stats['total_files']}")
    print(f"Total Size: {stats['total_size_mb']:.2f} MB")
    print(f"Indexed Documents: {stats['indexed_documents']}")
    
    if stats['document_types']:
        print(f"\n📁 Document Types:")
        for doc_type, count in stats['document_types'].items():
            print(f"  • {doc_type}: {count} files")
    
    if stats['files']:
        print(f"\n📄 Files:")
        for file_info in stats['files']:
            print(f"  • {file_info['name']} ({file_info['type']}) - {file_info['size_mb']:.2f} MB")
    
    if stats['total_files'] == 0:
        print("📁 No documents found. Please upload PDFs to the documents folder.")
    else:
        # Build or load the index
        print(f"\n🔧 Building/Loading Index...")
        index = doc_journal.build_or_load_index()
        
        # Get collection info
        collection_info = doc_journal.get_collection_info()
        print(f"\n🗄️ ChromaDB Collection Info:")
        print(f"Collection: {collection_info.get('collection_name', 'N/A')}")
        print(f"Documents: {collection_info.get('document_count', 0)}")
        print(f"Storage Path: {collection_info.get('chroma_db_path', 'N/A')}")
        
        # Example Query with formatting
        print(f"\n🔍 Testing Query Results:")
        result = doc_journal.query("What information is available in the documents?", top_k=5)
        
        # Display formatted results
        formatted_output = doc_journal.format_results_for_display(result, "What information is available in the documents?")
        print("\n" + "="*60)
        print("OUTPUT Results:")
        print("="*60)
        print(formatted_output)