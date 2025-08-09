import streamlit as st
import sys
from pathlib import Path
import time
import shutil
from typing import Dict, Any
import logging
import os

# Configure logging to suppress verbose output
logging.basicConfig(level=logging.WARNING)

# Check Python version
python_version = sys.version_info
if python_version >= (3, 12):
    st.warning(f"⚠️ Python {python_version.major}.{python_version.minor} detected. This app works best with Python 3.11. Some dependencies may have compatibility issues.")

# Try to import the document journal with better error handling
try:
    from document_journal import DocumentJournalIndexer, LLAMAINDEX_AVAILABLE
    if not LLAMAINDEX_AVAILABLE:
        st.error("❌ **LlamaIndex dependencies are not available.**\n\nPlease install the required packages with:\n```bash\npip install llama-index llama-index-vector-stores-chroma llama-index-embeddings-huggingface llama-index-readers-file chromadb sentence-transformers\n```")
        st.stop()
except ImportError as e:
    st.error(f"❌ **Import Error:** {str(e)}\n\nPlease ensure all required dependencies are installed:\n```bash\npip install llama-index llama-index-vector-stores-chroma llama-index-embeddings-huggingface llama-index-readers-file chromadb sentence-transformers streamlit\n```")
    st.stop()
except Exception as e:
    st.error(f"❌ **Initialization Error:** {str(e)}\n\nThere was an issue initializing the document indexer. This might be due to:\n- Python version compatibility (recommend Python 3.11)\n- Missing system dependencies\n- ChromaDB compatibility issues")
    st.info("💡 **Troubleshooting:**\n1. Try downgrading to Python 3.11\n2. Reinstall dependencies in a fresh virtual environment\n3. Check system requirements for ChromaDB")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="AI Document Journal",
    page_icon="📓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .query-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    
    .answer-box {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    
    .source-snippet {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .metadata-tag {
        display: inline-block;
        background-color: #6c757d;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        margin: 0.2rem;
    }
    
    .score-badge {
        background-color: #17a2b8;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    .stats-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
    
    .upload-box {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    
    .search-options {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
    }
    
    .option-hint {
        color: #6c757d;
        font-size: 0.85rem;
        font-style: italic;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_indexer():
    """Initialize the Document Journal Indexer with caching."""
    try:
        indexer = DocumentJournalIndexer(
            documents_dir="documents",
            chroma_db_dir="storage/chroma_db",
            collection_name="document_journal",
            openai_api_key=None  # Using local embeddings
        )
        return indexer
    except Exception as e:
        st.error(f"❌ Failed to initialize indexer: {str(e)}")
        return None

@st.cache_resource
def load_index_with_cache(_indexer):
    """Load the document index with caching."""
    try:
        if not _indexer.index_exists():
            return None, "No documents found in the knowledge base."
        
        _indexer.build_or_load_index()
        return _indexer, "Index loaded successfully!"
    except Exception as e:
        return None, f"Failed to load index: {str(e)}"

def save_uploaded_file(uploaded_file):
    """Save uploaded file to documents directory."""
    try:
        documents_dir = Path("documents")
        documents_dir.mkdir(exist_ok=True)
        
        file_path = documents_dir / uploaded_file.name
        
        # Check if file already exists
        if file_path.exists():
            return False, f"File '{uploaded_file.name}' already exists."
        
        # Save the file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return True, f"File '{uploaded_file.name}' uploaded successfully!"
        
    except Exception as e:
        return False, f"Error uploading file: {str(e)}"

def format_file_size(size_bytes):
    """Convert file size to human readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes/(1024**2):.1f} MB"
    else:
        return f"{size_bytes/(1024**3):.1f} GB"

def delete_file(filename):
    """Delete a file from the documents directory and trigger index rebuild."""
    try:
        file_path = Path("documents") / filename
        if file_path.exists():
            file_path.unlink()
            return True, f"File '{filename}' deleted successfully! Please rebuild the index to update search results."
        else:
            return False, f"File '{filename}' not found."
    except Exception as e:
        return False, f"Error deleting file: {str(e)}"

def clear_all_documents():
    """Delete all documents from the documents directory."""
    try:
        documents_dir = Path("documents")
        deleted_count = 0
        
        if documents_dir.exists():
            for pdf_file in documents_dir.glob("*.pdf"):
                pdf_file.unlink()
                deleted_count += 1
        
        return True, f"Deleted {deleted_count} documents successfully!"
        
    except Exception as e:
        return False, f"Error clearing documents: {str(e)}"

def display_document_stats(stats):
    """Display document statistics in a formatted way."""
    st.markdown("### 📊 Knowledge Base Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Files", stats['total_files'])
    
    with col2:
        st.metric("Total Size", f"{stats['total_size_mb']:.2f} MB")
    
    with col3:
        st.metric("Indexed Documents", stats['indexed_documents'])
    
    # Document types
    if stats.get('document_types'):
        st.markdown("#### 📁 Document Types")
        for doc_type, count in stats['document_types'].items():
            st.write(f"• **{doc_type}**: {count} files")
    
    # File management section
    if stats.get('files'):
        with st.expander("📄 Manage Files"):
            # Clear all button
            if st.button("🗑️ Delete All Documents", type="secondary", help="Delete all PDF files and clear the index"):
                if st.session_state.get('confirm_delete_all', False):
                    success, message = clear_all_documents()
                    if success:
                        # Also reset the index
                        try:
                            indexer = initialize_indexer()
                            if indexer:
                                indexer.reset_collection()
                            st.cache_resource.clear()
                            st.success(message)
                            st.session_state['confirm_delete_all'] = False
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error clearing index: {str(e)}")
                    else:
                        st.error(message)
                else:
                    st.session_state['confirm_delete_all'] = True
                    st.warning("⚠️ Click again to confirm deletion of all documents!")
            
            st.markdown("---")
            st.markdown("**Individual File Management:**")
            
            # Individual file deletion
            for file_info in stats['files']:
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                with col1:
                    st.write(f"📄 {file_info['name']}")
                with col2:
                    st.write(f"{file_info['size_mb']:.2f} MB")
                with col3:
                    st.write(f"{file_info['type']}")
                with col4:
                    if st.button("🗑️", key=f"delete_{file_info['name']}", help=f"Delete {file_info['name']}"):
                        success, message = delete_file(file_info['name'])
                        if success:
                            st.success(message)
                            st.warning("⚠️ Index still contains old data. Click 'Rebuild Index' to update search results.")
                            st.rerun()
                        else:
                            st.error(message)

def display_source_snippet(source: Dict[str, Any], index: int):
    """Display a source snippet with metadata."""
    with st.expander(f"📄 Source {index + 1}: {source.get('metadata', {}).get('filename', 'Unknown')}"):
        # Content preview
        st.markdown("**Content Preview:**")
        st.markdown(f"*{source.get('summary', 'No content available.')}*")
        
        # Metadata
        metadata = source.get('metadata', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'document_type' in metadata:
                st.markdown(f'<span class="metadata-tag">📋 {metadata["document_type"]}</span>', unsafe_allow_html=True)
            if 'file_size' in metadata:
                file_size = format_file_size(metadata['file_size'])
                st.markdown(f'<span class="metadata-tag">💾 {file_size}</span>', unsafe_allow_html=True)
        
        with col2:
            if source.get('score') is not None:
                score_percent = round(source['score'] * 100, 1)
                st.markdown(f'<span class="score-badge">⭐ {score_percent}% relevance</span>', unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">📓 AI Document Journal</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Your intelligent document search and analysis assistant</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # System info
    python_version = sys.version_info
    if python_version >= (3, 12):
        st.info(f"ℹ️ Running on Python {python_version.major}.{python_version.minor}. For best compatibility, consider using Python 3.11.")
    
    # Initialize the indexer
    with st.spinner("🔧 Initializing AI Document Journal..."):
        indexer = initialize_indexer()
    
    if indexer is None:
        st.stop()
    
    # Sidebar for file management and stats
    with st.sidebar:
        st.header("📁 Document Management")
        
        # File upload section
        st.subheader("📥 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload PDF documents to add to your knowledge base"
        )
        
        if uploaded_files:
            st.write(f"Selected {len(uploaded_files)} file(s)")
            
            if st.button("💾 Save Uploaded Files", type="primary"):
                success_count = 0
                errors = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Uploading {uploaded_file.name}...")
                    success, message = save_uploaded_file(uploaded_file)
                    
                    if success:
                        success_count += 1
                    else:
                        errors.append(message)
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.empty()
                progress_bar.empty()
                
                if success_count > 0:
                    st.success(f"✅ Successfully uploaded {success_count} files!")
                    st.rerun()  # Refresh to show new files
                
                if errors:
                    for error in errors:
                        st.warning(f"⚠️ {error}")
        
        st.markdown("---")
        
        # Index management
        st.subheader("🔧 Index Management")
        
        if st.button("🔄 Rebuild Index", help="Rebuild the search index from current documents"):
            with st.spinner("Rebuilding index from current documents..."):
                try:
                    # Force rebuild to sync with current files
                    indexer.build_or_load_index(force_rebuild=True)
                    # Clear the cache to reload with new index
                    st.cache_resource.clear()
                    st.success("✅ Index rebuilt successfully! Now synchronized with current documents.")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error rebuilding index: {str(e)}")
        
        # Show index sync status
        stats = indexer.get_document_stats()
        if stats['total_files'] != stats['indexed_documents']:
            st.warning(f"⚠️ Index out of sync! Files: {stats['total_files']}, Indexed: {stats['indexed_documents']}")
            st.info("💡 Click 'Rebuild Index' to synchronize")
        
        st.markdown("---")
        
        # Document statistics
        display_document_stats(stats)
    
    # Main content area
    # Load the index
    with st.spinner("📚 Loading document knowledge base..."):
        loaded_indexer, load_message = load_index_with_cache(indexer)
    
    if loaded_indexer:
        st.success(f"✅ {load_message}")
        
        # Query interface
        st.subheader("💬 Ask Your Documents")
        
        # Example questions
        with st.expander("💡 Example Questions"):
            example_questions = [
                "What are the main topics covered in the documents?",
                "Summarize the key findings from the reports",
                "What information is available about [specific topic]?",
                "Find references to [keyword or phrase]",
                "What are the recommendations mentioned?",
                "Show me financial data or numbers",
                "What dates or deadlines are mentioned?",
                "Who are the key people or organizations mentioned?"
            ]
            
            # Display in two columns
            col1, col2 = st.columns(2)
            for i, question in enumerate(example_questions):
                if i % 2 == 0:
                    col1.write(f"• {question}")
                else:
                    col2.write(f"• {question}")
        
        # Query input
        query = st.text_input(
            "Enter your question:",
            placeholder="e.g., What are the main topics in these documents?",
            help="Ask any question about your document collection"
        )
        
        # Enhanced Search Options
        st.markdown("### ⚙️ Search Options")
        
        with st.container():
            col1, col2 = st.columns([2, 1])
            
            with col1:
                num_results = st.slider(
                    "🔢 Number of sources to show:",
                    min_value=1,
                    max_value=10,
                    value=3,  # Default to 3 results
                    step=1,
                    help="Select how many relevant document sources to display."
                )
            
            with col2:
                show_metadata = st.checkbox(
                    "📋 Show detailed metadata",
                    value=True,
                    help="Include file information and relevance scores"
                )
            
            # Display current configuration
            if num_results == 1:
                st.markdown('<div class="option-hint">💡 Showing the most relevant source for focused results</div>', 
                          unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="option-hint">📚 Showing top {num_results} sources for comprehensive search</div>', 
                          unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Search button
        if st.button("🔍 Search Documents", type="primary", use_container_width=True):
            if query.strip():
                with st.spinner(f"🔍 Searching your documents for top {num_results} result{'s' if num_results > 1 else ''}..."):
                    try:
                        # Execute the query with dynamic top_k
                        result = loaded_indexer.query(
                            query_text=query,
                            top_k=num_results
                        )
                        
                        # Display the answer
                        st.markdown("## 💡 Answer")
                        answer_text = result.get('answer', 'No answer generated')
                        
                        st.markdown(f"""
                        <div class="answer-box">
                            <h3>📋 Response</h3>
                            <p>{answer_text}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display sources
                        if result.get('sources'):
                            source_count = len(result['sources'])
                            if source_count == 1:
                                st.markdown("## 📄 Source Document")
                                st.write("Found the most relevant source:")
                            else:
                                st.markdown("## 📚 Source Documents")
                                st.write(f"Found {source_count} relevant sources:")
                            
                            for i, source in enumerate(result['sources']):
                                if show_metadata:
                                    display_source_snippet(source, i)
                                else:
                                    # Simple display without metadata
                                    filename = source.get('metadata', {}).get('filename', 'Unknown')
                                    summary = source.get('summary', 'No content available')
                                    st.markdown(f"**📄 {filename}**")
                                    st.write(summary)
                                    if i < len(result['sources']) - 1:
                                        st.markdown("---")
                        else:
                            st.info("ℹ️ No specific source documents were identified for this answer.")
                        
                        # Show search summary
                        st.markdown("---")
                        search_col1, search_col2, search_col3 = st.columns(3)
                        
                        with search_col1:
                            st.metric("🎯 Query Status", "✅ Complete")
                        with search_col2:
                            st.metric("📊 Sources Found", len(result.get('sources', [])))
                        with search_col3:
                            st.metric("⚙️ Search Depth", f"Top {num_results}")
                            
                    except Exception as e:
                        st.error(f"❌ Error during search: {str(e)}")
                        st.write("Please try rephrasing your question or check if the knowledge base is properly loaded.")
                        st.info("💡 **Troubleshooting:**\n- Try rebuilding the index\n- Check if documents are properly uploaded\n- Ensure Python version compatibility")
            else:
                st.warning("⚠️ Please enter a question to search.")
        
        # Quick stats display in main area
        if stats['total_files'] > 0:
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📄 Documents", stats['total_files'])
            with col2:
                st.metric("💾 Total Size", f"{stats['total_size_mb']:.1f} MB")
            with col3:
                st.metric("🗄️ Indexed", stats['indexed_documents'])
            with col4:
                types_count = len(stats.get('document_types', {}))
                st.metric("📁 Types", types_count)
    
    else:
        # No documents available
        st.warning("📁 No Document Knowledge Base Available")
        
        st.markdown(f"""
        **Status:** {load_message}
        
        **Get Started:**
        1. **Upload Documents**: Use the sidebar to upload PDF files
        2. **Build Index**: Click "Rebuild Index" after uploading
        3. **Start Searching**: Ask questions about your documents
        
        **System Requirements:**
        - Python 3.11 (recommended) or 3.10+
        - PDF documents in the `documents/` directory
        - ChromaDB storage for vector indexing
        - HuggingFace embeddings (local, no API key needed)
        """)
        
        # Show upload area in main content if no documents
        st.markdown("### 📥 Quick Upload")
        st.markdown("""
        <div class="upload-box">
            <h4>👆 Use the sidebar to upload PDF documents</h4>
            <p>Supported formats: PDF files</p>
            <p>The system will automatically process and index your documents for intelligent search</p>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #6c757d; font-size: 0.9rem;">
        <p>🤖 Powered by LlamaIndex, ChromaDB & HuggingFace | Python {python_version.major}.{python_version.minor}</p>
        <p>💡 Upload PDFs • 🔍 Ask Questions • 📊 Get Insights</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()