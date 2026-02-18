import logging
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.documents import Document
from src.config import get_settings
from src.vectorstore.milvus_client import MilvusVectorStore
from src.retrieval.query_analyzer import QueryAnalyzer
from src.retrieval.reranker import RerankerService

logger = logging.getLogger(__name__)

class EnhancedRetriever:
    """LCEL retrieval chain with query analysis and reranking."""
    
    def __init__(self):
        self.settings = get_settings()
        self.vectorstore = MilvusVectorStore()
        self.query_analyzer = QueryAnalyzer()
        self.reranker = RerankerService()
    
    def _build_retrieval_chain(self):
        """Build the LCEL retrieval chain."""
        
        def analyze_and_retrieve(query: str) -> list[Document]:
            """Analyze query, search, and rerank."""
            logger.info(f"Processing query: {query[:100]}...")
            
            # Analyze query
            metadata, sub_queries = self.query_analyzer.analyze(query)
            logger.debug(f"Extracted metadata: {metadata}")
            logger.debug(f"Sub-queries: {sub_queries}")
            
            # Retrieve for each sub-query
            all_documents = []
            for sub_query in sub_queries:
                logger.debug(f"Retrieving for sub-query: {sub_query}")
                
                # Vector search (retrieve more than top_k for reranking)
                retriever = self.vectorstore.as_retriever(k=self.settings.retrieval_rerank_k)
                docs = retriever.invoke(sub_query)
                
                # Apply metadata filters if applicable
                if metadata.namespace:
                    docs = [
                        doc for doc in docs
                        if doc.metadata.get("namespace") == metadata.namespace
                    ]
                
                if metadata.pod:
                    docs = [
                        doc for doc in docs
                        if doc.metadata.get("pod") == metadata.pod
                    ]
                
                all_documents.extend(docs)
            
            # Deduplicate
            seen_ids = set()
            unique_docs = []
            for doc in all_documents:
                doc_id = doc.metadata.get("id") or doc.page_content[:100]
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    unique_docs.append(doc)
            
            logger.debug(f"Retrieved {len(unique_docs)} unique documents before reranking")
            
            # Rerank
            reranked = self.reranker.rerank(
                unique_docs,
                query,
                top_k=self.settings.retrieval_top_k
            )
            
            logger.info(f"Retrieval complete: returned {len(reranked)} documents")
            return reranked
        
        return RunnableLambda(analyze_and_retrieve)
    
    def get_chain(self):
        """Get the retrieval chain."""
        return self._build_retrieval_chain()