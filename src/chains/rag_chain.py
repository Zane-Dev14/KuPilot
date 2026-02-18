import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from src.config import get_settings
from src.retrieval.retriever import EnhancedRetriever
from src.chains.model_selector import ModelSelector
from src.models.schemas import FailureDiagnosis

logger = logging.getLogger(__name__)

class RAGChain:
    """RAG chain for Kubernetes failure diagnosis."""
    
    def __init__(self):
        self.settings = get_settings()
        self.retriever = EnhancedRetriever()
        self.model_selector = ModelSelector()
    
    def _format_documents(self, docs: list[Document]) -> str:
        """Format retrieved documents for context."""
        if not docs:
            return "No relevant documents found."
        
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            kind = doc.metadata.get("kind", "unknown")
            formatted.append(f"[Document {i}] (Kind: {kind}, Source: {source})")
            formatted.append(doc.page_content)
            formatted.append("")
        
        return "\n".join(formatted)
    
    def _get_llm(self, model: str) -> ChatOllama:
        """Get a ChatOllama instance."""
        return ChatOllama(
            model=model,
            temperature=0,
            base_url=self.settings.ollama_base_url,
        )
    
    def diagnose(self, query: str, force_model: str | None = None) -> FailureDiagnosis:
        """
        Diagnose a Kubernetes failure.
        
        Args:
            query: User question/description
            force_model: Override model selection
        
        Returns:
            FailureDiagnosis with explanation, fix, confidence, sources
        """
        logger.info(f"Starting diagnosis for query: {query[:100]}...")
        
        # Retrieve relevant documents
        retrieval_chain = self.retriever.get_chain()
        documents = retrieval_chain.invoke(query)
        
        logger.info(f"Retrieved {len(documents)} documents")
        
        # Select model
        selected_model = self.model_selector.select_model(query, force_model)
        
        # Format context
        context = self._format_documents(documents)
        
        # Build prompt
        prompt = ChatPromptTemplate.from_template(
        """
            You are a Kubernetes failure diagnosis expert. Based on the provided context and failure descriptions, provide:
            1. Root cause analysis
            2. Clear explanation of the failure
            3. Recommended fix(es)
            4. Confidence score (0.0-1.0)

            Be concise, technical, and actionable.

            Context:
            {context}

            User Query:
            {query}

            Provide your response in JSON format with keys: root_cause, explanation, recommended_fix, confidence, reasoning.
        """
        )
        
        # Create LLM with structured output
        llm = self._get_llm(selected_model)
        structured_llm = llm.with_structured_output(FailureDiagnosis)
        
        # Build chain
        chain = prompt | structured_llm
        
        # Invoke
        try:
            result = chain.invoke({
                "context": context,
                "query": query,
            })
            
            # Build source list
            source_list = [
                doc.metadata.get("source", "unknown") for doc in documents
            ]
            
            # Determine thinking chain
            thinking = (
                f"[{selected_model} reasoning enabled]"
                if selected_model == self.settings.complex_model
                else None
            )
            
            # Handle result from with_structured_output (can be dict or BaseModel)
            if isinstance(result, dict):
                data = result
            elif isinstance(result, FailureDiagnosis):
                data = result.model_dump()
            else:
                data = result.model_dump()
            
            diagnosis = FailureDiagnosis(
                **data,
                reasoning_model_used=selected_model,
                thinking_chain=thinking,
                sources=source_list,
            )
            
            logger.info(f"Diagnosis complete. Model: {selected_model}")
            return diagnosis
        
        except Exception as e:
            logger.error(f"Diagnosis failed: {e}")
            raise