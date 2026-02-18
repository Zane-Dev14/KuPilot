import logging
import re

from src.models.schemas import QueryMetadata

logger = logging.getLogger(__name__)


class QueryAnalyzer:
    """Analyze queries to extract K8s metadata and decompose multi-part questions."""
    
    # Kubernetes error types
    ERROR_PATTERNS = {
        r"(?:crash|crashing|crashed)": "crashed",
        r"(?:oom|out of memory|memory killed)": "oom_killed",
        r"(?:image pull|registry)": "image_pull_error",
        r"(?:scheduling|pending|not scheduled)": "scheduling_failed",
        r"(?:node(?:not )?ready|node unhealthy)": "node_issue",
        r"(?:evicted|eviction)": "evicted",
        r"(?:unknown|host unreachable)": "host_unreachable",
    }
    
    # K8s object patterns
    NAMESPACE_PATTERN = r"(?:namespace|ns|in namespace)\s+([a-z0-9\-]+)"
    POD_PATTERN = r"(?:pod|pods?)\s+([a-z0-9\-]+)"
    CONTAINER_PATTERN = r"(?:container|containers?)\s+([a-z0-9\-]+)"
    NODE_PATTERN = r"(?:node|nodes)\s+([a-z0-9\-\.]+)"
    LABEL_PATTERN = r"label[s]?\s+([a-z0-9\-=,\s]+)"
    
    # Query decomposition
    DECOMPOSITION_SEPARATORS = [";", " and also ", " additionally "]
    MULTI_PART_KEYWORDS = ["and", "also", "plus", "besides"]
    
    def extract_k8s_metadata(self, query: str) -> QueryMetadata:
        """Extract K8s object references from human query."""
        query_lower = query.lower()
        metadata = QueryMetadata()
        
        # Extract namespace
        ns_match = re.search(self.NAMESPACE_PATTERN, query_lower)
        if ns_match:
            metadata.namespace = ns_match.group(1)
            logger.debug(f"Extracted namespace: {metadata.namespace}")
        
        # Extract pod
        pod_match = re.search(self.POD_PATTERN, query_lower)
        if pod_match:
            metadata.pod = pod_match.group(1)
            logger.debug(f"Extracted pod: {metadata.pod}")
        
        # Extract container
        container_match = re.search(self.CONTAINER_PATTERN, query_lower)
        if container_match:
            metadata.container = container_match.group(1)
            logger.debug(f"Extracted container: {metadata.container}")
        
        # Extract node
        node_match = re.search(self.NODE_PATTERN, query_lower)
        if node_match:
            metadata.node = node_match.group(1)
            logger.debug(f"Extracted node: {metadata.node}")
        
        # Extract error type
        for pattern, error_type in self.ERROR_PATTERNS.items():
            if re.search(pattern, query_lower):
                metadata.error_type = error_type
                logger.debug(f"Extracted error type: {metadata.error_type}")
                break
        
        # Extract labels (simple parsing)
        label_match = re.search(self.LABEL_PATTERN, query_lower)
        if label_match:
            label_str = label_match.group(1)
            for pair in label_str.split(","):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    metadata.labels_dict[k.strip()] = v.strip()
            logger.debug(f"Extracted labels: {metadata.labels_dict}")
        
        return metadata
    
    def decompose_query(self, query: str) -> list[str]:
        """Split multi-part query into sub-queries."""
        sub_queries = [query]
        
        # Check for explicit separators
        for sep in self.DECOMPOSITION_SEPARATORS:
            if sep in query:
                parts = query.split(sep)
                sub_queries = [p.strip() for p in parts if p.strip()]
                logger.debug(f"Decomposed query into {len(sub_queries)} parts")
                return sub_queries
        
        # Check for multi-part keywords
        keyword_count = sum(1 for kw in self.MULTI_PART_KEYWORDS if f" {kw} " in f" {query.lower()} ")
        if keyword_count > 1:
            logger.debug(f"Query has {keyword_count} potential parts (not decomposing)")
        
        return sub_queries
    
    def analyze(self, query: str) -> tuple[QueryMetadata, list[str]]:
        """Unified entry point: extract metadata and decompose."""
        metadata = self.extract_k8s_metadata(query)
        sub_queries = self.decompose_query(query)
        return metadata, sub_queries