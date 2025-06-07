from transformers import pipeline
from typing import Dict, Optional

class DocumentAnalyzer:
    """Class for analyzing legal documents using transformer models."""
    
    def __init__(self):
        """Initialize the document analyzer with required models."""
        # Initialize transformers pipelines
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=-1  # Use CPU, change to 0 for GPU
        )
        self.classifier = pipeline(
            "text-classification",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            device=-1
        )
    
    def _chunk_text(self, text: str, max_chunk_size: int = 1024) -> list:
        """Split text into chunks that can be processed by the models."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            if current_size + len(word) + 1 > max_chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_size = len(word)
            else:
                current_chunk.append(word)
                current_size += len(word) + 1
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks

    def summarize(self, text: str, max_length: int = 130, min_length: int = 30) -> str:
        """Generate a summary of the input text."""
        chunks = self._chunk_text(text)
        summaries = []
        
        for chunk in chunks:
            if len(chunk.split()) < min_length:
                continue
                
            summary = self.summarizer(
                chunk,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            summaries.append(summary[0]['summary_text'])
        
        return ' '.join(summaries)

    def classify(self, text: str) -> str:
        """Classify the type of legal document."""
        # For now, using sentiment classifier as placeholder
        # TODO: Replace with actual legal document classifier
        result = self.classifier(text[:512])[0]
        return result['label']

    def analyze(self, text: str) -> Dict[str, Optional[str]]:
        """Perform full analysis of the document."""
        try:
            summary = self.summarize(text)
            category = self.classify(text)
            
            return {
                "summary": summary,
                "category": category,
                "error": None
            }
        except Exception as e:
            return {
                "summary": None,
                "category": None,
                "error": str(e)
            }

    def extract_key_points(self, text: str) -> list:
        """Extract key points from the document."""
        # TODO: Implement key points extraction
        summary = self.summarize(text, max_length=150, min_length=40)
        # Simple approach: split summary into sentences
        points = [point.strip() for point in summary.split('.') if point.strip()]
        return points 