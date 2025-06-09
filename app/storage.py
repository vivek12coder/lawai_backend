import json
from datetime import datetime
from typing import Dict, List, Optional

# Load the initial data
INITIAL_DATA = {
    "qa_pairs": [
        {
            "id": 1,
            "question": "What is law?",
            "answer": "Law is a system of rules created and enforced through social or governmental institutions to regulate behavior. In India, it encompasses constitutional law, statutory law, customary law, and case law.",
            "category": "legal_basics",
            "created_at": "2024-03-21T10:00:00Z"
        },
        {
            "id": 2,
            "question": "What is the Constitution of India?",
            "answer": "The Constitution of India is the supreme law of India, adopted on 26th January 1950. It lays down the framework defining fundamental political principles and establishes the structure, procedures, powers, and duties of government institutions.",
            "category": "constitutional_law",
            "created_at": "2024-03-21T10:01:00Z"
        }
    ]
}

class JSONStorage:
    def __init__(self):
        self._data = INITIAL_DATA

    def _load_data(self) -> Dict:
        """Load data from memory."""
        return self._data

    def _save_data(self, data: Dict):
        """Save data to memory."""
        self._data = data

    def get_qa_pairs(self, category: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """Get Q&A pairs with optional category filter."""
        data = self._load_data()
        qa_pairs = data['qa_pairs']
        
        # Apply category filter if specified
        if category:
            qa_pairs = [qa for qa in qa_pairs if qa.get('category') == category]
        
        # Sort by creation date (newest first) and apply limit
        def get_created_at(qa):
            try:
                return qa.get('created_at', '')
            except:
                return ''
        
        qa_pairs.sort(key=get_created_at, reverse=True)
        return qa_pairs[:limit]

    def find_similar_questions(self, question: str, threshold: float = 0.5) -> List[Dict]:
        """Find similar questions using simple string matching."""
        data = self._load_data()
        question = question.lower()
        
        similar_questions = []
        for qa in data['qa_pairs']:
            # Calculate similarity (simple contains check for now)
            if question in qa['question'].lower() or qa['question'].lower() in question:
                similar_questions.append(qa)
        
        return similar_questions

    def get_categories(self) -> List[str]:
        """Get all unique categories."""
        data = self._load_data()
        return list(set(qa.get('category', '') for qa in data['qa_pairs'])) 