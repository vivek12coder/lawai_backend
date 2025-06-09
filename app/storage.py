import json
from datetime import datetime
from typing import Dict, List, Optional
import re
from difflib import SequenceMatcher

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

def clean_text(text: str) -> str:
    """Clean and normalize text for better matching."""
    # Convert to lowercase and remove extra whitespace
    text = text.lower().strip()
    # Remove special characters except basic punctuation
    text = re.sub(r'[^\w\s\-.,?!]', '', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text

def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate text similarity using SequenceMatcher."""
    text1 = clean_text(text1)
    text2 = clean_text(text2)
    
    # Quick exact match check
    if text1 == text2:
        return 1.0
        
    # Check for containment
    if text1 in text2 or text2 in text1:
        return 0.8
        
    # Use sequence matcher for more detailed comparison
    return SequenceMatcher(None, text1, text2).ratio()

class JSONStorage:
    def __init__(self):
        self._data = INITIAL_DATA
        self._validate_data()

    def _validate_data(self):
        """Validate the data structure and content."""
        if not isinstance(self._data, dict):
            raise ValueError("Invalid data format: root must be a dictionary")
        
        if 'qa_pairs' not in self._data:
            raise ValueError("Invalid data format: missing 'qa_pairs' key")
            
        if not isinstance(self._data['qa_pairs'], list):
            raise ValueError("Invalid data format: 'qa_pairs' must be a list")
            
        # Validate each QA pair
        for qa in self._data['qa_pairs']:
            required_fields = ['id', 'question', 'answer', 'category']
            for field in required_fields:
                if field not in qa:
                    raise ValueError(f"Invalid QA pair: missing '{field}' field")

    def _load_data(self) -> Dict:
        """Load data from memory."""
        return self._data

    def _save_data(self, data: Dict):
        """Save data to memory."""
        self._data = data
        self._validate_data()

    def get_qa_pairs(self, category: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """Get Q&A pairs with optional category filter."""
        try:
            data = self._load_data()
            qa_pairs = data['qa_pairs']
            
            # Apply category filter if specified
            if category:
                category = category.lower()
                qa_pairs = [qa for qa in qa_pairs if qa.get('category', '').lower() == category]
            
            # Sort by creation date (newest first) and apply limit
            def get_created_at(qa):
                try:
                    return qa.get('created_at', '')
                except:
                    return ''
            
            qa_pairs.sort(key=get_created_at, reverse=True)
            return qa_pairs[:limit]
        except Exception as e:
            raise ValueError(f"Error fetching QA pairs: {str(e)}")

    def find_similar_questions(self, question: str, threshold: float = 0.5) -> List[Dict]:
        """Find similar questions using improved text matching."""
        try:
            data = self._load_data()
            question = clean_text(question)
            
            # Calculate similarity scores for all questions
            scored_questions = []
            for qa in data['qa_pairs']:
                similarity = calculate_similarity(question, qa['question'])
                if similarity >= threshold:
                    scored_questions.append((similarity, qa))
            
            # Sort by similarity score (highest first)
            scored_questions.sort(reverse=True, key=lambda x: x[0])
            
            # Return the questions only
            return [qa for score, qa in scored_questions]
        except Exception as e:
            raise ValueError(f"Error finding similar questions: {str(e)}")

    def get_categories(self) -> List[str]:
        """Get all unique categories."""
        try:
            data = self._load_data()
            # Get unique categories and filter out empty ones
            categories = set(qa.get('category', '').strip() for qa in data['qa_pairs'])
            return sorted([cat for cat in categories if cat])
        except Exception as e:
            raise ValueError(f"Error fetching categories: {str(e)}")