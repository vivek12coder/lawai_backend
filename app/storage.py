import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

class JSONStorage:
    def __init__(self):
        self.data_dir = Path(__file__).parent.parent / 'data'
        self.qa_file = self.data_dir / 'qa_data.json'
        self._ensure_data_file()

    def _ensure_data_file(self):
        """Ensure the data directory and file exist."""
        self.data_dir.mkdir(exist_ok=True)
        if not self.qa_file.exists():
            self._save_data({"qa_pairs": []})

    def _load_data(self) -> Dict:
        """Load data from JSON file."""
        try:
            with open(self.qa_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading data: {e}")
            return {"qa_pairs": []}

    def _save_data(self, data: Dict):
        """Save data to JSON file."""
        try:
            with open(self.qa_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving data: {e}")

    def add_qa_pair(self, question: str, answer: str, category: str = "general") -> Dict:
        """Add a new Q&A pair."""
        data = self._load_data()
        
        # Generate new ID
        new_id = max([qa['id'] for qa in data['qa_pairs']], default=0) + 1
        
        # Create new Q&A pair
        qa_pair = {
            "id": new_id,
            "question": question,
            "answer": answer,
            "category": category,
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Add to data and save
        data['qa_pairs'].append(qa_pair)
        self._save_data(data)
        
        return qa_pair

    def get_qa_pairs(self, category: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """Get Q&A pairs with optional category filter."""
        data = self._load_data()
        qa_pairs = data['qa_pairs']
        
        # Apply category filter if specified
        if category:
            qa_pairs = [qa for qa in qa_pairs if qa['category'] == category]
        
        # Sort by creation date (newest first) and apply limit
        qa_pairs.sort(key=lambda x: x['created_at'], reverse=True)
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
        return list(set(qa['category'] for qa in data['qa_pairs'])) 