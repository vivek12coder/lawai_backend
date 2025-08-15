from typing import Optional, List

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _SENTENCE_TF_AVAILABLE = True
except Exception:
    SentenceTransformer = None  # type: ignore
    _SENTENCE_TF_AVAILABLE = False

try:
    import numpy as np  # type: ignore
    _NUMPY_AVAILABLE = True
except Exception:
    np = None  # type: ignore
    _NUMPY_AVAILABLE = False

try:
    from ..app.firebase import legal_qa_collection  # type: ignore
    _FIREBASE_AVAILABLE = True
except Exception:
    legal_qa_collection = None  # type: ignore
    _FIREBASE_AVAILABLE = False


class LegalQA:
    """Legal Question Answering system using semantic search."""

    def __init__(self):
        """Initialize the QA system with the sentence transformer model."""
        if not _SENTENCE_TF_AVAILABLE:
            raise RuntimeError("sentence_transformers is not installed. Install it to use LegalQA.")
        self.model = SentenceTransformer('all-mpnet-base-v2')

    def encode_question(self, question: str) -> List[float]:
        """Encode a question into a vector using the sentence transformer."""
        return self.model.encode(question).tolist()

    async def find_similar_questions(
        self,
        question: str,
        threshold: float = 0.7,
        max_results: int = 5
    ) -> List[dict]:
        """Find similar questions in the database (Firebase)."""
        if not _FIREBASE_AVAILABLE:
            raise RuntimeError("Firebase is not configured in this project (missing app.firebase).")
        if not _NUMPY_AVAILABLE:
            raise RuntimeError("numpy is required to compute similarities.")
        # Encode the input question
        question_embedding = self.model.encode(question)

        # Get all QA pairs from Firestore
        qa_docs = legal_qa_collection.stream()

        similarities = []
        for qa in qa_docs:
            qa_data = qa.to_dict()
            stored_embedding = qa_data.get('embedding')

            if stored_embedding:
                # Convert stored embedding back to numpy array
                stored_embedding = np.array(stored_embedding)
                # Calculate cosine similarity
                similarity = np.dot(question_embedding, stored_embedding) / (
                    np.linalg.norm(question_embedding) * np.linalg.norm(stored_embedding)
                )

                if similarity >= threshold:
                    similarities.append({
                        'id': qa.id,
                        'question': qa_data['question'],
                        'answer': qa_data['answer'],
                        'similarity': float(similarity)
                    })

        # Sort by similarity score and return top results
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:max_results]

    async def find_answer(self, question: str) -> Optional[str]:
        """Find the best answer for a given question."""
        similar_questions = await self.find_similar_questions(question)

        if not similar_questions:
            return None

        # Return the answer with highest similarity
        return similar_questions[0]['answer']

    async def add_qa_pair(
        self,
        question: str,
        answer: str,
        category: str = "general"
    ) -> str:
        """Add a new QA pair to the database."""
        if not _FIREBASE_AVAILABLE:
            raise RuntimeError("Firebase is not configured in this project (missing app.firebase).")
        # Encode the question
        embedding = self.encode_question(question)

        # Add to Firestore
        doc_ref = legal_qa_collection.add({
            'question': question,
            'answer': answer,
            'category': category,
            'embedding': embedding
        })

        return doc_ref.id

    def preprocess_question(self, question: str) -> str:
        """Preprocess the question for better matching."""
        # TODO: Implement question preprocessing
        # - Remove punctuation
        # - Normalize text
        # - Remove stop words
        # - Lemmatization
        return question.strip().lower()

    async def get_question_categories(self) -> List[str]:
        """Get all unique categories of questions in the database (Firebase)."""
        if not _FIREBASE_AVAILABLE:
            raise RuntimeError("Firebase is not configured in this project (missing app.firebase).")
        docs = legal_qa_collection.stream()

        categories = set()

        for doc in docs:
            category = doc.to_dict().get('category')
            if category:
                categories.add(category)

        return sorted(list(categories))