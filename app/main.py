from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import re

from .storage import JSONStorage

app = FastAPI(
    title="Legal AI System",
    description="API for legal document analysis and question answering",
    version="1.0.0"
)

# Configure CORS with all necessary origins
origins = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
   "https://lawai-frontend-ten.vercel.app",
   "https://lawai-backend.vercel.app",
    "https://lawai-frontend-git-main-viveks-projects-44c9f3e1.vercel.app",
    "https://lawai-frontend-*.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Use specific origins instead of "*"
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Specify allowed methods
    allow_headers=["*"],
)

# Initialize storage
storage = JSONStorage()

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    category: Optional[str] = Field(None, max_length=100)

class ErrorResponse(BaseModel):
    detail: str

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "API is running"}

@app.post("/legal-qa")
async def legal_qa(request: QuestionRequest):
    """Main endpoint for legal questions"""
    try:
        # Clean and validate the question
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Remove any potentially harmful characters
        question = re.sub(r'[^\w\s\-.,?!]', '', question)

        similar_questions = storage.find_similar_questions(question)
        
        if not similar_questions:
            return {
                "answer": "I apologize, but I couldn't find a suitable answer to your question. Please try rephrasing your question or ask something else.",
                "confidence or data not found..": 0
            }
        
        best_match = similar_questions[0]
        # Calculate confidence based on question similarity
        question_lower = question.lower()
        best_match_lower = best_match['question'].lower()
        
        # Improved confidence calculation
        if question_lower == best_match_lower:
            confidence = 1.0
        elif question_lower in best_match_lower or best_match_lower in question_lower:
            confidence = 0.8
        else:
            confidence = 0.5
        
        return {
            "answer": best_match['answer'],
            "confidence": confidence,
            "category": best_match.get('category', 'general')
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your request: {str(e)}"
        )

@app.get("/api/qa-pairs")
async def get_qa_pairs(
    category: Optional[str] = None,
    limit: int = Query(default=10, ge=1, le=100)  # Add validation for limit
):
    """Get Q&A pairs with optional category filter."""
    try:
        qa_pairs = storage.get_qa_pairs(category=category, limit=limit)
        return {"qa_pairs": qa_pairs, "total": len(qa_pairs)}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch QA pairs: {str(e)}"
        )

@app.get("/api/categories")
async def get_categories():
    """Get all available categories."""
    try:
        categories = storage.get_categories()
        return {"categories": categories, "total": len(categories)}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch categories: {str(e)}"
        )