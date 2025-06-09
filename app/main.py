from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List

from .storage import JSONStorage

app = FastAPI(
    title="Legal AI System",
    description="API for legal document analysis and question answering",
    version="1.0.0"
)

# Configure CORS
origins = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
    "https://lawaichat-one.vercel.app",
    "https://lawaichat-one-git-main.vercel.app",
    "https://lawaichat-one-*.vercel.app",
    "https://lawai-frontend-ten.vercel.app/",
    "https://lawai-backend.vercel.app",
    "https://lawai-frontend-viveks-projects-44c9f3e1.vercel.app/",
    "https://lawai-backend-git-main-viveks-projects-44c9f3e1.vercel.app/",
    "https://lawai-frontend-git-main-viveks-projects-44c9f3e1.vercel.app/",
    "https://lawai-backend-57gzdym3e-viveks-projects-44c9f3e1.vercel.app/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Use the defined origins instead of "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize storage (moved outside of routes for better performance)
storage = JSONStorage()

class QuestionRequest(BaseModel):
    question: str
    category: Optional[str] = None

class ErrorResponse(BaseModel):
    detail: str

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "API is running"}

@app.get("/api/qa-pairs")
async def get_qa_pairs(category: Optional[str] = None, limit: int = 10):
    """Get Q&A pairs with optional category filter."""
    try:
        qa_pairs = storage.get_qa_pairs(category=category, limit=limit)
        return {"qa_pairs": qa_pairs}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Failed to fetch QA pairs: {str(e)}"}
        )

@app.get("/api/categories")
async def get_categories():
    """Get all available categories."""
    try:
        categories = storage.get_categories()
        return {"categories": categories}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Failed to fetch categories: {str(e)}"}
        )

@app.post("/api/legal-qa")
async def answer_question(request: QuestionRequest):
    """Find answer to a legal question."""
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        similar_questions = storage.find_similar_questions(request.question)
        
        if not similar_questions:
            return JSONResponse(
                status_code=200,
                content={
                    "answer": "I apologize, but I couldn't find a suitable answer to your question. Please try rephrasing your question or ask something else.",
                    "confidence": 0
                }
            )
        
        best_match = similar_questions[0]
        confidence = 0.8 if request.question.lower() == best_match['question'].lower() else 0.5
        
        return {
            "answer": best_match['answer'],
            "confidence": confidence
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your request: {str(e)}"
        )

@app.post("/legal-qa")
async def legal_qa_redirect(request: QuestionRequest):
    """Redirect for /legal-qa to maintain compatibility with frontend."""
    return await answer_question(request)