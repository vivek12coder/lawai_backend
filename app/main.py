from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import uvicorn

from .storage import JSONStorage

app = FastAPI(
    title="Legal AI System",
    description="API for legal document analysis and question answering",
    version="1.0.0",
    root_path=""  # This helps with path handling in serverless
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
    "https://lawai-frontend-ten.vercel.app",
    "*"  # Allow all origins in development
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize storage
storage = JSONStorage()

class QuestionRequest(BaseModel):
    question: str
    category: Optional[str] = None

class ErrorResponse(BaseModel):
    detail: str

@app.post("/api/legal-qa", responses={
    200: {"description": "Successful response"},
    400: {"model": ErrorResponse},
    500: {"model": ErrorResponse}
})
async def answer_question(request: QuestionRequest):
    """Find answer to a legal question."""
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        # Find similar questions
        similar_questions = storage.find_similar_questions(request.question)
        
        if not similar_questions:
            return JSONResponse(
                status_code=200,
                content={
                    "answer": "I apologize, but I couldn't find a suitable answer to your question. Please try rephrasing your question or ask something else.",
                    "confidence": 0
                }
            )
        
        # Use the most similar question's answer
        best_match = similar_questions[0]
        confidence = 0.8 if request.question.lower() == best_match['question'].lower() else 0.5
        
        return JSONResponse(
            status_code=200,
            content={
                "answer": best_match['answer'],
                "confidence": confidence
            }
        )
    except HTTPException as he:
        return JSONResponse(
            status_code=he.status_code,
            content={"detail": he.detail}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"An error occurred while processing your request: {str(e)}"}
        )

@app.get("/api/qa-pairs", responses={
    200: {"description": "Successful response"},
    500: {"model": ErrorResponse}
})
async def get_qa_pairs(category: Optional[str] = None, limit: int = 10):
    """Get Q&A pairs with optional category filter."""
    try:
        qa_pairs = storage.get_qa_pairs(category=category, limit=limit)
        return JSONResponse(
            status_code=200,
            content={"qa_pairs": qa_pairs}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Failed to fetch QA pairs: {str(e)}"}
        )

@app.get("/api/categories", responses={
    200: {"description": "Successful response"},
    500: {"model": ErrorResponse}
})
async def get_categories():
    """Get all available categories."""
    try:
        categories = storage.get_categories()
        return JSONResponse(
            status_code=200,
            content={"categories": categories}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Failed to fetch categories: {str(e)}"}
        )

# Add a health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API is running"}

# Add a redirect for the /legal-qa endpoint
@app.post("/legal-qa")
async def legal_qa_redirect(request: QuestionRequest):
    """Redirect for /legal-qa to maintain compatibility with frontend."""
    return await answer_question(request)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)