from fastapi import FastAPI, HTTPException, Query, Response, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import re
import httpx
import time

from .storage import JSONStorage
from .config import settings

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
    "https://lawai-frontend-git-main-viveks-projects-44c9f3e1.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Use specific origins instead of "*"
    allow_origin_regex=r"https://lawai-frontend-.*\.vercel\.app$",
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Specify allowed methods
    allow_headers=["Content-Type", "Authorization", "Accept"],  # Specify allowed headers
    expose_headers=["Content-Length"],  # Specify headers that can be exposed to the browser
    max_age=600,  # Cache preflight requests for 10 minutes
)

# Initialize storage
storage = JSONStorage()

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    category: Optional[str] = Field(None, max_length=100)

class ErrorResponse(BaseModel):
    detail: str

async def fetch_gemini_answer(question: str) -> Optional[str]:
    """Call Google Gemini API to get an answer as a fallback.
    Returns the response text or None on failure.
    """
    api_key = settings.GEMINI_API_KEY
    if not api_key:
        return None
    url = f"{settings.GEMINI_API_URL}/models/{settings.GEMINI_MODEL}:generateContent?key={api_key}"
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": (
                            "You are a helpful Indian legal assistant. "
                            "Answer concisely and accurately. If the question is non-legal, say you don't have enough context.\n\n"
                            f"Question: {question}"
                        )
                    }
                ]
            }
        ]
    }
    try:
        timeout = httpx.Timeout(15.0, connect=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code != 200:
                return None
            data = resp.json()
            # Expected: candidates[0].content.parts[0].text
            candidates = data.get("candidates") or []
            if not candidates:
                return None
            content = candidates[0].get("content") or {}
            parts = content.get("parts") or []
            if not parts:
                return None
            text = parts[0].get("text")
            return text
    except Exception:
        return None

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "API is running"}

@app.get("/")
async def root():
    """Root endpoint with API info and links."""
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "endpoints": {
            "health": "/health",
            "legal_qa": "/legal-qa",
            "qa_pairs": "/api/qa-pairs",
            "categories": "/api/categories",
            "docs": "/docs"
        }
    }

@app.get("/favicon.ico")
async def favicon():
    """Return empty 204 to suppress favicon 404 noise."""
    return Response(status_code=204)

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
            # Fallback to Gemini if no local answer
            gemini_answer = await fetch_gemini_answer(question)
            if gemini_answer:
                return {
                    "answer": gemini_answer,
                    "confidence": 0.6,
                    "category": "general",
                    "source": "gemini"
                }
            return {
                "answer": "I apologize, but I couldn't find a suitable answer to your question. Please try rephrasing your question or ask something else.",
                "confidence": 0,
                "category": "general",
                "source": "local"
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

@app.post("/api/analyze-document")
async def analyze_document(
    file: UploadFile = File(...),
    title: str = Form(...)
):
    """Analyze a legal document and provide a summary."""
    try:
        # Check file size
        file_size = 0
        contents = b""
        
        # Read file in chunks to avoid memory issues
        while chunk := await file.read(1024 * 1024):  # 1MB chunks
            contents += chunk
            file_size += len(chunk)
            if file_size > settings.MAX_DOCUMENT_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size is {settings.MAX_DOCUMENT_SIZE/(1024*1024)}MB"
                )
        
        # Check content type
        content_type = file.content_type
        if content_type not in settings.SUPPORTED_DOCUMENT_TYPES:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type: {content_type}. Supported types: {', '.join(settings.SUPPORTED_DOCUMENT_TYPES)}"
            )
        
        # For now, return a simple response
        # In a real implementation, you would process the document and extract meaningful information
        document_id = f"doc_{int(time.time())}"
        
        return {
            "summary": f"Analysis of {title}: This appears to be a legal document related to Indian law.",
            "category": "general_legal",
            "document_id": document_id
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze document: {str(e)}"
        )