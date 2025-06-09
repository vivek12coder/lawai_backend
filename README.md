# LawAI Backend

## Overview
This is the backend API for the LawAI application, a legal AI assistant for Indian law. It provides endpoints for legal question answering and document analysis.

## Deployment
The backend is configured for deployment on Vercel. The frontend is deployed at `https://lawai-frontend-ten.vercel.app`.

## API Endpoints

### Legal Question Answering
- **POST /api/legal-qa**: Answer legal questions
  - Request body: `{"question": "string", "category": "string" (optional)}`
  - Response: `{"answer": "string", "confidence": number}`

### QA Pairs
- **GET /api/qa-pairs**: Get question-answer pairs
  - Query parameters: `category` (optional), `limit` (default: 10)
  - Response: `{"qa_pairs": [{"question": "string", "answer": "string", "category": "string"}]}`

### Categories
- **GET /api/categories**: Get all available categories
  - Response: `{"categories": ["string"]}`

### Health Check
- **GET /health**: Check if the API is running
  - Response: `{"status": "healthy", "message": "API is running"}`

## Connecting Frontend and Backend

### CORS Configuration
The backend is configured to accept requests from the following origins:
- Local development: `http://localhost:3000`, `http://localhost:8000`
- Vercel deployments: 
  - `https://lawaichat-one.vercel.app`
  - `https://lawaichat-one-git-main.vercel.app`
  - `https://lawai-frontend-ten.vercel.app`

### Frontend Configuration
In your frontend application, make API requests to the backend using the following base URL:

```javascript
// For production
const API_BASE_URL = 'https://your-backend-deployment-url.vercel.app';

// For local development
// const API_BASE_URL = 'http://localhost:8000';

// Example API call
async function askLegalQuestion(question, category = null) {
  const response = await fetch(`${API_BASE_URL}/api/legal-qa`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ question, category }),
  });
  
  return await response.json();
}
```

## Local Development

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the server:
   ```
   uvicorn app.main:app --reload
   ```

3. The API will be available at `http://localhost:8000`