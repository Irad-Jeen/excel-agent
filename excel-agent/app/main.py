# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .controller import router as excel_router

app = FastAPI(
    title="Excel Agent API",
    version="1.0.0",
    description="Smart Excel analysis agent powered by Gemma (OpenRouter). JSON-only (Base64) input.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(excel_router, prefix="", tags=["Excel Analysis"])

@app.get("/")
async def root():
    return {
        "message": "Excel Agent is running! JSON-only endpoint is /chat/analyze",
        "endpoints": ["/chat/analyze"],
    }
