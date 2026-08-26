"""FastAPI Application Entrypoint."""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import health, documents, chat, evaluation
from api.dependencies import get_container
from core.config import settings
from core.logger import logger
from core.exceptions import RAGException


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION} [{settings.ENVIRONMENT}]")
    get_container()
    yield
    logger.info(f"Shutting down {settings.PROJECT_NAME}")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Production-grade Agentic Retrieval-Augmented Generation (RAG) backend service.",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Enable CORS for Streamlit / Frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RAGException)
async def rag_exception_handler(request: Request, exc: RAGException):
    """Domain exception handler."""
    return JSONResponse(
        status_code=400,
        content={"error": exc.__class__.__name__, "message": exc.message, "details": exc.details},
    )


# Register routes
app.include_router(health.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(evaluation.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    )
