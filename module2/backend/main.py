"""
PROJECT AKSHAR - Main FastAPI Application
==========================================
Unified backend for all modules:
  - Module 2: Image Workbench (upload, transform, dewarp, deskew, enhance, export)
  - Module 3: OCR Layout Extraction (via pipeline router)
  - Module 4: ChromaDB RAG QA (via pipeline router)
  - Pipeline: Orchestration (detect, extract pages, assemble PDF)
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from routes.upload        import router as upload_router
from routes.transform     import router as transform_router
from routes.enhance       import router as enhance_router
from routes.export        import router as export_router
from routes.dewarp        import router as dewarp_router
from routes.pipeline      import router as pipeline_router
from routes.scantailor    import router as scantailor_router
from routes.corners      import router as corners_router
# Module 1 — SAM-based document processing pipeline
from routes.module1_process import router as module1_router

# ---------------------------------------------------------------------------
# App Initialization
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PROJECT AKSHAR API",
    description=(
        "Full-stack AI document processing system: "
        "image workbench, OCR layout extraction, and RAG-based QA with references."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS — Allow React dev server on common ports
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Static File Directories
# ---------------------------------------------------------------------------

BACKEND_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BACKEND_DIR, "uploads")
PROCESSED_DIR = os.path.join(BACKEND_DIR, "processed")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

app.mount("/static/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/static/processed", StaticFiles(directory=PROCESSED_DIR), name="processed")

# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
async def root():
    """Health check and endpoint directory."""
    return {
        "project": "PROJECT AKSHAR",
        "version": "2.0.0",
        "status": "running",
        "modules": {
            "module1_sam_pipeline": {
                "process": "POST /m1/process",
            },
            "module2_image_workbench": {
                "upload": "POST /upload",
                "transform": "POST /transform",
                "dewarp": "POST /dewarp",
                "deskew_auto": "POST /deskew",
                "deskew_manual": "POST /deskew/manual",
                "enhance_otsu": "POST /enhance/otsu",
                "enhance_adaptive": "POST /enhance/adaptive",
                "export_pdf": "POST /export/pdf",
                "corners_detect": "POST /corners/detect",
                "corners_apply": "POST /corners/apply",
            },
            "pipeline_orchestration": {
                "upload": "POST /pipeline/upload",
                "detect_type": "POST /pipeline/detect",
                "extract_pages": "POST /pipeline/extract-pages",
                "convert_to_pdf": "POST /pipeline/convert-to-pdf",
                "run_ocr": "POST /pipeline/run-ocr",
                "index": "POST /pipeline/index",
                "query": "POST /pipeline/query",
            },
        },
        "docs": "/docs",
    }

# ---------------------------------------------------------------------------
# Router Registration
# ---------------------------------------------------------------------------

app.include_router(upload_router,    tags=["Upload"])
app.include_router(transform_router, tags=["Transform"])
app.include_router(dewarp_router,    tags=["Dewarp & Deskew"])
app.include_router(enhance_router,   tags=["Enhancement"])
app.include_router(export_router,    tags=["Export"])
app.include_router(pipeline_router)
app.include_router(scantailor_router,  tags=["ScanTailor"])
app.include_router(corners_router,     tags=["Corner Detection"])
# Module 1 — registered under /m1 prefix
app.include_router(module1_router,   tags=["Module 1"])

# ---------------------------------------------------------------------------
# Dev Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
