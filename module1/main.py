"""
Module 1 — Standalone FastAPI Application
==========================================
This is the self-contained reference server for Module 1.
In production the /m1/process endpoint is served through module2's backend
on port 8000.  Run this file ONLY for standalone testing of Module 1.

    cd module1/backend
    uvicorn main:app --port 8001 --reload
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.process import router as process_router

app = FastAPI(
    title="PROJECT AKSHAR — Module 1 API",
    description="SAM-based document page extraction and enhancement pipeline.",
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(process_router, tags=["Module 1"])

@app.get("/", tags=["Health"])
async def root():
    return {
        "module": "Module 1 — SAM Pipeline",
        "status": "running",
        "endpoints": {"process": "POST /m1/process"},
        "docs": "/docs",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
