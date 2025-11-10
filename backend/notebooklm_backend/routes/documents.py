from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from ..config import AppConfig
from ..services.ingestion import IngestionService, IngestionResult

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/ingest")
async def ingest_document(
    request: Request,
    file: UploadFile = File(...),
    notebook_id: str | None = Form(None),
) -> JSONResponse:
    """
    Upload and ingest a document into a notebook.
    """
    settings: AppConfig = request.app.state.settings
    ingestion_service: IngestionService = request.app.state.ingestion_service
    
    if not notebook_id:
        notebook_id = uuid.uuid4().hex
    
    # Save uploaded file temporarily
    settings.ensure_directories()
    temp_dir = settings.data_dir / "uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = temp_dir / file.filename
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    try:
        result: IngestionResult = ingestion_service.ingest_path(
            notebook_id=notebook_id,
            path=file_path,
            recursive=False,
        )
        
        # Clean up temp file
        try:
            file_path.unlink()
        except Exception:
            pass
        
        return JSONResponse(
            content={
                "notebook_id": result.notebook_id,
                "documents_processed": result.documents_processed,
                "chunks_indexed": result.chunks_indexed,
            }
        )
    except Exception as e:
        # Clean up temp file on error
        try:
            file_path.unlink()
        except Exception:
            pass
        raise HTTPException(status_code=400, detail=str(e))

