from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

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
    
    # Save uploaded file permanently (for preview)
    settings.ensure_directories()
    uploads_dir = settings.data_dir / "uploads" / notebook_id
    uploads_dir.mkdir(parents=True, exist_ok=True)
    
    # Use original filename but ensure uniqueness
    original_filename = file.filename or "document"
    file_extension = Path(original_filename).suffix
    base_name = Path(original_filename).stem
    # If file already exists, add a suffix
    file_path = uploads_dir / original_filename
    counter = 1
    while file_path.exists():
        file_path = uploads_dir / f"{base_name}_{counter}{file_extension}"
        counter += 1
    
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        # Verify file was saved and has content
        if not file_path.exists():
            raise HTTPException(status_code=500, detail="File was not saved correctly")
        if file_path.stat().st_size == 0:
            file_path.unlink()
            raise HTTPException(status_code=500, detail="File was saved but is empty")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    try:
        result: IngestionResult = ingestion_service.ingest_path(
            notebook_id=notebook_id,
            path=file_path,
            recursive=False,
        )
        
        # Verify file still exists after ingestion
        if not file_path.exists():
            raise HTTPException(status_code=500, detail="File was deleted during ingestion")
        
        return JSONResponse(
            content={
                "notebook_id": result.notebook_id,
                "documents_processed": result.documents_processed,
                "chunks_indexed": result.chunks_indexed,
            }
        )
    except Exception as e:
        # Don't delete files on error - we need them for preview
        # Even if ingestion fails, the file should be available for preview
        import logging
        logging.error(f"Ingestion error for {file_path}: {e}")
        logging.info(f"Keeping file for preview: {file_path}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/list")
async def list_documents(
    request: Request,
    notebook_id: str,
) -> JSONResponse:
    """
    List all documents in a notebook.
    """
    from ..services.vector_store import VectorStoreManager
    
    vector_store: VectorStoreManager = request.app.state.vector_store
    
    try:
        collection = vector_store.get_collection(notebook_id)
        
        # Get all documents from the collection
        results = collection.get()
        
        if not results or not results.get("documents"):
            return JSONResponse(content={"documents": []})
        
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        
        # Group by source file
        source_files: dict[str, dict] = {}
        for doc, metadata in zip(documents, metadatas):
            source_path = metadata.get("source_path", "unknown") if isinstance(metadata, dict) else "unknown"
            if source_path not in source_files:
                source_files[source_path] = {
                    "filename": Path(source_path).name if source_path != "unknown" else "Unknown",
                    "source_path": source_path,
                    "chunk_count": 0,
                    "preview": "",  # First chunk as preview
                }
            source_files[source_path]["chunk_count"] += 1
            # Use first chunk as preview
            if not source_files[source_path]["preview"] and doc:
                source_files[source_path]["preview"] = doc[:200] + "..." if len(doc) > 200 else doc
        
        documents_list = list(source_files.values())
        
        return JSONResponse(content={"documents": documents_list})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@router.get("/preview")
async def preview_document(
    request: Request,
    notebook_id: str,
    source_path: str,
) -> FileResponse:
    """
    Serve a document file for preview.
    """
    from urllib.parse import unquote
    
    settings: AppConfig = request.app.state.settings
    
    # Decode URL-encoded path
    decoded_path = unquote(source_path)
    file_path = Path(decoded_path)
    filename = file_path.name
    
    # Security: ensure the file is within the uploads directory
    uploads_dir = settings.data_dir / "uploads" / notebook_id
    uploads_base = settings.data_dir / "uploads"
    
    import logging
    
    try:
        # Strategy: Try multiple locations in order of preference
        # 1. New format: uploads/{notebook_id}/filename (most likely)
        potential_path = uploads_dir / filename
        if potential_path.exists():
            file_path = potential_path
            logging.info(f"Found file at: {file_path}")
        # 2. Absolute path from metadata (if it exists and is within uploads)
        elif file_path.is_absolute() and file_path.exists():
            resolved_path = file_path.resolve()
            resolved_uploads_base = uploads_base.resolve()
            
            if not str(resolved_path).startswith(str(resolved_uploads_base)):
                raise HTTPException(status_code=403, detail="Access denied")
            
            file_path = resolved_path
            logging.info(f"Found file at absolute path: {file_path}")
        # 3. Old format: uploads/filename (for backward compatibility)
        elif (uploads_base / filename).exists():
            file_path = uploads_base / filename
            logging.info(f"Found file at old location: {file_path}")
        # 4. Search in all notebook subdirectories
        else:
            found = False
            if uploads_base.exists():
                for subdir in uploads_base.iterdir():
                    if subdir.is_dir():
                        candidate = subdir / filename
                        if candidate.exists():
                            file_path = candidate
                            found = True
                            logging.info(f"Found file in subdirectory: {file_path}")
                            break
            
            if not found:
                # Log all possible locations for debugging
                logging.error(f"File not found: {filename}")
                logging.error(f"Searched in: {uploads_dir}")
                logging.error(f"Searched in: {uploads_base}")
                if uploads_base.exists():
                    logging.error(f"Subdirectories: {list(uploads_base.iterdir())}")
                raise HTTPException(
                    status_code=404,
                    detail=f"File not found: {filename}. Notebook ID: {notebook_id}",
                )
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        return FileResponse(
            path=str(file_path),
            filename=file_path.name,
            media_type="application/octet-stream",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to serve file: {str(e)}")
