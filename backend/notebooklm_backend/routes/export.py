from __future__ import annotations

import io
import zipfile
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..models.export import ConversationExportRequest
from ..services.vector_store import VectorStoreManager

router = APIRouter(prefix="/export", tags=["export"])


@router.post("/conversation")
async def export_conversation(payload: ConversationExportRequest) -> StreamingResponse:
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    filename = f"{payload.title.replace(' ', '_')}-{timestamp}.md"
    lines = [f"# {payload.title}", ""]
    for msg in payload.messages:
        role = msg.role.capitalize()
        lines.append(f"**{role}:** {msg.content}")
        lines.append("")
    markdown = "\n".join(lines).encode("utf-8")
    return StreamingResponse(
        io.BytesIO(markdown),
        media_type="text/markdown",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


@router.get("/notebook/{notebook_id}")
async def export_notebook(request: Request, notebook_id: str) -> StreamingResponse:
    vector_store: VectorStoreManager = request.app.state.vector_store
    summaries = vector_store.get_all_document_summaries(notebook_id)
    if not summaries:
        raise HTTPException(status_code=404, detail="No summaries found for this notebook")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for summary in summaries:
            name = Path(summary.source_path).name or "document.txt"
            content = f"# {name}\n\n{summary.summary}\n\nChunks indexed: {summary.chunk_count}\n"
            zf.writestr(f"{name}.md", content)
    buffer.seek(0)
    filename = f"notebook-{notebook_id[:8]}.zip"
    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
