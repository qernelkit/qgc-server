"""FastAPI application for QGC Server."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from .routes import gadgets_router, compile_router, resolve_router, ingest_router, changes_router
from .services.registry import get_registry


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup: Initialize registry (loads gadgets from disk)
    registry = get_registry()
    gadget_count = len(registry.get_all_gadgets())
    print(f"QGC Server started with {gadget_count} gadgets loaded")
    yield
    # Shutdown: nothing to clean up


app = FastAPI(
    title="Quantum Gadget Catalog Server",
    description="Registry server for quantum circuit gadgets",
    version="0.1.0",
    lifespan=lifespan,
)

# Mount routes
# IMPORTANT: /compile, /resolve, /ingest are at root, NOT under /gadgets
app.include_router(compile_router)
app.include_router(resolve_router)
app.include_router(ingest_router, prefix="/ingest", tags=["ingest"])
app.include_router(gadgets_router, prefix="/gadgets", tags=["gadgets"])
app.include_router(changes_router, prefix="/gadgets", tags=["changes"])


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/stats")
async def database_stats():
    """Get database statistics."""
    from .services.database import get_database
    db = get_database()
    stats = db.get_stats()
    stats["database_path"] = str(db.db_path)
    return stats


def run():
    """Entry point for qgc-server command."""
    import uvicorn
    from .config import settings

    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    run()
