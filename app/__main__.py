"""Entry point: ``python -m app`` starts the API server."""

import uvicorn

from app.core.config import settings
from app.main import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host=settings.host, port=settings.port)
