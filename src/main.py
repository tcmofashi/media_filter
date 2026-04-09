"""Application entry point"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import uvicorn


def main():
    """Run the FastAPI application"""
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=31211,
        reload=True,
        reload_dirs=[str(project_root / "src")],
    )


if __name__ == "__main__":
    main()
