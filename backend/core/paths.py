from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
FEEDBACK_DATA_DIR = PROJECT_ROOT / "data" / "feedback"
FEEDBACK_DB_PATH = FEEDBACK_DATA_DIR / "mathrag.db"
STORAGE_DIR = PROJECT_ROOT / "storage"
DOCUMENTS_DIR = STORAGE_DIR / "documents"
INDEXES_DIR = STORAGE_DIR / "indexes"
BACKEND_DB_PATH = STORAGE_DIR / "mathrag_backend.db"
