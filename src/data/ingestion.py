"""
ingestion.py
============
Mengambil data mentah dari Azure Blob Storage.
Jika koneksi Azure tidak tersedia, otomatis fallback ke file lokal di data/raw/.

Cara pakai:
    python -m src.data.ingestion
    atau
    from src.data.ingestion import run_ingestion
    run_ingestion()
"""

import os
import logging
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent.parent
RAW_DIR  = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Mapping: nama blob di Azure → path lokal tujuan
BLOB_FILES = {
    "produksi_padi.csv" : RAW_DIR / "produksi_padi.csv",
    "harga_beras.csv"   : RAW_DIR / "harga_beras.csv",
    "cuaca_bmkg.csv"    : RAW_DIR / "cuaca_bmkg.csv",
    "enso_index.csv"    : RAW_DIR / "enso_index.csv",
    "ndvi_sentinel.csv" : RAW_DIR / "ndvi_sentinel.csv",
}

# ─────────────────────────────────────────────
# AZURE BLOB
# ─────────────────────────────────────────────
def _download_blob(container: str, blob_name: str, dest_path: Path) -> bool:
    """Download satu file dari Azure Blob Storage. Return True jika berhasil."""
    try:
        from azure.storage.blob import BlobServiceClient

        conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if not conn_str:
            logger.warning("AZURE_STORAGE_CONNECTION_STRING tidak ada di .env")
            return False

        client      = BlobServiceClient.from_connection_string(conn_str)
        blob_client = client.get_blob_client(container=container, blob=blob_name)

        with open(dest_path, "wb") as f:
            blob_client.download_blob().readinto(f)

        logger.info(f"✅ Downloaded: {blob_name} → {dest_path.name}")
        return True

    except ImportError:
        logger.warning("Package azure-storage-blob tidak terinstall.")
        logger.warning("Install dengan: pip install azure-storage-blob")
        return False
    except Exception as e:
        logger.warning(f"❌ Gagal download {blob_name}: {e}")
        return False


def ingest_from_azure(container: str = "sipadi-data") -> bool:
    """
    Download semua file raw data dari Azure Blob Storage.

    Args:
        container: nama container di Azure Blob Storage

    Returns:
        True jika semua file berhasil didownload
    """
    logger.info(f"Memulai ingestion dari Azure Blob container: '{container}'")
    results = {}

    for blob_name, dest_path in BLOB_FILES.items():
        results[blob_name] = _download_blob(container, blob_name, dest_path)

    success_count = sum(results.values())
    total         = len(results)
    logger.info(f"Azure ingestion selesai: {success_count}/{total} file berhasil")

    return success_count == total


# ─────────────────────────────────────────────
# LOCAL FALLBACK
# ─────────────────────────────────────────────
def check_local_files() -> dict:
    """
    Cek ketersediaan file lokal di data/raw/.

    Returns:
        dict {nama_file: True/False}
    """
    status = {}
    for fname, fpath in BLOB_FILES.items():
        exists        = fpath.exists()
        status[fname] = exists
        icon          = "✅" if exists else "❌"
        logger.info(f"{icon} {fname}: {'Ada' if exists else 'Tidak ditemukan'} ({fpath})")
    return status


def validate_raw_files() -> bool:
    """
    Validasi bahwa semua file raw tersedia dan bisa dibaca.

    Returns:
        True jika semua file valid
    """
    all_ok = True
    for fname, fpath in BLOB_FILES.items():
        if not fpath.exists():
            logger.error(f"File tidak ditemukan: {fpath}")
            all_ok = False
            continue
        try:
            df = pd.read_csv(fpath, nrows=5)
            if df.empty:
                logger.warning(f"File kosong: {fname}")
                all_ok = False
            else:
                logger.info(f"✅ {fname}: {fpath.stat().st_size / 1024:.1f} KB, "
                            f"kolom: {list(df.columns)}")
        except Exception as e:
            logger.error(f"Gagal baca {fname}: {e}")
            all_ok = False

    return all_ok


# ─────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────
def run_ingestion(container: str = "sipadi-data", force_azure: bool = False) -> bool:
    """
    Jalankan proses ingestion lengkap.

    Alur:
    1. Cek file lokal → jika semua ada dan force_azure=False, skip download
    2. Coba download dari Azure Blob
    3. Jika Azure gagal, gunakan file lokal yang ada
    4. Validasi semua file

    Args:
        container   : nama Azure Blob container
        force_azure : paksa download ulang dari Azure meski file lokal sudah ada

    Returns:
        True jika pipeline siap dilanjutkan
    """
    logger.info("=" * 55)
    logger.info("SiPADI — Data Ingestion Pipeline")
    logger.info("=" * 55)

    # Step 1: Cek file lokal
    local_status  = check_local_files()
    all_local_ok  = all(local_status.values())

    if all_local_ok and not force_azure:
        logger.info("Semua file lokal tersedia. Melewati download Azure.")
        logger.info("(Gunakan force_azure=True untuk paksa download ulang)")
    else:
        # Step 2: Coba Azure
        missing = [f for f, ok in local_status.items() if not ok]
        if missing:
            logger.info(f"File hilang: {missing}")
        logger.info("Mencoba download dari Azure Blob Storage...")
        azure_ok = ingest_from_azure(container)

        if not azure_ok:
            logger.warning("Sebagian/semua download Azure gagal.")
            logger.warning("Pastikan AZURE_STORAGE_CONNECTION_STRING sudah diset di .env")
            logger.warning("dan file sudah diupload ke container Azure Blob.")

    # Step 3: Validasi akhir
    logger.info("\nValidasi file raw data:")
    all_valid = validate_raw_files()

    if all_valid:
        logger.info("\n✅ Ingestion selesai. Semua file siap untuk cleaning.")
    else:
        logger.error("\n❌ Ada file yang belum tersedia. Cek log di atas.")

    return all_valid


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SiPADI Data Ingestion")
    parser.add_argument("--container",    default="sipadi-data",
                        help="Nama Azure Blob container (default: sipadi-data)")
    parser.add_argument("--force-azure",  action="store_true",
                        help="Paksa download ulang dari Azure meski file lokal ada")
    parser.add_argument("--check-only",   action="store_true",
                        help="Hanya cek status file tanpa download")
    args = parser.parse_args()

    if args.check_only:
        check_local_files()
    else:
        success = run_ingestion(container=args.container, force_azure=args.force_azure)
        exit(0 if success else 1)