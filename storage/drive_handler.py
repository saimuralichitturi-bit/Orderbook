"""
Google Drive handler — supports both My Drive folders and Shared Drives.
Folder ID starting with 0A = Shared Drive (needs supportsAllDrives=True).
"""

import io
import json
import os
from pathlib import Path
from loguru import logger

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload
import pandas as pd

SCOPES = ["https://www.googleapis.com/auth/drive"]


def _service(credentials_path: str = "credentials.json"):
    creds = service_account.Credentials.from_service_account_file(
        credentials_path, scopes=SCOPES
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


class DriveHandler:
    def __init__(self, folder_id: str = None, credentials_path: str = None):
        from config import GDRIVE_FOLDER_ID, GDRIVE_CREDENTIALS, IS_SHARED_DRIVE
        self.root_id      = (folder_id or GDRIVE_FOLDER_ID).strip()
        self.creds_path   = credentials_path or GDRIVE_CREDENTIALS
        self.shared_drive = IS_SHARED_DRIVE   # True when folder ID starts with 0A
        self.svc          = _service(self.creds_path)
        self._cache: dict[str, str] = {}
        logger.info(f"Drive mode: {'Shared Drive' if self.shared_drive else 'My Drive'} | root: {self.root_id}")

    # ── Shared Drive aware kwargs ──────────────────────────────────

    def _list_kwargs(self) -> dict:
        """Extra kwargs needed for Shared Drive file listing."""
        if self.shared_drive:
            return {
                "supportsAllDrives": True,
                "includeItemsFromAllDrives": True,
                "corpora": "drive",
                "driveId": self.root_id,
            }
        return {}

    def _op_kwargs(self) -> dict:
        """Extra kwargs for create/update/get on Shared Drive."""
        if self.shared_drive:
            return {"supportsAllDrives": True}
        return {}

    # ── Folder helpers ────────────────────────────────────────────

    def _get_or_create(self, name: str, parent: str = None) -> str:
        parent = parent or self.root_id
        key = f"{parent}/{name}"
        if key in self._cache:
            return self._cache[key]

        q = (
            f"name='{name}' and mimeType='application/vnd.google-apps.folder'"
            f" and '{parent}' in parents and trashed=false"
        )
        list_kw = self._list_kwargs()
        # For shared drive listing, corpora + driveId replaces the root query
        res = self.svc.files().list(
            q=q, fields="files(id,name)", **list_kw
        ).execute().get("files", [])

        if res:
            fid = res[0]["id"]
        else:
            meta = {
                "name": name,
                "mimeType": "application/vnd.google-apps.folder",
                "parents": [parent],
            }
            fid = self.svc.files().create(
                body=meta, fields="id", **self._op_kwargs()
            ).execute()["id"]
            logger.info(f"Created Drive folder: {name} under {parent}")

        self._cache[key] = fid
        return fid

    # ── Upload ────────────────────────────────────────────────────

    def _upsert(self, buf: io.BytesIO, filename: str, parent_id: str, mime: str) -> str:
        """Upload or overwrite file by name in parent folder."""
        q = f"name='{filename}' and '{parent_id}' in parents and trashed=false"
        list_kw = self._list_kwargs()
        existing = self.svc.files().list(
            q=q, fields="files(id)", **list_kw
        ).execute().get("files", [])

        buf.seek(0)
        media = MediaIoBaseUpload(buf, mimetype=mime, resumable=False)

        if existing:
            fid = existing[0]["id"]
            self.svc.files().update(
                fileId=fid, media_body=media, **self._op_kwargs()
            ).execute()
        else:
            meta = {"name": filename, "parents": [parent_id]}
            fid = self.svc.files().create(
                body=meta, media_body=media, fields="id", **self._op_kwargs()
            ).execute()["id"]
        return fid

    def upload_parquet(self, local_path: str | Path, subfolder: str = "parquet") -> str:
        path = Path(local_path)
        parent = self._get_or_create(subfolder)
        with open(path, "rb") as f:
            buf = io.BytesIO(f.read())
        fid = self._upsert(buf, path.name, parent, "application/octet-stream")
        logger.info(f"Uploaded: {path.name} → Drive/{subfolder}/")
        return fid

    def upload_pdf(self, local_path: str | Path, subfolder: str = "pdfs") -> str:
        path = Path(local_path)
        parent = self._get_or_create(subfolder)
        with open(path, "rb") as f:
            buf = io.BytesIO(f.read())
        fid = self._upsert(buf, path.name, parent, "application/pdf")
        logger.info(f"Uploaded PDF: {path.name}")
        return fid

    def upload_json(self, local_path: str | Path, subfolder: str = "analysis") -> str:
        path = Path(local_path)
        parent = self._get_or_create(subfolder)
        with open(path, "rb") as f:
            buf = io.BytesIO(f.read())
        fid = self._upsert(buf, path.name, parent, "application/json")
        return fid

    # ── Download ─────────────────────────────────────────────────

    def _download_fid(self, file_id: str) -> io.BytesIO:
        req = self.svc.files().get_media(fileId=file_id, **self._op_kwargs())
        buf = io.BytesIO()
        dl = MediaIoBaseDownload(buf, req)
        done = False
        while not done:
            _, done = dl.next_chunk()
        buf.seek(0)
        return buf

    def download_parquet(self, filename: str, subfolder: str = "parquet") -> pd.DataFrame:
        parent = self._get_or_create(subfolder)
        q = f"name='{filename}' and '{parent}' in parents and trashed=false"
        files = self.svc.files().list(
            q=q, fields="files(id,name)", **self._list_kwargs()
        ).execute().get("files", [])
        if not files:
            return pd.DataFrame()
        buf = self._download_fid(files[0]["id"])
        return pd.read_parquet(buf)

    def download_pdf_by_name(self, filename: str, subfolder: str = "pdfs") -> io.BytesIO | None:
        """Find and download a PDF from the Drive pdfs subfolder by filename. Returns BytesIO or None."""
        try:
            parent = self._get_or_create(subfolder)
            q = f"name='{filename}' and '{parent}' in parents and trashed=false"
            files = self.svc.files().list(
                q=q, fields="files(id,name)", **self._list_kwargs()
            ).execute().get("files", [])
            if not files:
                return None
            buf = self._download_fid(files[0]["id"])
            logger.info(f"Downloaded from Drive: {filename}")
            return buf
        except Exception as e:
            logger.warning(f"Drive PDF download failed ({filename}): {e}")
            return None

    # ── List files ────────────────────────────────────────────────

    def list_files(self, subfolder: str = "parquet") -> list[str]:
        """Return filenames available in a Drive subfolder."""
        try:
            parent = self._get_or_create(subfolder)
            q = (
                f"'{parent}' in parents and trashed=false "
                f"and mimeType!='application/vnd.google-apps.folder'"
            )
            results = []
            page_token = None
            while True:
                kwargs = dict(q=q, fields="nextPageToken,files(id,name)", pageSize=200)
                kwargs.update(self._list_kwargs())
                if page_token:
                    kwargs["pageToken"] = page_token
                resp = self.svc.files().list(**kwargs).execute()
                results.extend(f["name"] for f in resp.get("files", []))
                page_token = resp.get("nextPageToken")
                if not page_token:
                    break
            return results
        except Exception as e:
            logger.warning(f"Drive list_files failed ({subfolder}): {e}")
            return []

    # ── Sync FROM Drive → local ────────────────────────────────────

    def sync_parquets_from_drive(self, local_dir: "Path | None" = None) -> list[str]:
        """
        Download all parquet files from Drive that are missing locally.
        Returns list of filenames that were downloaded.
        Safe to call repeatedly — skips files already present locally.
        """
        from config import PARQUET_DIR
        target = local_dir or PARQUET_DIR
        target.mkdir(parents=True, exist_ok=True)

        remote_files = self.list_files("parquet")
        parquet_files = [f for f in remote_files if f.endswith(".parquet")]

        if not parquet_files:
            logger.info("No parquet files found on Drive.")
            return []

        downloaded = []
        for fname in parquet_files:
            local_path = target / fname
            if local_path.exists():
                logger.debug(f"Already local: {fname}")
                continue
            try:
                buf = self._download_fid(self._find_file_id(fname, "parquet"))
                if buf:
                    local_path.write_bytes(buf.read())
                    downloaded.append(fname)
                    logger.info(f"Pulled from Drive: {fname}")
            except Exception as e:
                logger.warning(f"Failed to download {fname} from Drive: {e}")

        if downloaded:
            logger.info(f"Pulled {len(downloaded)} parquets from Drive → local")
        return downloaded

    def _find_file_id(self, filename: str, subfolder: str) -> str | None:
        """Find a file's Drive ID by name within a subfolder."""
        parent = self._get_or_create(subfolder)
        q = f"name='{filename}' and '{parent}' in parents and trashed=false"
        files = self.svc.files().list(
            q=q, fields="files(id)", **self._list_kwargs()
        ).execute().get("files", [])
        return files[0]["id"] if files else None

    def sync_analysis_from_drive(self, local_dir: "Path | None" = None) -> list[str]:
        """Download analysis JSONs from Drive that are missing locally."""
        from config import ANALYSIS_DIR
        target = local_dir or ANALYSIS_DIR
        target.mkdir(parents=True, exist_ok=True)

        remote_files = self.list_files("analysis")
        json_files = [f for f in remote_files if f.endswith(".json")]

        downloaded = []
        for fname in json_files:
            local_path = target / fname
            if local_path.exists():
                continue
            try:
                fid = self._find_file_id(fname, "analysis")
                if fid:
                    buf = self._download_fid(fid)
                    local_path.write_bytes(buf.read())
                    downloaded.append(fname)
            except Exception as e:
                logger.warning(f"Failed to download analysis {fname}: {e}")

        if downloaded:
            logger.info(f"Pulled {len(downloaded)} analysis JSONs from Drive")
        return downloaded

    # ── Bulk sync ─────────────────────────────────────────────────

    def sync_all(self):
        """Push all local parquet + PDFs + analysis JSONs to Drive."""
        from config import PARQUET_DIR, PDF_DIR, ANALYSIS_DIR

        parquets = list(PARQUET_DIR.glob("*.parquet"))
        pdfs     = list(PDF_DIR.glob("*.pdf"))
        jsons    = list(ANALYSIS_DIR.glob("*.json"))

        logger.info(f"Syncing → Drive: {len(parquets)} parquets | {len(pdfs)} PDFs | {len(jsons)} JSONs")
        for p in parquets: self.upload_parquet(p)
        for p in pdfs:     self.upload_pdf(p)
        for p in jsons:    self.upload_json(p)
        logger.info("Drive sync complete")
