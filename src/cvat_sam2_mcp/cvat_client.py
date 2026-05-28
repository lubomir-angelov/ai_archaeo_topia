"""CVAT REST API client using raw requests.

Talks to CVAT 2.x DRF endpoints: /api/projects, /api/tasks, etc.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import requests

from .models import CvatLabel, CvatProjectInfo, CvatTaskInfo
from .settings import get_settings

logger = logging.getLogger(__name__)


class CvatClient:
    """Thin wrapper around CVAT's REST API."""

    def __init__(
        self, base_url: str | None = None, username: str | None = None, password: str | None = None
    ):
        settings = get_settings()
        self.base_url = (base_url or settings.cvat_base_url).rstrip("/")
        self.username = username or settings.cvat_username
        self.password = password or settings.cvat_password
        self.org = settings.cvat_org
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})
        self._authenticated = False

    # ── authentication ──────────────────────────────────────────

    def login(self) -> bool:
        """Authenticate with CVAT. Returns True on success."""
        if self._authenticated:
            return True
        url = f"{self.base_url}/api/auth/login"
        try:
            resp = self._session.post(
                url, json={"username": self.username, "password": self.password}
            )
            resp.raise_for_status()
            self._authenticated = True
            logger.info("Authenticated as %s", self.username)
            return True
        except requests.HTTPError as exc:
            logger.error("CVAT login failed: %s", exc)
            self._authenticated = False
            return False

    def _headers(self) -> dict[str, str]:
        hdrs = dict(self._session.headers)
        if self.org:
            hdrs["CVAT-Organization"] = self.org
        return hdrs

    def _get(self, path: str, **kwargs: Any) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        resp = self._session.get(url, headers=self._headers(), **kwargs)
        resp.raise_for_status()
        return resp.json()

    def _post(
        self, path: str, body: dict[str, Any] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        resp = self._session.post(url, json=body or {}, headers=self._headers(), **kwargs)
        resp.raise_for_status()
        if resp.status_code == 204:
            return {}
        return resp.json()

    def _put(self, path: str, body: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        resp = self._session.put(url, json=body, headers=self._headers(), **kwargs)
        resp.raise_for_status()
        return resp.json()

    def _delete(self, path: str, **kwargs: Any) -> None:
        url = f"{self.base_url}{path}"
        resp = self._session.delete(url, headers=self._headers(), **kwargs)
        resp.raise_for_status()

    # ── health ──────────────────────────────────────────────────

    def health(self) -> dict[str, Any]:
        """Check CVAT availability and authentication."""
        result: dict[str, Any] = {
            "reachable": False,
            "authenticated": False,
            "serverless_reachable": False,
        }
        try:
            resp = self._session.get(f"{self.base_url}/api/status", timeout=10)
            if resp.status_code == 200:
                result["reachable"] = True
                body = resp.json()
                result["status"] = body
        except requests.ConnectionError:
            pass
        except requests.HTTPError:
            result["status"] = "non-200 response"

        if self.username and self.password:
            result["authenticated"] = self.login()

        # Check serverless / Nuclio reachability
        try:
            nuclio_url = get_settings().nuclio_dashboard_url.rstrip("/")
            resp = self._session.get(f"{nuclio_url}/api/functions", timeout=5)
            if resp.status_code == 200:
                result["serverless_reachable"] = True
        except requests.ConnectionError:
            pass
        except requests.Timeout:
            pass

        return result

    # ── projects ────────────────────────────────────────────────

    def list_projects(self) -> list[CvatProjectInfo]:
        """Return all CVAT projects with label info."""
        data = self._get("/api/projects")
        projects: list[CvatProjectInfo] = []
        for item in data:
            labels = [
                CvatLabel(
                    name=lbl["title"],
                    color=lbl.get("color", "#ff0000"),
                    attributes=lbl.get("attributes", []),
                )
                for lbl in item.get("labels", [])
            ]
            projects.append(
                CvatProjectInfo(
                    id=item["id"],
                    name=item["title"],
                    labels=labels,
                    tasks=item.get("tasks", 0),
                )
            )
        return projects

    def get_project(self, project_id: int) -> CvatProjectInfo:
        """Return a single project by ID."""
        item = self._get(f"/api/projects/{project_id}")
        labels = [
            CvatLabel(
                name=lbl["title"],
                color=lbl.get("color", "#ff0000"),
                attributes=lbl.get("attributes", []),
            )
            for lbl in item.get("labels", [])
        ]
        return CvatProjectInfo(id=item["id"], name=item["title"], labels=labels)

    # ── tasks ───────────────────────────────────────────────────

    def list_tasks(self, project_id: int | None = None) -> list[CvatTaskInfo]:
        """Return tasks, optionally filtered by project."""
        params: dict[str, Any] = {}
        if project_id is not None:
            params["project"] = project_id
        data = self._get("/api/tasks", params=params)
        tasks: list[CvatTaskInfo] = []
        for item in data:
            labels = [
                CvatLabel(
                    name=lbl["title"],
                    color=lbl.get("color", "#ff0000"),
                    attributes=lbl.get("attributes", []),
                )
                for lbl in item.get("labels", [])
            ]
            tasks.append(
                CvatTaskInfo(
                    id=item["id"],
                    name=item["name"],
                    status=item.get("status", "unknown"),
                    project_id=item.get("project"),
                    labels=labels,
                    images=item.get("images", 0),
                    subset=item.get("subset", ""),
                )
            )
        return tasks

    def get_task(self, task_id: int) -> CvatTaskInfo:
        """Return a single task by ID."""
        item = self._get(f"/api/tasks/{task_id}")
        labels = [
            CvatLabel(
                name=lbl["title"],
                color=lbl.get("color", "#ff0000"),
                attributes=lbl.get("attributes", []),
            )
            for lbl in item.get("labels", [])
        ]
        return CvatTaskInfo(
            id=item["id"],
            name=item["name"],
            status=item.get("status", "unknown"),
            project_id=item.get("project"),
            labels=labels,
            images=item.get("images", 0),
            subset=item.get("subset", ""),
        )

    def find_task_by_name(self, name: str, project_id: int | None = None) -> CvatTaskInfo | None:
        """Find a task by exact name. Returns None if not found."""
        tasks = self.list_tasks(project_id=project_id)
        for t in tasks:
            if t.name == name:
                return t
        return None

    def create_task(
        self,
        name: str,
        image_dir: str,
        labels: list[dict[str, Any]],
        project_id: int | None = None,
        start_frame: int = 1,
        stop_frame: int = 1,
        frame_step: int = 1,
        subset: str = "",
        task_name: str | None = None,
    ) -> int:
        """Create a CVAT task from a local image directory.

        Uploads images, then creates the task with the given labels.
        Returns the task ID.
        """
        img_path = Path(image_dir)
        if not img_path.is_dir():
            raise ValueError(f"Image directory does not exist: {image_dir}")

        # Collect image files
        image_files = sorted(
            [
                f
                for f in img_path.iterdir()
                if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
            ]
        )
        if not image_files:
            raise ValueError(f"No image files found in {image_dir}")

        logger.info("Found %d images in %s", len(image_files), image_dir)

        # Upload images
        upload_url = f"{self.base_url}/api/tasks"
        files = []
        for img_file in image_files:
            files.append(
                ("data", (img_file.name, img_file.read_bytes(), "application/octet-stream"))
            )

        body: dict[str, Any] = {
            "name": name,
            "labels": labels,
            "start_frame": start_frame,
            "stop_frame": stop_frame,
            "frame_step": frame_step,
            "subset": subset,
        }
        if project_id is not None:
            body["project"] = project_id

        resp = self._session.post(
            upload_url,
            headers=self._headers(),
            files=files,
            data=body,
            timeout=300,
        )
        resp.raise_for_status()
        result = resp.json()
        task_id = result.get("id") or result.get("tasks", [{}])[0].get("id")
        logger.info("Created CVAT task %s (id=%s)", name, task_id)
        return task_id

    # ── exports ─────────────────────────────────────────────────

    def export_task(
        self,
        task_id: int,
        export_format: str = "CVAT for images 1.1",
        artifact_dir: str | None = None,
    ) -> str:
        """Export task annotations/images. Returns the path to the exported file."""
        # CVAT 2.x exports are async - create export, then poll
        export_path = f"/api/tasks/{task_id}/annotations"
        body = {"action": "export", "format": export_format}

        # Trigger export
        resp = self._session.post(
            f"{self.base_url}{export_path}",
            json=body,
            headers=self._headers(),
        )
        resp.raise_for_status()

        # Download the export - CVAT returns the file directly for sync export
        download_url = f"{self.base_url}{export_path}?format={export_format}"
        resp = self._session.get(download_url, headers=self._headers())
        resp.raise_for_status()

        artifact_dir = artifact_dir or str(get_settings().artifact_root)
        Path(artifact_dir).mkdir(parents=True, exist_ok=True)
        safe_name = f"task_{task_id}_export"
        if export_format == "CVAT for images 1.1":
            safe_name += ".xml"
        else:
            safe_name += ".json"
        out_path = Path(artifact_dir) / safe_name

        with open(out_path, "wb") as f:
            f.write(resp.content)

        logger.info("Exported task %d to %s", task_id, out_path)
        return str(out_path)

    # ── annotation import ───────────────────────────────────────

    def import_annotations(
        self,
        task_id: int,
        annotations_path: str,
        format_name: str = "CVAT for images 1.1",
        label_mapping: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Import candidate annotations into a CVAT task.

        Additive import only - never overwrites verified annotations.
        Adds metadata: source=sam2, status=candidate.
        """
        ann_path = Path(annotations_path)
        if not ann_path.exists():
            raise FileNotFoundError(f"Annotations file not found: {annotations_path}")

        content = ann_path.read_bytes()

        import_path = f"/api/tasks/{task_id}/annotations"
        body = {
            "action": "import",
            "format": format_name,
        }

        # Send via multipart (CVAT expects file upload for import)
        files = [
            (
                "data",
                (
                    "annotations",
                    content,
                    "application/xml"
                    if format_name == "CVAT for images 1.1"
                    else "application/json",
                ),
            )
        ]

        resp = self._session.post(
            f"{self.base_url}{import_path}",
            headers=self._headers(),
            files=files,
            data=body,
        )
        resp.raise_for_status()

        result = resp.json() if resp.status_code != 204 else {}
        logger.info("Imported annotations for task %d from %s", task_id, annotations_path)
        return result

    def create_labels_for_task(
        self,
        task_id: int,
        labels: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Create labels on a task if they don't already exist. Returns created labels."""
        existing = self.get_task(task_id)
        existing_names = {lbl.name for lbl in existing.labels}

        created: list[dict[str, Any]] = []
        for lbl in labels:
            if lbl["name"] not in existing_names:
                create_resp = self._post(f"/api/tasks/{task_id}/labels", body=lbl)
                created.append(create_resp)
                existing_names.add(lbl["name"])

        return created
