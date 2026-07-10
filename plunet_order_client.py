"""
Self-contained Plunet SOAP client for order file downloads.

Provides authenticated access to Plunet orders via three SOAP services:
- PlunetAPI — authentication
- DataOrder30 — order lookup and file listing
- DataDocument30 — file download

"""

import sys
import time
from pathlib import Path
from typing import Callable, Optional

import requests
from requests.adapters import HTTPAdapter

try:
    from urllib3.util.retry import Retry
except Exception:
    Retry = None

from zeep import Client, Settings
from zeep.transports import Transport
from zeep.helpers import serialize_object


class ProgressTransport(Transport):
    """HTTP transport that reports bytes read while Plunet builds the SOAP response."""

    def __init__(self, *args, on_bytes: Optional[Callable[[int, Optional[int]], None]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_bytes = on_bytes

    def post(self, address, message, headers):
        response = self.session.post(
            address, data=message, headers=headers,
            timeout=self.operation_timeout, stream=True,
        )
        total = response.headers.get("Content-Length")
        try:
            total = int(total) if total is not None else None
        except (TypeError, ValueError):
            total = None

        chunks = []
        received = 0
        for chunk in response.iter_content(chunk_size=256 * 1024):
            if not chunk:
                continue
            chunks.append(chunk)
            received += len(chunk)
            if self.on_bytes:
                self.on_bytes(received, total)

        response._content = b"".join(chunks)
        response.close()
        return response


def _invalid_session(status: dict) -> bool:
    """Check if a response indicates an expired/invalid session."""
    if not isinstance(status, dict):
        return False
    code = status.get("statusCode")
    alpha = status.get("statusCodeAlphanumeric", "")
    msg = str(status.get("statusMessage", ""))
    return code == -5 or alpha == "GEN_2" or "session-UUID" in msg


def _retryable_text(e: Exception) -> bool:
    """Check if an exception message suggests a retryable network error."""
    t = str(e)
    return any(k in t for k in (
        "RemoteDisconnected", "Connection aborted", "Read timed out",
        "timed out", "Broken pipe", "ProtocolError",
    ))


# Folder type constants for DataDocument30 service
ORDER_FOLDER_TYPES = {
    "source": 6,
    "reference": 2,
    "final": 3,
}


class PlunetOrderClient:
    """
    Plunet client focused on order file operations.

    Usage:
        client = PlunetOrderClient(base_url, username, password)
        client.login()
        order_id = client.order_name_to_id("O-30600")
        files = client.get_source_files(order_id)
        filename, content = client.download_file(order_id, files[0])
    """

    def __init__(self, base_url: str, username: str, password: str,
                 progress_callback: Optional[Callable[[int, Optional[int]], None]] = None):
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self._token = None
        self._clients = {}
        self._progress_callback = progress_callback
        self._build_clients()

    def set_progress_callback(self, callback: Optional[Callable[[int, Optional[int]], None]]):
        self._progress_callback = callback
        self._build_clients()

    def _make_session(self) -> requests.Session:
        session = requests.Session()
        if Retry is not None:
            retry_cfg = Retry(
                total=5, connect=5, read=5, backoff_factor=0.8,
                status_forcelist=[502, 503, 504, 520, 521, 522, 524],
                allowed_methods=frozenset(["GET", "POST"]),
                raise_on_status=False,
            )
            adapter = HTTPAdapter(
                max_retries=retry_cfg, pool_connections=10, pool_maxsize=20,
            )
            session.mount("https://", adapter)
            session.mount("http://", adapter)
        return session

    def _build_clients(self):
        """Initialize SOAP clients with retry-capable transport."""
        session = self._make_session()
        settings = Settings(strict=False, xml_huge_tree=True)

        transport_cls = Transport
        transport_kwargs = {"session": session, "timeout": 3600}
        if self._progress_callback:
            transport_cls = ProgressTransport
            transport_kwargs["on_bytes"] = self._progress_callback

        transport = transport_cls(**transport_kwargs)

        def mk(service_name):
            wsdl = f"{self.base_url}/{service_name}?wsdl"
            return Client(wsdl=wsdl, transport=transport, settings=settings)

        self._clients = {
            "api": mk("PlunetAPI"),
            "order": mk("DataOrder30"),
            "doc": mk("DataDocument30"),
        }

    def _call(self, label: str, fn, *args):
        """
        Call a SOAP method with retry on network errors and auto re-login
        on session expiry.

        Args:
            label: Human-readable label for error messages
            fn: SOAP service method to call
            *args: Arguments to pass (first arg should be the session token)

        Returns:
            Raw SOAP result (callers serialize and parse as needed)

        Raises:
            RuntimeError: After retries exhausted on network errors
        """
        last_exc = None
        for attempt in range(1, 5):
            try:
                r = fn(*args)
                s = serialize_object(r)
                # Auto re-login on session expiry
                if isinstance(s, dict) and _invalid_session(s):
                    if args and isinstance(args[0], str):
                        new_tok = self.login()
                        return fn(new_tok, *args[1:])
                return r
            except Exception as e:
                last_exc = e
                if attempt < 4 and _retryable_text(e):
                    time.sleep(0.8 * attempt)
                    continue
                raise
        raise last_exc if last_exc else RuntimeError(f"{label} failed")

    # ---- Auth ----

    def login(self) -> str:
        """Authenticate and store the session token."""
        token = self._clients["api"].service.login(self.username, self.password)
        self._token = token
        return token

    def token(self) -> str:
        """Get the current token, logging in if necessary."""
        return self._token or self.login()

    # ---- Order lookup ----

    def order_name_to_id(self, name: str) -> int:
        """
        Convert an order display name (e.g., "O-30600") to its internal ID.

        Args:
            name: Order display name

        Returns:
            Internal order ID

        Raises:
            ValueError: If order not found
        """
        svc = self._clients["order"].service
        result = self._call("Get order ID", svc.getOrderID, self.token(), name)
        data = serialize_object(result)

        if isinstance(data, dict):
            if data.get("statusCodeAlphanumeric") == "GEN_1":
                return data.get("data", 0)
            msg = data.get("statusMessage", "Unknown error")
            raise ValueError(f"Order '{name}' not found: {msg}")
        return int(data) if data else 0

    def get_order_display_name(self, order_id: int) -> str:
        """
        Get the display name for an order ID.

        Args:
            order_id: Internal order ID

        Returns:
            Display name string (e.g., "O-30600")
        """
        svc = self._clients["order"].service
        result = self._call("Get order display name", svc.getOrderNo_for_View, self.token(), order_id)
        data = serialize_object(result)
        if isinstance(data, dict):
            return data.get("data", f"Order-{order_id}")
        return f"Order-{order_id}"

    # ---- File listing ----

    def get_source_files(self, order_id: int) -> list:
        """
        Get list of source file paths from an order.

        Args:
            order_id: Internal order ID

        Returns:
            List of file path strings (backslash-separated Plunet paths)

        Raises:
            RuntimeError: If the API call fails
        """
        svc = self._clients["order"].service
        result = self._call("Get source files", svc.getDocuments_Within_SourceFolder, self.token(), order_id)
        data = serialize_object(result)
        if isinstance(data, dict) and data.get("data"):
            return list(data["data"])
        return []

    # ---- File download ----

    @staticmethod
    def _transform_path(file_path: str) -> str:
        """
        Transform path from Order service format to Document service format.

        Order service returns: \\source\\en-ca\\filename.docx
        Document service needs: \\en-ca\\filename.docx (strip folder prefix)
        """
        path = file_path.replace("\\", "/").lstrip("/")
        for prefix in ("source/", "reference/", "final/", "out/"):
            if path.lower().startswith(prefix):
                path = path[len(prefix):]
                break
        return "\\" + path.replace("/", "\\")

    def download_file(self, order_id: int, file_path: str,
                      folder_type: str = "source") -> tuple:
        """
        Download a single file from a Plunet order.

        Args:
            order_id: Internal order ID
            file_path: File path as returned by get_source_files()
            folder_type: One of "source", "reference", "final"

        Returns:
            (filename, file_bytes) tuple

        Raises:
            RuntimeError: If download fails
        """
        svc = self._clients["doc"].service
        folder_code = ORDER_FOLDER_TYPES.get(folder_type, 6)
        download_path = self._transform_path(file_path)

        result = self._call(
            "Download file", svc.download_Document,
            self.token(), order_id, folder_code, download_path,
        )
        data = serialize_object(result)

        if isinstance(data, dict) and data.get("statusCodeAlphanumeric") == "GEN_1":
            file_content = data.get("fileContent")
            if file_content:
                filename = Path(file_path.replace("\\", "/")).name
                return (filename, bytes(file_content))

        raise RuntimeError(
            f"Failed to download '{file_path}' from order {order_id}"
        )
