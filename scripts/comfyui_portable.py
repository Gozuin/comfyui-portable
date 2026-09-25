#!/usr/bin/env python3
"""Discover, validate, and run ComfyUI workflows through a local profile."""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import mimetypes
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2
RESULT_SCHEMA_VERSION = 1
SKILL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = SKILL_ROOT / "config.local.json"
DEFAULT_SERVER = "http://127.0.0.1:8188"

EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_UNREACHABLE = 3
EXIT_EXECUTION = 4
EXIT_UNKNOWN = 5
EXIT_INTERRUPTED = 130

PARAMETER_NAMES = {
    "prompt",
    "negative",
    "seed",
    "steps",
    "cfg",
    "denoise",
    "sampler_name",
    "scheduler",
    "width",
    "height",
    "batch_size",
    "resolution",
}

REFERENCE_ROLE = "reference"
SUPPORTED_BINDING_ROLES = PARAMETER_NAMES | {REFERENCE_ROLE}

MODEL_LOADER_ROLES: dict[str, dict[str, str]] = {
    "CheckpointLoaderSimple": {"ckpt_name": "checkpoint"},
    "CheckpointLoader": {"ckpt_name": "checkpoint"},
    "UNETLoader": {"unet_name": "unet"},
    "CLIPLoader": {
        "clip_name": "clip",
        "clip_name1": "clip_1",
        "clip_name2": "clip_2",
        "clip_name3": "clip_3",
    },
    "DualCLIPLoader": {
        "clip_name1": "clip_1",
        "clip_name2": "clip_2",
    },
    "TripleCLIPLoader": {
        "clip_name1": "clip_1",
        "clip_name2": "clip_2",
        "clip_name3": "clip_3",
    },
    "VAELoader": {"vae_name": "vae"},
    "LoraLoader": {"lora_name": "lora"},
    "LoraLoaderModelOnly": {"lora_name": "lora"},
    "ControlNetLoader": {"control_net_name": "controlnet"},
    "DiffControlNetLoader": {"control_net_name": "controlnet"},
    "StyleModelLoader": {"style_model_name": "style_model"},
    "CLIPVisionLoader": {"clip_vision_name": "clip_vision"},
    "IPAdapterModelLoader": {"ipadapter_file": "ipadapter"},
    "UpscaleModelLoader": {"model_name": "upscale_model"},
}

POSITIVE_TEXT_INPUTS = (
    "prompt",
    "positive_prompt",
    "positive",
    "text",
)

NEGATIVE_TEXT_INPUTS = (
    "negative_prompt",
    "negative",
    "text",
)

ROLE_FALLBACK_TYPES = {
    "prompt": "string",
    "negative": "string",
    "seed": "integer",
    "steps": "integer",
    "cfg": "number",
    "denoise": "number",
    "sampler_name": "string",
    "scheduler": "string",
    "width": "integer",
    "height": "integer",
    "batch_size": "integer",
    "resolution": "integer",
}


class ToolError(RuntimeError):
    """A user-facing command error with a stable machine contract."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "ERROR",
        state: str = "failed",
        exit_code: int = EXIT_CONFIG,
        details: dict[str, Any] | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.state = state
        self.exit_code = exit_code
        self.details = details or {}
        self.retryable = retryable

    def as_error(self) -> dict[str, Any]:
        error: dict[str, Any] = {
            "code": self.code,
            "message": str(self),
            "retryable": self.retryable,
        }
        error.update(self.details)
        return error


class UnreachableError(ToolError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(
            message,
            code="SERVER_UNREACHABLE",
            state="unreachable",
            exit_code=EXIT_UNREACHABLE,
            details=details,
        )


class TransportError(UnreachableError):
    """A connection failed after the request was attempted."""


class ExecutionError(ToolError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "EXECUTION_FAILED",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            code=code,
            state="failed",
            exit_code=EXIT_EXECUTION,
            details=details,
        )


class UnknownStateError(ToolError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "STATUS_UNKNOWN",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            code=code,
            state="unknown",
            exit_code=EXIT_UNKNOWN,
            details=details,
            retryable=False,
        )


def eprint(*parts: Any) -> None:
    print(*parts, file=sys.stderr)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ToolError(f"JSON file not found: {path}", code="FILE_NOT_FOUND") from exc
    except json.JSONDecodeError as exc:
        raise ToolError(
            f"Invalid JSON in {path}: {exc}",
            code="INVALID_JSON",
        ) from exc
    except OSError as exc:
        raise ToolError(
            f"Cannot read {path}: {exc}",
            code="READ_FAILED",
        ) from exc


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except OSError as exc:
        with contextlib.suppress(OSError):
            temporary.unlink()
        raise ToolError(
            f"Cannot write {path}: {exc}",
            code="WRITE_FAILED",
        ) from exc


def config_path(value: str | None) -> Path:
    configured = value or os.environ.get("COMFYUI_SKILL_CONFIG")
    if configured:
        return Path(configured).expanduser().resolve()
    return DEFAULT_CONFIG_PATH


def load_config(path: Path, required: bool = True) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise ToolError(
                f"Local profile not found: {path}",
                code="PROFILE_NOT_FOUND",
                details={"profile": str(path)},
            )
        return {}
    value = read_json(path)
    if not isinstance(value, dict):
        raise ToolError(
            f"Local profile must contain a JSON object: {path}",
            code="INVALID_PROFILE",
        )
    version = value.get("schema_version")
    if version == 1:
        migrated = copy.deepcopy(value)
        migrated["schema_version"] = SCHEMA_VERSION
        migrated["_legacy_schema"] = True
        migrated.setdefault("validation_state", "draft")
        return migrated
    if version != SCHEMA_VERSION:
        raise ToolError(
            f"Unsupported schema_version in {path}: {version!r}; "
            f"expected {SCHEMA_VERSION}",
            code="PROFILE_SCHEMA_UNSUPPORTED",
        )
    return value


def resolve_stored_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (SKILL_ROOT / path).resolve()


def store_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(SKILL_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def server_url(config: dict[str, Any]) -> str:
    return str(config.get("server") or DEFAULT_SERVER).rstrip("/")


def merged_headers(
    config: dict[str, Any],
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    headers = {
        str(key): str(value) for key, value in (config.get("headers") or {}).items()
    }
    api_key = os.environ.get("COMFYUI_API_KEY")
    if api_key and not any(key.lower() == "authorization" for key in headers):
        headers["Authorization"] = f"Bearer {api_key}"
    if extra:
        headers.update(extra)
    return headers


def server_security_warnings(config: dict[str, Any]) -> list[str]:
    url = server_url(config)
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "http" and parsed.hostname not in {
        None,
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        return [
            (
                "The configured server uses plain HTTP on a non-loopback host; "
                "credentials and job data may be visible on the network."
            )
        ]
    return []


def http_request(
    url: str,
    *,
    method: str = "GET",
    payload: Any = None,
    headers: dict[str, str] | None = None,
    timeout: float = 30,
    raw: bool = False,
) -> Any:
    request_headers = dict(headers or {})
    data: bytes | None = None
    if payload is not None:
        if isinstance(payload, (bytes, bytearray)):
            data = bytes(payload)
        else:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            request_headers.setdefault("Content-Type", "application/json")

    request = urllib.request.Request(
        url,
        data=data,
        headers=request_headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ToolError(
            f"HTTP {exc.code} from {urllib.parse.urlparse(url).path}: {detail[:2000]}",
            code="HTTP_ERROR",
            details={"http_status": exc.code, "endpoint": url},
        ) from exc
    except urllib.error.URLError as exc:
        raise TransportError(
            f"Cannot reach {url}: {exc.reason}",
            endpoint=url,
        ) from exc
    except (TimeoutError, OSError) as exc:
        raise TransportError(
            f"Connection to {url} failed: {exc}", endpoint=url
        ) from exc

    if raw:
        return body
    if not body.strip():
        return {}
    try:
        return json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ToolError(
            f"Non-JSON response from {urllib.parse.urlparse(url).path}: {body[:500]!r}",
            code="INVALID_SERVER_RESPONSE",
        ) from exc


def get_system_stats(config: dict[str, Any]) -> dict[str, Any]:
    data = http_request(
        server_url(config) + "/system_stats",
        headers=merged_headers(config),
        timeout=float(config.get("timeout_seconds", 30)),
    )
    if not isinstance(data, dict):
        raise ToolError(
            "/system_stats did not return a JSON object",
            code="INVALID_SERVER_RESPONSE",
        )
    return data


def get_object_info(config: dict[str, Any]) -> dict[str, Any]:
    data = http_request(
        server_url(config) + "/object_info",
        headers=merged_headers(config),
        timeout=float(config.get("timeout_seconds", 30)),
    )
    if not isinstance(data, dict):
        raise ToolError(
            "/object_info did not return a JSON object",
            code="INVALID_SERVER_RESPONSE",
        )
    return data


def option_values(spec: Any) -> list[str] | None:
    """Return enum options, preserving a valid empty enum as []."""
    if isinstance(spec, list) and spec:
        values = spec[0]
        if isinstance(values, list) and all(isinstance(item, str) for item in values):
            return list(values)
    if isinstance(spec, dict):
        values = spec.get("options")
        if isinstance(values, list) and all(isinstance(item, str) for item in values):
            return list(values)
    return None


def input_spec(
    object_info: dict[str, Any] | None,
    class_name: str,
    input_name: str,
) -> Any:
    if not object_info:
        return None
    details = object_info.get(class_name)
    if not isinstance(details, dict):
        return None
    groups = details.get("input") or {}
    if not isinstance(groups, dict):
        return None
    for group_name in ("required", "optional"):
        group = groups.get(group_name) or {}
        if isinstance(group, dict) and input_name in group:
            return group[input_name]
    return None


def enumerations_from_object_info(
    object_info: dict[str, Any] | None,
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    if not object_info:
        return result
    for class_name, details in object_info.items():
        if not isinstance(details, dict):
            continue
        groups = details.get("input") or {}
        if not isinstance(groups, dict):
            continue
        for group_name in ("required", "optional"):
            group = groups.get(group_name) or {}
            if not isinstance(group, dict):
                continue
            for input_name, spec in group.items():
                values = option_values(spec)
                if values is not None:
                    result[f"{class_name}.{input_name}"] = values
    return result


def model_role_for_input(class_name: str, input_name: str) -> str | None:
    return MODEL_LOADER_ROLES.get(class_name, {}).get(input_name)


def model_options_from_object_info(
    object_info: dict[str, Any] | None,
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    if not object_info:
        return result
    for class_name, mapping in MODEL_LOADER_ROLES.items():
        details = object_info.get(class_name)
        if not isinstance(details, dict):
            continue
        groups = details.get("input") or {}
        if not isinstance(groups, dict):
            continue
        for group_name in ("required", "optional"):
            group = groups.get(group_name) or {}
            if not isinstance(group, dict):
                continue
            for input_name in mapping:
                if input_name not in group:
                    continue
                values = option_values(group[input_name])
                if values is not None:
                    result[f"{class_name}.{input_name}"] = values
    return result


def is_comfy_root(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "main.py").is_file()
        and (
            (path / "folder_paths.py").is_file()
            or (path / "comfy").is_dir()
            or (path / "nodes.py").is_file()
        )
    )


def normalize_comfy_root(path: Path) -> Path | None:
    candidate = path.expanduser().resolve()
    if is_comfy_root(candidate):
        return candidate
    nested = candidate / "ComfyUI"
    if is_comfy_root(nested):
        return nested.resolve()
    return None


def _candidate_roots() -> Iterable[Path]:
    for variable in ("COMFYUI_ROOT", "COMFYUI_HOME"):
        if os.environ.get(variable):
            yield Path(os.environ[variable])

    current = Path.cwd().resolve()
    yield current
    for parent in list(current.parents)[:5]:
        yield parent
        yield parent / "ComfyUI"

    home = Path.home()
    yield home / "ComfyUI"
    yield home / "comfyui"
    yield home / "Documents" / "ComfyUI"
    yield home / "Downloads" / "ComfyUI"

    if os.name == "nt":
        for drive in ("C:", "D:", "E:", "F:"):
            yield Path(drive + "\\")
            yield Path(drive + "\\ComfyUI")
            yield Path(drive + "\\comfyui")
            yield Path(drive + "\\ComfyUI_windows_portable")

    for base in (home / "Documents", home / "Downloads"):
        if base.is_dir():
            try:
                yield from base.glob("*ComfyUI*")
            except OSError:
                pass


def discover_comfy_root(explicit: str | None = None) -> Path | None:
    if explicit:
        normalized = normalize_comfy_root(Path(explicit))
        if normalized:
            return normalized
        raise ToolError(
            f"Not a ComfyUI root: {explicit}",
            code="COMFY_ROOT_NOT_FOUND",
        )

    seen: set[Path] = set()
    for candidate in _candidate_roots():
        try:
            key = candidate.expanduser().resolve()
        except OSError:
            continue
        if key in seen:
            continue
        seen.add(key)
        normalized = normalize_comfy_root(key)
        if normalized:
            return normalized
    return None


def find_comfy_python(root: Path | None) -> Path | None:
    configured = os.environ.get("COMFYUI_PYTHON")
    if configured:
        configured_path = Path(configured).expanduser().resolve()
        if not configured_path.is_file():
            raise ToolError(
                f"COMFYUI_PYTHON does not point to a file: {configured_path}",
                code="PYTHON_NOT_FOUND",
            )
        return configured_path

    if root:
        for base in (root, root.parent, root.parent.parent):
            candidates = (
                base / "python_embeded" / "python.exe",
                base / "python_embeded" / "python",
                base / ".venv" / "Scripts" / "python.exe",
                base / ".venv" / "bin" / "python",
                base / "venv" / "Scripts" / "python.exe",
                base / "venv" / "bin" / "python",
            )
            for candidate in candidates:
                if candidate.is_file():
                    return candidate.resolve()
    executable = shutil.which("python") or shutil.which("python3")
    if executable:
        return Path(executable).resolve()
    return Path(sys.executable).resolve()


def load_graph(path: Path) -> dict[str, Any]:
    graph = read_json(path)
    if not isinstance(graph, dict) or not graph:
        raise ToolError(
            f"Workflow must be a non-empty API-format JSON object: {path}",
            code="INVALID_WORKFLOW",
        )
    for node_id, node in graph.items():
        if not isinstance(node, dict) or "class_type" not in node:
            raise ToolError(
                f"Node {node_id!r} in {path} is not API format "
                "(expected class_type and inputs)",
                code="INVALID_WORKFLOW",
            )
        if not isinstance(node.get("inputs"), dict):
            raise ToolError(
                f"Node {node_id!r} in {path} has no inputs object",
                code="INVALID_WORKFLOW",
            )
    return graph


def sorted_node_ids(graph: dict[str, Any]) -> list[str]:
    def key(value: str) -> tuple[int, int | str, str]:
        return (0, int(value), value) if value.isdigit() else (1, value, value)

    return sorted(graph, key=key)


def node_class(node: dict[str, Any]) -> str:
    return str(node.get("class_type") or "")


def link_target(graph: dict[str, Any], link: Any) -> tuple[str, int] | None:
    if (
        isinstance(link, list)
        and len(link) >= 2
        and str(link[0]) in graph
        and isinstance(link[1], int)
    ):
        return str(link[0]), link[1]
    return None


def is_link_value(value: Any) -> bool:
    return isinstance(value, list) and len(value) >= 2


def find_model_bindings(
    graph: dict[str, Any],
    object_info: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    used: set[str] = set()
    for node_id in sorted_node_ids(graph):
        node = graph[node_id]
        class_name = node_class(node)
        for input_name, value in (node.get("inputs") or {}).items():
            role = model_role_for_input(class_name, input_name)
            if not role or is_link_value(value):
                continue
            if input_spec(object_info, class_name, input_name) is None and object_info:
                continue
            key = role
            index = 2
            while key in used:
                key = f"{role}_{index}"
                index += 1
            used.add(key)
            result[key] = {
                "node": node_id,
                "input": input_name,
                "value": value,
            }
    return result


def is_sampler_node(class_name: str, inputs: dict[str, Any]) -> bool:
    lowered = class_name.lower()
    return "sampler" in lowered and any(
        key in inputs for key in ("seed", "noise_seed", "steps")
    )


def literal_input(
    inputs: dict[str, Any], candidates: Iterable[str]
) -> tuple[str, Any] | None:
    for key in candidates:
        if key in inputs and not is_link_value(inputs[key]):
            return key, inputs[key]
    return None


def text_binding_for(
    graph: dict[str, Any],
    target_id: str,
    *,
    negative: bool,
    same_node_as_other: bool = False,
) -> Any:
    inputs = graph[target_id].get("inputs") or {}
    class_name = node_class(graph[target_id])

    if negative and same_node_as_other:
        for key in ("negative_prompt", "negative"):
            if key in inputs:
                return {"node": target_id, "input": key}
        return None

    if "CLIPTextEncodeSDXL" in class_name and {
        "text_g",
        "text_l",
    }.issubset(inputs):
        return [
            {"node": target_id, "input": "text_g"},
            {"node": target_id, "input": "text_l"},
        ]

    candidates = NEGATIVE_TEXT_INPUTS if negative else POSITIVE_TEXT_INPUTS
    for key in candidates:
        if key in inputs and not is_link_value(inputs[key]):
            return {"node": target_id, "input": key}
    return None


def inspect_workflow(
    graph: dict[str, Any],
    object_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bindings: dict[str, Any] = {}
    defaults: dict[str, Any] = {}
    ambiguities: list[dict[str, Any]] = []

    sampler_ids = [
        node_id
        for node_id in sorted_node_ids(graph)
        if is_sampler_node(
            node_class(graph[node_id]),
            graph[node_id].get("inputs") or {},
        )
    ]
    if len(sampler_ids) > 1:
        ambiguities.append(
            {
                "code": "MULTIPLE_SAMPLERS",
                "message": (
                    "Multiple sampler nodes were found; add explicit descriptor "
                    "bindings before running this workflow."
                ),
                "candidates": sampler_ids,
            }
        )
    elif sampler_ids:
        sampler_id = sampler_ids[0]
        sampler_inputs = graph[sampler_id].get("inputs") or {}
        for role, candidates in {
            "seed": ("seed", "noise_seed"),
            "steps": ("steps",),
            "cfg": ("cfg",),
            "denoise": ("denoise",),
            "sampler_name": ("sampler_name",),
            "scheduler": ("scheduler",),
        }.items():
            found = literal_input(sampler_inputs, candidates)
            if found:
                key, value = found
                bindings[role] = {"node": sampler_id, "input": key}
                if role != "seed":
                    defaults[role] = value

        positive_target = link_target(graph, sampler_inputs.get("positive"))
        negative_target = link_target(graph, sampler_inputs.get("negative"))
        if positive_target:
            positive_id = positive_target[0]
            same_node = bool(negative_target and negative_target[0] == positive_id)
            binding = text_binding_for(
                graph,
                positive_id,
                negative=False,
                same_node_as_other=same_node,
            )
            if binding:
                bindings["prompt"] = binding
        if negative_target:
            negative_id = negative_target[0]
            binding = text_binding_for(
                graph,
                negative_id,
                negative=True,
                same_node_as_other=bool(
                    positive_target and positive_target[0] == negative_id
                ),
            )
            if binding:
                bindings["negative"] = binding

    for node_id in sorted_node_ids(graph):
        node = graph[node_id]
        inputs = node.get("inputs") or {}
        class_name = node_class(node)
        if (
            "width" in inputs
            and "height" in inputs
            and ("latent" in class_name.lower() or class_name.startswith("Empty"))
            and "sampler" not in class_name.lower()
        ):
            bindings["width"] = {"node": node_id, "input": "width"}
            bindings["height"] = {"node": node_id, "input": "height"}
            if not is_link_value(inputs["width"]):
                defaults["width"] = inputs["width"]
            if not is_link_value(inputs["height"]):
                defaults["height"] = inputs["height"]
            if "batch_size" in inputs:
                bindings["batch_size"] = {"node": node_id, "input": "batch_size"}
                if not is_link_value(inputs["batch_size"]):
                    defaults["batch_size"] = inputs["batch_size"]
            break

    if "resolution" not in bindings:
        for node_id in sorted_node_ids(graph):
            inputs = graph[node_id].get("inputs") or {}
            found = literal_input(inputs, ("resolution",))
            if found:
                key, value = found
                bindings["resolution"] = {"node": node_id, "input": key}
                defaults["resolution"] = value
                break

    references = []
    for node_id in sorted_node_ids(graph):
        node = graph[node_id]
        class_name = node_class(node)
        inputs = node.get("inputs") or {}
        if class_name == "LoadImage" and "image" in inputs:
            references.append({"node": node_id, "input": "image"})
        elif class_name in {"LoadVideo", "VHS_LoadVideo"} and "video" in inputs:
            references.append({"node": node_id, "input": "video"})
    if references:
        bindings["reference"] = references

    required_nodes = sorted(
        {node_class(node) for node in graph.values() if node_class(node)}
    )
    return {
        "bindings": bindings,
        "models": find_model_bindings(graph, object_info),
        "defaults": defaults,
        "required_nodes": required_nodes,
        "ambiguities": ambiguities,
    }


def validate_descriptor(descriptor: Any, path: Path | None = None) -> dict[str, Any]:
    label = str(path) if path else "descriptor"
    if not isinstance(descriptor, dict):
        raise ToolError(
            f"Descriptor must be a JSON object: {label}",
            code="INVALID_DESCRIPTOR",
        )

    bindings = descriptor.get("bindings") or {}
    if not isinstance(bindings, dict):
        raise ToolError(
            f"{label}.bindings must be an object",
            code="INVALID_DESCRIPTOR",
        )
    unknown_roles = sorted(set(bindings) - SUPPORTED_BINDING_ROLES)
    if unknown_roles:
        raise ToolError(
            f"{label} contains unsupported binding roles: {', '.join(unknown_roles)}",
            code="UNKNOWN_BINDING_ROLE",
        )

    models = descriptor.get("models") or {}
    if not isinstance(models, dict):
        raise ToolError(
            f"{label}.models must be an object",
            code="INVALID_DESCRIPTOR",
        )
    for role, binding in models.items():
        if not isinstance(binding, dict):
            raise ToolError(
                f"{label}.models.{role} must be an object",
                code="INVALID_MODEL_BINDING",
            )
        if not isinstance(binding.get("node"), (str, int)):
            raise ToolError(
                f"{label}.models.{role}.node must be a node ID",
                code="INVALID_MODEL_BINDING",
            )
        if not isinstance(binding.get("input"), str) or not binding["input"]:
            raise ToolError(
                f"{label}.models.{role}.input must be a non-empty string",
                code="INVALID_MODEL_BINDING",
            )

    defaults = descriptor.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise ToolError(
            f"{label}.defaults must be an object",
            code="INVALID_DESCRIPTOR",
        )
    unknown_defaults = sorted(set(defaults) - PARAMETER_NAMES)
    if unknown_defaults:
        raise ToolError(
            f"{label} contains unsupported defaults: {', '.join(unknown_defaults)}",
            code="UNKNOWN_DEFAULT_ROLE",
        )

    required_nodes = descriptor.get("required_nodes") or []
    if not isinstance(required_nodes, list) or not all(
        isinstance(item, str) and item for item in required_nodes
    ):
        raise ToolError(
            f"{label}.required_nodes must be a list of class names",
            code="INVALID_DESCRIPTOR",
        )

    params = descriptor.get("params") or {}
    if not isinstance(params, dict):
        raise ToolError(
            f"{label}.params must be an object",
            code="INVALID_DESCRIPTOR",
        )
    unknown_params = sorted(set(params) - PARAMETER_NAMES)
    if unknown_params:
        raise ToolError(
            f"{label} contains unsupported params: {', '.join(unknown_params)}",
            code="UNKNOWN_PARAMETER_ROLE",
        )
    allowed_types = {"integer", "number", "boolean", "string"}
    for role, spec in params.items():
        if not isinstance(spec, dict):
            raise ToolError(
                f"{label}.params.{role} must be an object",
                code="INVALID_DESCRIPTOR",
            )
        value_type = spec.get("type")
        if value_type not in allowed_types:
            raise ToolError(
                f"{label}.params.{role}.type must be one of: "
                f"{', '.join(sorted(allowed_types))}",
                code="INVALID_PARAMETER_TYPE",
            )
        enum = spec.get("enum")
        if enum is not None and (
            not isinstance(enum, list)
            or not enum
            or not all(isinstance(item, str) for item in enum)
        ):
            raise ToolError(
                f"{label}.params.{role}.enum must be a non-empty list of strings",
                code="INVALID_PARAMETER_ENUM",
            )
    return descriptor


def merge_descriptor(
    automatic: dict[str, Any],
    descriptor: dict[str, Any] | None,
    *,
    descriptor_path: Path | None = None,
) -> dict[str, Any]:
    merged = copy.deepcopy(automatic)
    if not descriptor:
        return merged
    descriptor = validate_descriptor(descriptor, descriptor_path)

    for role, binding in (descriptor.get("bindings") or {}).items():
        merged["bindings"][role] = copy.deepcopy(binding)
    for role, binding in (descriptor.get("models") or {}).items():
        model_binding = dict(binding)
        model_binding.pop("required", None)
        merged["models"][role] = model_binding
    merged["defaults"].update(descriptor.get("defaults") or {})
    merged["required_nodes"] = sorted(
        set(merged.get("required_nodes") or [])
        | set(descriptor.get("required_nodes") or [])
    )
    merged["params"] = copy.deepcopy(descriptor.get("params") or {})
    if descriptor.get("task_type"):
        merged["task_type"] = str(descriptor["task_type"])
    return merged


def binding_items(binding: Any, label: str = "binding") -> list[dict[str, Any]]:
    if isinstance(binding, dict):
        items = [binding]
    elif isinstance(binding, list) and binding:
        items = binding
    else:
        raise ToolError(
            f"{label} must be an object or a non-empty list of objects",
            code="INVALID_BINDING",
        )
    if not all(isinstance(item, dict) for item in items):
        raise ToolError(
            f"{label} contains a non-object binding",
            code="INVALID_BINDING",
        )
    return items


def validate_binding(
    graph: dict[str, Any],
    binding: Any,
    label: str,
) -> list[dict[str, Any]]:
    items = binding_items(binding, label)
    seen: set[tuple[str, str]] = set()
    for item in items:
        raw_node_id = item.get("node")
        input_name = item.get("input")
        if not isinstance(raw_node_id, (str, int)) or str(raw_node_id) == "":
            raise ToolError(
                f"{label} has no valid node ID",
                code="INVALID_BINDING",
            )
        node_id = str(raw_node_id)
        if node_id not in graph:
            raise ToolError(
                f"{label} points to missing node {node_id!r}",
                code="BINDING_NODE_MISSING",
                details={"binding": label, "node": node_id},
            )
        if (
            not isinstance(input_name, str)
            or input_name not in graph[node_id]["inputs"]
        ):
            raise ToolError(
                f"{label} points to missing input {input_name!r} on node {node_id}",
                code="BINDING_INPUT_MISSING",
                details={"binding": label, "node": node_id, "input": input_name},
            )
        key = (node_id, input_name)
        if key in seen:
            raise ToolError(
                f"{label} binds the same input more than once: {node_id}.{input_name}",
                code="DUPLICATE_BINDING",
            )
        seen.add(key)
    return [{"node": str(item["node"]), "input": str(item["input"])} for item in items]


def set_node_input(
    graph: dict[str, Any],
    binding: dict[str, Any],
    value: Any,
) -> None:
    node_id = str(binding["node"])
    input_name = str(binding["input"])
    if node_id not in graph or input_name not in graph[node_id]["inputs"]:
        raise ToolError(
            f"Binding points to missing {node_id}.{input_name}",
            code="BINDING_INVALID",
        )
    graph[node_id]["inputs"][input_name] = value


def parse_cli_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def parse_key_values(values: list[str], label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ToolError(
                f"{label} must use NAME=VALUE: {value!r}",
                code="INVALID_ARGUMENT",
            )
        key, item = value.split("=", 1)
        if not key or not item:
            raise ToolError(
                f"{label} must use NAME=VALUE: {value!r}",
                code="INVALID_ARGUMENT",
            )
        result[key] = item
    return result


def parse_scoped_values(
    values: list[str],
    label: str,
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for value in values:
        if "=" not in value or "." not in value.split("=", 1)[0]:
            raise ToolError(
                f"{label} must use WORKFLOW.ROLE=VALUE: {value!r}",
                code="INVALID_ARGUMENT",
            )
        scope, item = value.split("=", 1)
        workflow, key = scope.split(".", 1)
        if not workflow or not key or not item:
            raise ToolError(
                f"{label} must use WORKFLOW.ROLE=VALUE: {value!r}",
                code="INVALID_ARGUMENT",
            )
        result.setdefault(workflow, {})[key] = item
    return result


def parse_workflow_arguments(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" in value:
            name, raw_path = value.split("=", 1)
        else:
            raw_path = value
            name = Path(value).name
            if name.endswith(".api.json"):
                name = name[: -len(".api.json")]
            else:
                name = Path(name).stem
        if not name or not raw_path:
            raise ToolError(
                f"Workflow must use NAME=PATH: {value!r}",
                code="INVALID_ARGUMENT",
            )
        result[name] = Path(raw_path).expanduser().resolve()
    return result


def parse_descriptor_arguments(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ToolError(
                f"Descriptor must use NAME=PATH: {value!r}",
                code="INVALID_ARGUMENT",
            )
        name, raw_path = value.split("=", 1)
        if not name or not raw_path:
            raise ToolError(
                f"Descriptor must use NAME=PATH: {value!r}",
                code="INVALID_ARGUMENT",
            )
        result[name] = Path(raw_path).expanduser().resolve()
    return result


def default_descriptor_for(workflow_path: Path) -> Path | None:
    if workflow_path.name.endswith(".api.json"):
        candidate = workflow_path.with_name(
            workflow_path.name[: -len(".api.json")] + ".descriptor.json"
        )
    else:
        candidate = workflow_path.with_suffix(".descriptor.json")
    return candidate if candidate.is_file() else None


def prompt_model_choice(
    label: str,
    current: Any,
    options: list[str],
) -> str:
    print(f"\n{label}: {current!r} is not installed.", file=sys.stderr)
    for index, option in enumerate(options, start=1):
        print(f"  {index}. {option}", file=sys.stderr)
    while True:
        answer = input(
            "Select a model number, or press Enter to keep the value: "
        ).strip()
        if not answer:
            return str(current)
        try:
            selected = int(answer)
        except ValueError:
            print("Enter a number or press Enter.", file=sys.stderr)
            continue
        if 1 <= selected <= len(options):
            return options[selected - 1]
        print("Selection is out of range.", file=sys.stderr)


def workflows_to_configure(
    explicit_workflows: dict[str, Path],
    explicit_descriptors: dict[str, Path],
) -> list[tuple[str, Path, Path | None]]:
    if not explicit_workflows:
        for path in sorted((SKILL_ROOT / "workflows").glob("*.api.json")):
            name = path.name[: -len(".api.json")]
            explicit_workflows[name] = path.resolve()

    if not explicit_workflows:
        raise ToolError(
            "No workflows configured. Pass --workflow NAME=PATH or place "
            "*.api.json files in the workflows directory.",
            code="NO_WORKFLOWS",
        )

    result = []
    for name, path in explicit_workflows.items():
        if not path.is_file():
            raise ToolError(
                f"Workflow file not found: {path}",
                code="FILE_NOT_FOUND",
            )
        descriptor = explicit_descriptors.get(name) or default_descriptor_for(path)
        if name in explicit_descriptors and not explicit_descriptors[name].is_file():
            raise ToolError(
                f"Descriptor file not found: {explicit_descriptors[name]}",
                code="FILE_NOT_FOUND",
            )
        result.append((name, path, descriptor))
    return result


def schema_type(spec: Any, role: str | None = None) -> str:
    if option_values(spec) is not None:
        return "string"
    if isinstance(spec, list) and spec:
        raw = spec[0]
        if isinstance(raw, str):
            lowered = raw.lower()
            if lowered in {"int", "integer"}:
                return "integer"
            if lowered in {"float", "number"}:
                return "number"
            if lowered == "boolean":
                return "boolean"
            if lowered == "string":
                return "string"
    if role:
        return ROLE_FALLBACK_TYPES.get(role, "string")
    return "unknown"


def binding_specs(
    graph: dict[str, Any],
    bindings: dict[str, Any],
    object_info: dict[str, Any] | None,
    descriptor_params: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    descriptor_params = descriptor_params or {}
    for role, binding in bindings.items():
        if role not in PARAMETER_NAMES:
            continue
        explicit = descriptor_params.get(role)
        if explicit is not None:
            if not isinstance(explicit, dict):
                raise ToolError(
                    f"params.{role} must be an object",
                    code="INVALID_DESCRIPTOR",
                )
            result[role] = copy.deepcopy(explicit)
            continue
        items = validate_binding(graph, binding, role)
        specs = [
            input_spec(
                object_info,
                node_class(graph[item["node"]]),
                item["input"],
            )
            for item in items
        ]
        first = specs[0] if specs else None
        role_type = schema_type(first, role)
        values = option_values(first)
        entry: dict[str, Any] = {"type": role_type}
        if values is not None:
            entry["enum"] = values
        result[role] = entry
    return result


def derive_defaults(
    graph: dict[str, Any],
    bindings: dict[str, Any],
) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for role, binding in bindings.items():
        if role not in PARAMETER_NAMES:
            continue
        values: list[Any] = []
        for item in binding_items(binding, role):
            value = graph[str(item["node"])]["inputs"].get(str(item["input"]))
            if is_link_value(value):
                values = []
                break
            values.append(value)
        if values and all(value == values[0] for value in values):
            defaults[role] = values[0]
    return defaults


def coerce_parameter(
    role: str,
    value: Any,
    spec: dict[str, Any] | None = None,
) -> Any:
    if value is None:
        return None
    spec = spec or {}
    enum = spec.get("enum")
    if isinstance(enum, list):
        string_value = str(value)
        if string_value not in enum:
            raise ToolError(
                f"--{role.replace('_', '-')} must be one of: {', '.join(enum)}",
                code="PARAMETER_ENUM_INVALID",
                details={"binding": role, "value": value, "candidates": enum},
            )
        return string_value

    value_type = spec.get("type") or ROLE_FALLBACK_TYPES.get(role, "string")
    try:
        if value_type == "integer":
            if isinstance(value, bool):
                raise ValueError("boolean is not an integer")
            return int(value)
        if value_type == "number":
            if isinstance(value, bool):
                raise ValueError("boolean is not a number")
            return float(value)
        if value_type == "boolean":
            if isinstance(value, bool):
                return value
            lowered = str(value).lower()
            if lowered in {"true", "1", "yes", "on"}:
                return True
            if lowered in {"false", "0", "no", "off"}:
                return False
            raise ValueError("expected a boolean")
        if value_type == "string":
            return str(value)
        if value_type == "unknown":
            return value
    except (TypeError, ValueError) as exc:
        raise ToolError(
            f"Invalid value for {role}: {value!r} ({exc})",
            code="PARAMETER_TYPE_INVALID",
            details={"binding": role, "value": value, "expected": value_type},
        ) from exc
    return value


def convert_parameter(role: str, value: Any) -> Any:
    return coerce_parameter(role, value)


def parameter_values(
    definition: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], set[str]]:
    values: dict[str, Any] = {}
    explicit: set[str] = set()
    for role in PARAMETER_NAMES:
        value = getattr(args, role, None)
        if value is not None:
            values[role] = value
            explicit.add(role)
    for role, value in (definition.get("defaults") or {}).items():
        if role in PARAMETER_NAMES and role not in values:
            values[role] = value
    return values, explicit


def build_plan(
    graph: dict[str, Any],
    definition: dict[str, Any],
    args: argparse.Namespace,
    workflow_name: str,
    job_id: str,
) -> dict[str, Any]:
    planned_graph = copy.deepcopy(graph)
    warnings: list[str] = []
    bindings = definition.get("bindings") or {}
    models = definition.get("models") or {}

    for role, binding in models.items():
        items = validate_binding(
            planned_graph,
            binding,
            f"{workflow_name}.{role}",
        )
        for item in items:
            set_node_input(planned_graph, item, binding.get("value"))

    values, explicit = parameter_values(definition, args)
    specs = definition.get("params") or {}
    for role in sorted(values):
        if role not in bindings:
            if role in explicit:
                raise ToolError(
                    f"--{role.replace('_', '-')} was provided, but workflow "
                    f"{workflow_name!r} has no {role!r} binding",
                    code="UNBOUND_PARAMETER",
                    details={"workflow": workflow_name, "binding": role},
                )
            continue
        items = validate_binding(
            planned_graph,
            bindings[role],
            f"{workflow_name}.{role}",
        )
        for item in items:
            original = planned_graph[item["node"]]["inputs"][item["input"]]
            if role in explicit and is_link_value(original):
                raise ToolError(
                    f"{workflow_name}.{role} points to a linked input "
                    f"({item['node']}.{item['input']}); refusing to overwrite it",
                    code="LINKED_INPUT",
                    details={
                        "workflow": workflow_name,
                        "binding": role,
                        "node": item["node"],
                        "input": item["input"],
                    },
                )
            converted = coerce_parameter(role, values[role], specs.get(role))
            set_node_input(planned_graph, item, converted)

    reference_bindings = (
        validate_binding(
            planned_graph,
            bindings.get(REFERENCE_ROLE) or [],
            f"{workflow_name}.reference",
        )
        if bindings.get(REFERENCE_ROLE)
        else []
    )
    if args.reference and not reference_bindings:
        raise ToolError(
            f"Workflow {workflow_name!r} has no reference input binding",
            code="UNBOUND_REFERENCE",
            details={"workflow": workflow_name},
        )
    if len(args.reference) > len(reference_bindings):
        raise ToolError(
            f"Workflow {workflow_name!r} accepts at most "
            f"{len(reference_bindings)} reference file(s)",
            code="TOO_MANY_REFERENCES",
            details={"workflow": workflow_name},
        )

    uploads: list[dict[str, Any]] = []
    for index, (raw_reference, binding) in enumerate(
        zip(args.reference, reference_bindings)
    ):
        reference = Path(raw_reference).expanduser().resolve()
        if not reference.is_file():
            raise ToolError(
                f"Reference file not found: {reference}",
                code="REFERENCE_NOT_FOUND",
                details={"reference": str(reference)},
            )
        try:
            with reference.open("rb"):
                pass
        except OSError as exc:
            raise ToolError(
                f"Reference file is not readable: {reference}: {exc}",
                code="REFERENCE_UNREADABLE",
                details={"reference": str(reference)},
            ) from exc
        placeholder = f"__COMFYUI_PORTABLE_UPLOAD_{index}__"
        uploads.append(
            {
                "index": index,
                "node": binding["node"],
                "input": binding["input"],
                "source": str(reference),
                "placeholder": placeholder,
                "overwrite": False,
                "job_id": job_id,
            }
        )
        set_node_input(planned_graph, binding, placeholder)

    for override in args.set:
        if "." not in override or "=" not in override:
            raise ToolError(
                f"--set must use NODE.INPUT=VALUE: {override!r}",
                code="INVALID_ARGUMENT",
            )
        node_input, raw_value = override.split("=", 1)
        node_id, input_name = node_input.split(".", 1)
        if (
            node_id not in planned_graph
            or input_name not in planned_graph[node_id]["inputs"]
        ):
            raise ToolError(
                f"--set points to missing {node_id}.{input_name}",
                code="SET_TARGET_MISSING",
            )
        planned_graph[node_id]["inputs"][input_name] = parse_cli_value(raw_value)

    request_summary = {
        name: values.get(name) for name in sorted(PARAMETER_NAMES) if name in values
    }
    return {
        "graph": planned_graph,
        "uploads": uploads,
        "request_summary": request_summary,
        "warnings": warnings,
    }


def execute_uploads(
    config: dict[str, Any],
    uploads: list[dict[str, Any]],
    graph: dict[str, Any],
) -> list[dict[str, Any]]:
    completed: list[dict[str, Any]] = []
    for upload in uploads:
        server_name = upload_reference(
            config,
            Path(upload["source"]),
            job_id=str(upload["job_id"]),
            index=int(upload["index"]),
        )
        set_node_input(
            graph,
            {"node": upload["node"], "input": upload["input"]},
            server_name,
        )
        completed.append(
            {
                "source": upload["source"],
                "server_path": server_name,
                "node": upload["node"],
                "input": upload["input"],
            }
        )
    return completed


def upload_reference(
    config: dict[str, Any],
    file_path: Path,
    *,
    job_id: str,
    index: int,
) -> str:
    if not file_path.is_file():
        raise ToolError(
            f"Reference file not found: {file_path}",
            code="REFERENCE_NOT_FOUND",
        )

    safe_stem = (
        "".join(
            character if character.isalnum() or character in {"-", "_", "."} else "_"
            for character in file_path.stem
        )
        or "reference"
    )
    suffix = "".join(
        character if character.isalnum() or character == "." else "_"
        for character in file_path.suffix
    )
    unique_name = f"{index + 1:02d}-{uuid.uuid4().hex[:12]}-{safe_stem}{suffix}"
    server_subfolder = f"comfyui-portable/{validate_job_id(job_id)}"

    boundary = "----comfyui-portable-" + uuid.uuid4().hex
    content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    try:
        file_bytes = file_path.read_bytes()
    except OSError as exc:
        raise ToolError(
            f"Cannot read reference file {file_path}: {exc}",
            code="REFERENCE_UNREADABLE",
        ) from exc
    chunks: list[bytes] = []

    def add_field(name: str, value: str) -> None:
        chunks.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n"
            ).encode()
        )

    add_field("type", "input")
    add_field("overwrite", "false")
    add_field("subfolder", server_subfolder)
    escaped_filename = unique_name.replace("\\", "\\\\").replace('"', '\\"')
    chunks.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; '
            f'filename="{escaped_filename}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode()
    )
    chunks.append(file_bytes)
    chunks.append(f"\r\n--{boundary}--\r\n".encode())

    result = http_request(
        server_url(config) + "/upload/image",
        method="POST",
        payload=b"".join(chunks),
        headers=merged_headers(
            config,
            {"Content-Type": f"multipart/form-data; boundary={boundary}"},
        ),
        timeout=float(config.get("timeout_seconds", 30)),
    )
    if not isinstance(result, dict) or not result.get("name"):
        raise ToolError(
            f"Upload response did not contain a filename: {result!r}",
            code="UPLOAD_INVALID_RESPONSE",
        )
    name = Path(str(result["name"])).name
    subfolder = str(result.get("subfolder") or "").strip("/")
    if ".." in Path(subfolder).parts:
        raise ToolError(
            f"Upload response contains an unsafe subfolder: {subfolder!r}",
            code="UPLOAD_INVALID_RESPONSE",
        )
    return f"{subfolder}/{name}" if subfolder else name


def gather_outputs(history_item: dict[str, Any]) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    raw_outputs = history_item.get("outputs") or {}
    if not isinstance(raw_outputs, dict):
        return outputs
    for node_id, node_output in raw_outputs.items():
        if not isinstance(node_output, dict):
            continue
        for output_key, values in node_output.items():
            if not isinstance(values, list):
                continue
            for output_index, item in enumerate(values):
                if isinstance(item, dict) and item.get("filename"):
                    enriched = dict(item)
                    enriched["node_id"] = str(node_id)
                    enriched["output_key"] = str(output_key)
                    enriched["output_index"] = output_index
                    outputs.append(enriched)
    return outputs


def unique_destination(
    directory: Path,
    filename: str,
    *,
    overwrite: bool,
    fallback_stem: str,
) -> Path:
    safe_name = Path(filename).name or fallback_stem
    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / safe_name
    if overwrite or not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    index = 2
    while True:
        candidate = directory / f"{stem}-{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def output_destinations(
    requested: str | None,
    workflow_name: str,
    outputs: list[dict[str, Any]],
    *,
    overwrite: bool = False,
) -> list[Path]:
    if requested:
        base = Path(requested).expanduser().resolve()
        single_file = len(outputs) == 1 and bool(base.suffix)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        base = SKILL_ROOT / "outputs" / f"{workflow_name}-{timestamp}"
        single_file = False

    if single_file:
        base.parent.mkdir(parents=True, exist_ok=True)
        if base.exists() and not overwrite:
            base = unique_destination(
                base.parent,
                base.name,
                overwrite=False,
                fallback_stem="output",
            )
        return [base]

    if base.suffix:
        base = base.with_suffix("")
    base.mkdir(parents=True, exist_ok=True)
    destinations: list[Path] = []
    for index, output in enumerate(outputs, start=1):
        original = Path(str(output.get("filename") or f"output-{index}.png")).name
        node_id = str(output.get("node_id") or "node")
        output_index = int(output.get("output_index") or 0)
        preferred = f"{node_id}-{output_index + 1}-{original}"
        destinations.append(
            unique_destination(
                base,
                preferred,
                overwrite=overwrite,
                fallback_stem=f"output-{index}.png",
            )
        )
    return destinations


def download_output(
    config: dict[str, Any],
    output: dict[str, Any],
    destination: Path,
    *,
    overwrite: bool,
) -> None:
    if destination.exists() and not overwrite:
        raise ExecutionError(
            f"Refusing to overwrite existing output: {destination}",
            code="OUTPUT_EXISTS",
            details={"path": str(destination)},
        )
    query = urllib.parse.urlencode(
        {
            "filename": output["filename"],
            "subfolder": output.get("subfolder", ""),
            "type": output.get("type", "output"),
        }
    )
    data = http_request(
        server_url(config) + "/view?" + query,
        headers=merged_headers(config),
        timeout=float(config.get("timeout_seconds", 30)),
        raw=True,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
    try:
        temporary.write_bytes(data)
        if overwrite:
            temporary.replace(destination)
        else:
            temporary.rename(destination)
    except OSError as exc:
        with contextlib.suppress(OSError):
            temporary.unlink()
        raise ExecutionError(
            f"Cannot save output {destination}: {exc}",
            code="OUTPUT_WRITE_FAILED",
            details={"path": str(destination)},
        ) from exc


def submit_prompt(
    graph: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    client_id = str(uuid.uuid4())
    try:
        result = http_request(
            server_url(config) + "/prompt",
            method="POST",
            payload={"prompt": graph, "client_id": client_id},
            headers=merged_headers(config),
            timeout=float(config.get("timeout_seconds", 30)),
        )
    except TransportError as exc:
        raise UnknownStateError(
            "The /prompt request did not receive a response; submission "
            "state is unknown. Do not resubmit automatically.",
            code="SUBMISSION_UNKNOWN",
            details={"client_id": client_id},
        ) from exc
    except ToolError as exc:
        status = exc.details.get("http_status")
        if isinstance(status, int) and status >= 500:
            raise UnknownStateError(
                "/prompt returned a server error; submission state is unknown. "
                "Do not resubmit automatically.",
                code="SUBMISSION_UNKNOWN",
                details={"client_id": client_id, "http_status": status},
            ) from exc
        exc.state = "rejected"
        raise
    if not isinstance(result, dict):
        raise ExecutionError(
            f"Unexpected /prompt response: {result!r}",
            code="INVALID_SERVER_RESPONSE",
        )
    prompt_id = result.get("prompt_id")
    node_errors = result.get("node_errors") or {}
    if not prompt_id:
        if node_errors:
            raise ToolError(
                "ComfyUI rejected the graph: "
                + json.dumps(node_errors, ensure_ascii=False),
                code="GRAPH_REJECTED",
                state="rejected",
                details={"node_errors": node_errors},
            )
        raise ExecutionError(
            f"/prompt did not return prompt_id: {result!r}",
            code="PROMPT_ID_MISSING",
        )
    return {
        "prompt_id": str(prompt_id),
        "client_id": client_id,
        "node_errors": node_errors,
        "state": "accepted_with_node_errors" if node_errors else "accepted",
    }


def history_item(
    prompt_id: str,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    history = http_request(
        server_url(config) + "/history/" + urllib.parse.quote(prompt_id),
        headers=merged_headers(config),
        timeout=float(config.get("timeout_seconds", 30)),
    )
    if not isinstance(history, dict):
        raise ExecutionError(
            f"Unexpected /history response: {history!r}",
            code="INVALID_SERVER_RESPONSE",
        )
    item = history.get(prompt_id)
    return item if isinstance(item, dict) else None


def queue_state(prompt_id: str, config: dict[str, Any]) -> str | None:
    try:
        queue = http_request(
            server_url(config) + "/queue",
            headers=merged_headers(config),
            timeout=float(config.get("timeout_seconds", 30)),
        )
    except ToolError:
        return None
    if not isinstance(queue, dict):
        return None
    for key, state in (("running", "running"), ("pending", "pending")):
        entries = queue.get(key) or []
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, list) or not entry:
                continue
            candidate = entry[1] if len(entry) > 1 else None
            if str(candidate) == prompt_id:
                return state
    return None


def job_state(
    prompt_id: str,
    config: dict[str, Any],
) -> tuple[str, dict[str, Any] | None]:
    item = history_item(prompt_id, config)
    if item is not None:
        status = item.get("status") or {}
        status_name = str(status.get("status_str") or "").lower()
        completed = bool(status.get("completed"))
        if status_name in {"error", "failed"} or (
            completed and status_name not in {"success"}
        ):
            return "failed", item
        if status_name == "success" or completed:
            return "completed", item
        return "running", item
    queued = queue_state(prompt_id, config)
    if queued:
        return queued, None
    return "unknown", None


def manifest_dir(config_path_value: Path) -> Path:
    return config_path_value.parent / ".comfyui-portable" / "jobs"


def validate_job_id(value: str) -> str:
    candidate = str(value).strip()
    if not candidate or any(
        not (character.isalnum() or character in {"-", "_"}) for character in candidate
    ):
        raise ToolError(
            "Job ID may only contain letters, numbers, hyphens, and underscores",
            code="INVALID_JOB_ID",
        )
    return candidate


def manifest_path_for_job(config_path_value: Path, job_id: str) -> Path:
    return manifest_dir(config_path_value) / f"{validate_job_id(job_id)}.json"


def resolve_job_manifest(config_path_value: Path, value: str) -> Path:
    direct = Path(value).expanduser()
    if direct.is_file():
        return direct.resolve()
    return manifest_path_for_job(config_path_value, value)


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = read_json(path)
    if not isinstance(manifest, dict):
        raise ToolError(
            f"Job manifest must be a JSON object: {path}",
            code="INVALID_JOB_MANIFEST",
        )
    return manifest


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    write_json(path, manifest)


def prepare_submission(
    args: argparse.Namespace,
    config_path_value: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    str,
    dict[str, Any],
    dict[str, Any],
    str,
]:
    config = load_config(config_path_value)
    if (
        config.get("_legacy_schema") or config.get("validation_state") != "ready"
    ) and not getattr(args, "allow_draft", False):
        raise ToolError(
            "The local profile is not ready. Run setup online or doctor "
            "before submitting work.",
            code="PROFILE_NOT_READY",
            state="needs_configuration",
            details={"validation_state": config.get("validation_state", "unknown")},
        )
    workflow_name, definition = select_workflow(config, args.workflow)
    validate_definition_current(
        config,
        workflow_name,
        definition,
        ignore_drift=bool(getattr(args, "ignore_drift", False)),
    )
    graph_path = resolve_stored_path(definition["file"])
    graph = load_graph(graph_path)
    job_id = getattr(args, "job_id", None) or str(uuid.uuid4())
    plan = build_plan(graph, definition, args, workflow_name, job_id)
    return config, definition, workflow_name, graph_path, plan, job_id


def submit_prepared(
    config: dict[str, Any],
    config_path_value: Path,
    workflow_name: str,
    graph_path: Path,
    plan: dict[str, Any],
    job_id: str,
) -> tuple[dict[str, Any], Path]:
    path = manifest_path_for_job(config_path_value, job_id)
    manifest: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "job_id": job_id,
        "state": "submitting",
        "server": server_url(config),
        "workflow": workflow_name,
        "workflow_file": str(graph_path),
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "request_summary": plan["request_summary"],
        "graph_sha256": None,
        "uploads": [],
        "prompt_id": None,
        "client_id": None,
        "node_errors": {},
        "outputs": [],
    }
    write_manifest(path, manifest)

    try:
        uploads = execute_uploads(config, plan["uploads"], plan["graph"])
        manifest["uploads"] = uploads
        manifest["graph_sha256"] = content_hash(plan["graph"])
        manifest["updated_at"] = utc_now()
        write_manifest(path, manifest)
        submitted = submit_prompt(plan["graph"], config)
    except UnknownStateError as exc:
        manifest["state"] = "unknown"
        manifest["client_id"] = exc.details.get("client_id")
        manifest["graph_sha256"] = content_hash(plan["graph"])
        manifest["updated_at"] = utc_now()
        manifest["error"] = exc.as_error()
        write_manifest(path, manifest)
        exc.details.setdefault("job_id", job_id)
        exc.details.setdefault("manifest", str(path))
        raise
    except ToolError as exc:
        manifest["state"] = exc.state
        manifest["graph_sha256"] = content_hash(plan["graph"])
        manifest["updated_at"] = utc_now()
        manifest["error"] = exc.as_error()
        write_manifest(path, manifest)
        exc.details.setdefault("job_id", job_id)
        exc.details.setdefault("manifest", str(path))
        raise

    manifest.update(submitted)
    manifest["updated_at"] = utc_now()
    write_manifest(path, manifest)
    return manifest, path


def result_payload(
    *,
    ok: bool,
    state: str,
    error: dict[str, Any] | None = None,
    prompt_id: str | None = None,
    files: list[str] | None = None,
    warnings: list[str] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "ok": ok,
        "state": state,
        "error": error,
        "prompt_id": prompt_id,
        "files": files or [],
        "warnings": warnings or [],
    }
    payload.update(extra)
    return payload


def emit_payload(
    args: argparse.Namespace,
    payload: dict[str, Any],
    human_lines: list[str] | None = None,
) -> None:
    if getattr(args, "json", False):
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    for line in human_lines or []:
        print(line)


def command_setup(args: argparse.Namespace) -> int:
    path = config_path(args.config)
    existing = load_config(path, required=False)

    server = (
        args.server
        or os.environ.get("COMFYUI_URL")
        or existing.get("server")
        or DEFAULT_SERVER
    ).rstrip("/")
    headers = dict(existing.get("headers") or {})
    headers.update(parse_key_values(args.header, "--header"))

    root: Path | None = None
    if args.comfy_root:
        root = normalize_comfy_root(Path(args.comfy_root))
        if not root:
            raise ToolError(
                f"Not a ComfyUI root: {args.comfy_root}",
                code="COMFY_ROOT_NOT_FOUND",
            )
    elif not args.no_discovery:
        root = discover_comfy_root()

    python_path: Path | None = None
    if args.python:
        python_path = Path(args.python).expanduser().resolve()
        if not python_path.is_file():
            raise ToolError(
                f"Python executable not found: {python_path}",
                code="PYTHON_NOT_FOUND",
            )
    else:
        python_path = find_comfy_python(root)

    temporary_config: dict[str, Any] = {
        **existing,
        "schema_version": SCHEMA_VERSION,
        "server": server,
        "timeout_seconds": int(args.timeout or existing.get("timeout_seconds", 30)),
        "poll_interval_seconds": float(
            args.poll_interval
            if args.poll_interval is not None
            else existing.get("poll_interval_seconds", 2)
        ),
        "max_wait_seconds": int(args.max_wait or existing.get("max_wait_seconds", 900)),
        "headers": headers,
        "state_dir": str(path.parent / ".comfyui-portable"),
    }
    temporary_config.pop("_legacy_schema", None)

    object_info: dict[str, Any] | None = None
    security_warnings = server_security_warnings(temporary_config)
    if not args.offline:
        try:
            get_system_stats(temporary_config)
            object_info = get_object_info(temporary_config)
        except ToolError as exc:
            raise UnreachableError(
                f"Online setup could not validate {server}: {exc}",
                server=server,
            ) from exc

    model_overrides = parse_scoped_values(args.model, "--model")
    binding_overrides = parse_scoped_values(args.binding, "--binding")
    default_overrides = parse_scoped_values(args.default, "--default")
    workflow_args = parse_workflow_arguments(args.workflow)
    descriptor_args = parse_descriptor_arguments(args.descriptor)
    workflows = workflows_to_configure(workflow_args, descriptor_args)
    workflow_names = {name for name, _, _ in workflows}
    for label, values in (
        ("--model", model_overrides),
        ("--binding", binding_overrides),
        ("--default", default_overrides),
    ):
        unknown_scopes = sorted(set(values) - workflow_names)
        if unknown_scopes:
            raise ToolError(
                f"{label} references unknown workflow scope(s): "
                f"{', '.join(unknown_scopes)}",
                code="UNKNOWN_WORKFLOW_SCOPE",
            )

    interactive = not args.non_interactive and sys.stdin.isatty()
    configured_workflows: dict[str, Any] = (
        dict(existing.get("workflows") or {})
        if existing.get("schema_version") == SCHEMA_VERSION
        else {}
    )
    setup_errors: list[dict[str, Any]] = []
    setup_warnings = list(security_warnings)
    model_options = model_options_from_object_info(object_info)
    all_enum_options = enumerations_from_object_info(object_info)

    for name, workflow_path, descriptor_path in workflows:
        graph = load_graph(workflow_path)
        automatic = inspect_workflow(graph, object_info)
        descriptor = read_json(descriptor_path) if descriptor_path else None
        merged = merge_descriptor(
            automatic,
            descriptor,
            descriptor_path=descriptor_path,
        )

        for role, raw_binding in binding_overrides.get(name, {}).items():
            if role not in SUPPORTED_BINDING_ROLES:
                raise ToolError(
                    f"Unknown binding role {role!r} for {name}",
                    code="UNKNOWN_BINDING_ROLE",
                )
            if "." not in raw_binding:
                raise ToolError(
                    f"--binding {name}.{role} must use NODE.INPUT: {raw_binding!r}",
                    code="INVALID_ARGUMENT",
                )
            node_id, input_name = raw_binding.split(".", 1)
            merged["bindings"][role] = {"node": node_id, "input": input_name}

        for role, binding in (merged.get("bindings") or {}).items():
            if role not in SUPPORTED_BINDING_ROLES:
                raise ToolError(
                    f"Unknown binding role {role!r} for {name}",
                    code="UNKNOWN_BINDING_ROLE",
                )
            validate_binding(graph, binding, f"{name}.{role}")

        binding_roles = set(merged.get("bindings") or {})
        unconsumed_params = sorted(set(merged.get("params") or {}) - binding_roles)
        if unconsumed_params:
            raise ToolError(
                f"{name} declares params without bindings: "
                f"{', '.join(unconsumed_params)}",
                code="UNCONSUMED_PARAMETER",
            )
        unconsumed_defaults = sorted(set(merged.get("defaults") or {}) - binding_roles)
        if unconsumed_defaults:
            raise ToolError(
                f"{name} declares defaults without bindings: "
                f"{', '.join(unconsumed_defaults)}",
                code="UNCONSUMED_DEFAULT",
            )

        final_models: dict[str, Any] = {}
        missing_overrides = set(model_overrides.get(name, {})) - set(
            merged.get("models") or {}
        )
        if missing_overrides:
            raise ToolError(
                f"Unknown model bindings for {name}: "
                f"{', '.join(sorted(missing_overrides))}",
                code="UNKNOWN_MODEL_ROLE",
            )

        for role, binding in (merged.get("models") or {}).items():
            items = validate_binding(graph, binding, f"{name}.{role}")
            item = items[0]
            node_id = item["node"]
            input_name = item["input"]
            current = graph[node_id]["inputs"].get(input_name)
            desired = model_overrides.get(name, {}).get(role, current)
            class_name = node_class(graph[node_id])
            option_key = f"{class_name}.{input_name}"
            options = model_options.get(option_key)
            if options is None:
                options = all_enum_options.get(option_key)

            if options is None and object_info:
                setup_errors.append(
                    {
                        "code": "MODEL_OPTIONS_UNKNOWN",
                        "message": (
                            f"{name}.{role}: {option_key} is not a recognized "
                            "model option list"
                        ),
                        "workflow": name,
                        "binding": role,
                    }
                )
            elif options == []:
                setup_errors.append(
                    {
                        "code": "NEEDS_MODEL_SELECTION",
                        "message": (
                            f"{name}.{role}: the server reports no installed "
                            "models for this loader"
                        ),
                        "workflow": name,
                        "binding": role,
                        "candidates": [],
                        "compatibility": "unverified",
                    }
                )
            elif options is not None and desired not in options:
                if interactive:
                    desired = prompt_model_choice(f"{name}.{role}", desired, options)
                if desired not in options:
                    setup_errors.append(
                        {
                            "code": "NEEDS_MODEL_SELECTION",
                            "message": (
                                f"{name}.{role}={desired!r} is not installed "
                                f"for {option_key}"
                            ),
                            "workflow": name,
                            "binding": role,
                            "candidates": options,
                            "compatibility": "unverified",
                        }
                    )

            final_models[role] = {
                "node": node_id,
                "input": input_name,
                "value": desired,
                "required": bool(binding.get("required", False)),
            }

        derived_defaults = derive_defaults(graph, merged.get("bindings") or {})
        derived_defaults.update(merged.get("defaults") or {})
        for key, value in default_overrides.get(name, {}).items():
            if key not in PARAMETER_NAMES:
                raise ToolError(
                    f"Unknown default role {key!r} for {name}",
                    code="UNKNOWN_DEFAULT_ROLE",
                )
            if key not in binding_roles:
                raise ToolError(
                    f"--default {name}.{key} has no matching binding",
                    code="UNCONSUMED_DEFAULT",
                )
            derived_defaults[key] = parse_cli_value(value)

        required_nodes = sorted(
            set(automatic.get("required_nodes") or [])
            | set(merged.get("required_nodes") or [])
        )
        if object_info:
            missing_nodes = [
                class_name
                for class_name in required_nodes
                if class_name not in object_info
            ]
            if missing_nodes:
                setup_errors.append(
                    {
                        "code": "NODE_CLASS_MISSING",
                        "message": (
                            f"{name}: missing node classes: {', '.join(missing_nodes)}"
                        ),
                        "workflow": name,
                        "required_nodes": missing_nodes,
                    }
                )

        parameter_specs = binding_specs(
            graph,
            merged.get("bindings") or {},
            object_info,
            merged.get("params") or {},
        )
        for role, value in derived_defaults.items():
            coerce_parameter(role, value, parameter_specs.get(role))
        configured_workflows[name] = {
            "file": store_path(workflow_path),
            "descriptor": store_path(descriptor_path) if descriptor_path else None,
            "bindings": merged.get("bindings") or {},
            "models": final_models,
            "defaults": derived_defaults,
            "params": parameter_specs,
            "required_nodes": required_nodes,
            "graph_sha256": file_hash(workflow_path),
            "descriptor_sha256": file_hash(descriptor_path)
            if descriptor_path
            else None,
            "node_schema_sha256": content_hash(
                {
                    class_name: (object_info or {}).get(class_name)
                    for class_name in required_nodes
                }
            ),
            "task_type": merged.get("task_type"),
        }

    config: dict[str, Any] = {
        **temporary_config,
        "comfyui": {
            "root": str(root) if root else None,
            "python": str(python_path) if python_path else None,
            "input_dir": str(root / "input") if root else None,
            "output_dir": str(root / "output") if root else None,
        },
        "workflows": configured_workflows,
        "validation_state": "draft" if (args.offline or setup_errors) else "ready",
        "validation": {
            "ok": not setup_errors,
            "errors": setup_errors,
            "warnings": setup_warnings,
        },
    }

    ready = not args.offline and not setup_errors
    should_write = (
        ready
        or args.offline
        or not path.exists()
        or existing.get("validation_state") != "ready"
    )
    if should_write:
        write_json(path, config)

    payload = result_payload(
        ok=ready or args.offline,
        state="ready" if ready else "draft",
        error=setup_errors[0] if setup_errors else None,
        prompt_id=None,
        files=[],
        warnings=setup_warnings,
        profile=str(path),
        validation={"errors": setup_errors, "warnings": setup_warnings},
        workflows=sorted(configured_workflows),
    )
    human = [
        f"Profile written: {path}" if should_write else f"Profile kept: {path}",
        f"Server: {server}",
        f"Validation: {'ready' if ready else 'draft'}",
    ]
    emit_payload(args, payload, human)
    if setup_errors and not args.offline:
        return EXIT_CONFIG
    return EXIT_OK


def add_check(
    checks: list[dict[str, str]],
    level: str,
    message: str,
    code: str | None = None,
) -> None:
    check = {"level": level, "message": message}
    if code:
        check["code"] = code
    checks.append(check)


def node_schema_fingerprint(
    object_info: dict[str, Any],
    required_nodes: Iterable[str],
) -> str:
    return content_hash(
        {
            class_name: object_info.get(class_name)
            for class_name in sorted(set(required_nodes))
        }
    )


def validate_definition_current(
    config: dict[str, Any],
    workflow_name: str,
    definition: dict[str, Any],
    *,
    ignore_drift: bool = False,
) -> list[str]:
    warnings: list[str] = []
    graph_path = resolve_stored_path(definition["file"])
    current_graph_hash = file_hash(graph_path)
    stored_graph_hash = definition.get("graph_sha256")
    if stored_graph_hash and stored_graph_hash != current_graph_hash:
        if not ignore_drift:
            raise ToolError(
                f"{workflow_name}: workflow changed since setup; run doctor "
                "and configure it again",
                code="WORKFLOW_DRIFT",
                state="needs_configuration",
                details={"workflow": workflow_name},
            )
        warnings.append(f"{workflow_name}: workflow drift ignored")

    descriptor_value = definition.get("descriptor")
    stored_descriptor_hash = definition.get("descriptor_sha256")
    if descriptor_value and stored_descriptor_hash:
        descriptor_path = resolve_stored_path(descriptor_value)
        current_descriptor_hash = file_hash(descriptor_path)
        if current_descriptor_hash != stored_descriptor_hash:
            if not ignore_drift:
                raise ToolError(
                    f"{workflow_name}: descriptor changed since setup; run "
                    "doctor and configure it again",
                    code="DESCRIPTOR_DRIFT",
                    state="needs_configuration",
                    details={"workflow": workflow_name},
                )
            warnings.append(f"{workflow_name}: descriptor drift ignored")
    return warnings


def command_doctor(args: argparse.Namespace) -> int:
    path = config_path(args.config)
    config = load_config(path)
    checks: list[dict[str, str]] = []
    warnings = server_security_warnings(config)
    add_check(checks, "PASS", f"Profile loaded: {path}")
    if config.get("_legacy_schema"):
        add_check(
            checks,
            "FAIL",
            "Profile uses schema v1; run setup again to migrate it",
            "PROFILE_SCHEMA_MIGRATION_REQUIRED",
        )
    if config.get("validation_state") != "ready":
        add_check(
            checks,
            "FAIL",
            f"Profile validation state is {config.get('validation_state')!r}",
            "PROFILE_NOT_READY",
        )

    comfyui = config.get("comfyui") or {}
    root_value = comfyui.get("root")
    if root_value:
        root = Path(root_value)
        if is_comfy_root(root):
            add_check(checks, "PASS", f"ComfyUI root: {root}")
        else:
            add_check(checks, "FAIL", f"ComfyUI root is invalid: {root}")
    else:
        add_check(checks, "WARN", "No local ComfyUI root; using HTTP-only mode")

    object_info: dict[str, Any] | None = None
    server_failed = False
    try:
        stats = get_system_stats(config)
        devices = stats.get("devices") or []
        names = [
            str(device.get("name") or device.get("type") or "unknown")
            for device in devices
            if isinstance(device, dict)
        ]
        add_check(
            checks,
            "PASS",
            f"Server reachable: {server_url(config)}"
            + (f" ({', '.join(names)})" if names else ""),
        )
        if args.require_cuda and not any(
            isinstance(device, dict) and str(device.get("type", "")).lower() == "cuda"
            for device in devices
        ):
            add_check(checks, "FAIL", "No CUDA device reported")
        object_info = get_object_info(config)
        add_check(checks, "PASS", f"Node registry: {len(object_info)} classes")
    except ToolError as exc:
        server_failed = True
        add_check(checks, "FAIL", str(exc), exc.code)

    model_options = model_options_from_object_info(object_info)
    all_enum_options = enumerations_from_object_info(object_info)
    workflows = config.get("workflows") or {}
    if not workflows:
        add_check(checks, "FAIL", "No workflows configured", "NO_WORKFLOWS")

    for name, definition in workflows.items():
        try:
            graph_path = resolve_stored_path(definition["file"])
            graph = load_graph(graph_path)
        except (KeyError, ToolError) as exc:
            add_check(checks, "FAIL", f"{name}: {exc}", "WORKFLOW_LOAD_FAILED")
            continue

        add_check(checks, "PASS", f"{name}: workflow graph loaded")
        try:
            validate_definition_current(config, name, definition)
        except ToolError as exc:
            add_check(checks, "FAIL", str(exc), exc.code)

        for role, binding in (definition.get("bindings") or {}).items():
            try:
                validate_binding(graph, binding, f"{name}.{role}")
            except ToolError as exc:
                add_check(checks, "FAIL", str(exc), exc.code)
        for role, binding in (definition.get("models") or {}).items():
            try:
                items = validate_binding(graph, binding, f"{name}.{role}")
            except ToolError as exc:
                add_check(checks, "FAIL", str(exc), exc.code)
                continue
            item = items[0]
            class_name = node_class(graph[item["node"]])
            option_key = f"{class_name}.{item['input']}"
            value = binding.get("value")
            options = model_options.get(option_key)
            if options is None:
                options = all_enum_options.get(option_key)
            if object_info and options is None:
                add_check(
                    checks,
                    "FAIL",
                    f"{name}.{role}: {option_key} has no recognized option list",
                    "MODEL_OPTIONS_UNKNOWN",
                )
            elif options == []:
                add_check(
                    checks,
                    "FAIL",
                    f"{name}.{role}: server reports no installed models",
                    "NEEDS_MODEL_SELECTION",
                )
            elif options is not None and value not in options:
                add_check(
                    checks,
                    "FAIL",
                    f"{name}.{role}: {value!r} is not installed for {option_key}",
                    "MODEL_NOT_INSTALLED",
                )
            elif options is not None:
                add_check(checks, "PASS", f"{name}.{role}: {value}")

        required_nodes = sorted(
            set(definition.get("required_nodes") or [])
            | {node_class(node) for node in graph.values() if node_class(node)}
        )
        if object_info:
            for class_name in required_nodes:
                if class_name in object_info:
                    add_check(checks, "PASS", f"{name}: node {class_name}")
                else:
                    add_check(
                        checks,
                        "FAIL",
                        f"{name}: missing node class {class_name}",
                        "NODE_CLASS_MISSING",
                    )
            stored_fingerprint = definition.get("node_schema_sha256")
            current_fingerprint = node_schema_fingerprint(
                object_info,
                required_nodes,
            )
            if stored_fingerprint and stored_fingerprint != current_fingerprint:
                add_check(
                    checks,
                    "FAIL",
                    f"{name}: node schemas changed since setup",
                    "NODE_SCHEMA_DRIFT",
                )

    failed = any(check["level"] == "FAIL" for check in checks)
    payload = result_payload(
        ok=not failed,
        state="ready" if not failed else "needs_configuration",
        error=(
            {
                "code": next(
                    (check.get("code") for check in checks if check["level"] == "FAIL"),
                    "DOCTOR_FAILED",
                ),
                "message": next(
                    check["message"] for check in checks if check["level"] == "FAIL"
                ),
                "retryable": False,
            }
            if failed
            else None
        ),
        warnings=warnings,
        checks=checks,
    )
    if args.json:
        emit_payload(args, payload)
    else:
        for check in checks:
            print(f"[{check['level']}] {check['message']}")
        for warning in warnings:
            print(f"[WARN] {warning}")
        print("Doctor result:", "FAIL" if failed else "PASS")
    if not failed:
        return EXIT_OK
    return EXIT_UNREACHABLE if server_failed else EXIT_CONFIG


def command_inspect(args: argparse.Namespace) -> int:
    path = Path(args.workflow).expanduser().resolve()
    graph = load_graph(path)
    object_info: dict[str, Any] | None = None
    if args.server:
        config = {"server": args.server, "timeout_seconds": args.timeout}
        object_info = get_object_info(config)
    automatic = inspect_workflow(graph, object_info)
    summary = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "ok": not automatic.get("ambiguities"),
        "state": "ready" if not automatic.get("ambiguities") else "needs_configuration",
        "workflow": str(path),
        "graph_sha256": file_hash(path),
        "nodes": [
            {
                "id": node_id,
                "class_type": node_class(graph[node_id]),
                "inputs": sorted((graph[node_id].get("inputs") or {}).keys()),
            }
            for node_id in sorted_node_ids(graph)
        ],
        **automatic,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return EXIT_OK if summary["ok"] else EXIT_CONFIG


def select_workflow(
    config: dict[str, Any],
    requested: str | None,
) -> tuple[str, dict[str, Any]]:
    workflows = config.get("workflows") or {}
    if requested:
        if requested not in workflows:
            raise ToolError(
                f"Unknown workflow {requested!r}. "
                f"Available: {', '.join(sorted(workflows)) or 'none'}",
                code="UNKNOWN_WORKFLOW",
            )
        return requested, workflows[requested]
    if len(workflows) == 1:
        name = next(iter(workflows))
        return name, workflows[name]
    raise ToolError(
        "Multiple workflows are configured; pass --workflow. "
        f"Available: {', '.join(sorted(workflows))}",
        code="WORKFLOW_REQUIRED",
    )


def command_submit(args: argparse.Namespace) -> int:
    path = config_path(args.config)
    config, _, workflow_name, graph_path, plan, job_id = prepare_submission(
        args,
        path,
    )
    manifest, manifest_path = submit_prepared(
        config,
        path,
        workflow_name,
        graph_path,
        plan,
        job_id,
    )
    payload = result_payload(
        ok=True,
        state=str(manifest["state"]),
        prompt_id=manifest.get("prompt_id"),
        warnings=plan.get("warnings") or [],
        job_id=job_id,
        manifest=str(manifest_path),
        workflow=workflow_name,
        node_errors=manifest.get("node_errors") or {},
        uploads=manifest.get("uploads") or [],
    )
    emit_payload(
        args,
        payload,
        [
            f"Job: {job_id}",
            f"State: {manifest['state']}",
            f"Prompt ID: {manifest.get('prompt_id')}",
            f"Manifest: {manifest_path}",
        ],
    )
    return EXIT_OK


def wait_for_manifest(
    manifest: dict[str, Any],
    manifest_path: Path,
    config: dict[str, Any],
) -> tuple[str, dict[str, Any] | None]:
    prompt_id = manifest.get("prompt_id")
    if not prompt_id:
        raise UnknownStateError(
            "Job has no prompt_id; its submission state cannot be recovered",
            details={"job_id": manifest.get("job_id")},
        )
    interval = max(0.2, float(config.get("poll_interval_seconds", 2)))
    deadline = time.monotonic() + float(config.get("max_wait_seconds", 900))
    current_state = str(manifest.get("state") or "unknown")
    item: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        current_state, item = job_state(str(prompt_id), config)
        if current_state in {"completed", "failed"}:
            manifest["state"] = current_state
            manifest["updated_at"] = utc_now()
            if item is not None:
                manifest["history"] = item
            write_manifest(manifest_path, manifest)
            return current_state, item
        manifest["state"] = current_state
        manifest["updated_at"] = utc_now()
        write_manifest(manifest_path, manifest)
        time.sleep(interval)
    manifest["state"] = current_state
    manifest["updated_at"] = utc_now()
    write_manifest(manifest_path, manifest)
    raise UnknownStateError(
        f"Timed out after {config.get('max_wait_seconds', 900)} seconds "
        f"waiting for {prompt_id}",
        code="WAIT_TIMEOUT",
        details={
            "job_id": manifest.get("job_id"),
            "prompt_id": prompt_id,
            "manifest": str(manifest_path),
        },
    )


def command_status(args: argparse.Namespace) -> int:
    profile_path = config_path(args.config)
    config = load_config(profile_path)
    manifest_path = resolve_job_manifest(profile_path, args.job)
    manifest = load_manifest(manifest_path)
    prompt_id = manifest.get("prompt_id")
    if prompt_id:
        state, item = job_state(str(prompt_id), config)
        manifest["state"] = state
        manifest["updated_at"] = utc_now()
        if item is not None:
            manifest["history"] = item
        write_manifest(manifest_path, manifest)
    else:
        state = str(manifest.get("state") or "unknown")
    payload = result_payload(
        ok=state not in {"failed", "unknown"},
        state=state,
        error=(
            {
                "code": "JOB_FAILED" if state == "failed" else "STATUS_UNKNOWN",
                "message": f"Job state is {state}",
                "retryable": False,
            }
            if state in {"failed", "unknown"}
            else None
        ),
        prompt_id=prompt_id,
        job_id=manifest.get("job_id"),
        manifest=str(manifest_path),
        history=manifest.get("history"),
    )
    emit_payload(
        args,
        payload,
        [
            f"Job: {manifest.get('job_id')}",
            f"State: {state}",
            f"Prompt ID: {prompt_id}",
        ],
    )
    if state == "failed":
        return EXIT_EXECUTION
    if state == "unknown":
        return EXIT_UNKNOWN
    return EXIT_OK


def command_wait(args: argparse.Namespace) -> int:
    profile_path = config_path(args.config)
    config = load_config(profile_path)
    manifest_path = resolve_job_manifest(profile_path, args.job)
    manifest = load_manifest(manifest_path)
    state, item = wait_for_manifest(manifest, manifest_path, config)
    payload = result_payload(
        ok=state == "completed",
        state=state,
        error=(
            {
                "code": "JOB_FAILED",
                "message": "ComfyUI execution failed",
                "retryable": False,
            }
            if state == "failed"
            else None
        ),
        prompt_id=manifest.get("prompt_id"),
        job_id=manifest.get("job_id"),
        manifest=str(manifest_path),
        history=item,
    )
    emit_payload(
        args,
        payload,
        [
            f"Job: {manifest.get('job_id')}",
            f"State: {state}",
        ],
    )
    return EXIT_OK if state == "completed" else EXIT_EXECUTION


def fetch_manifest_outputs(
    manifest: dict[str, Any],
    manifest_path: Path,
    config: dict[str, Any],
    *,
    requested_out: str | None,
    overwrite: bool,
) -> list[str]:
    if manifest.get("state") != "completed":
        raise ExecutionError(
            f"Job state is {manifest.get('state')!r}; outputs are not ready",
            code="JOB_NOT_COMPLETE",
            details={
                "job_id": manifest.get("job_id"),
                "prompt_id": manifest.get("prompt_id"),
            },
        )
    item = manifest.get("history")
    if not isinstance(item, dict):
        prompt_id = manifest.get("prompt_id")
        if not prompt_id:
            raise UnknownStateError(
                "Job has no prompt_id",
                details={"job_id": manifest.get("job_id")},
            )
        state, item = job_state(str(prompt_id), config)
        if state != "completed" or item is None:
            raise ExecutionError(
                f"Job state is {state!r}; outputs are not ready",
                code="JOB_NOT_COMPLETE",
            )
        manifest["history"] = item
    outputs = gather_outputs(item)
    if not outputs:
        raise ExecutionError(
            "ComfyUI completed successfully but returned no outputs",
            code="NO_OUTPUTS",
            details={"job_id": manifest.get("job_id")},
        )
    destinations = output_destinations(
        requested_out,
        str(manifest.get("workflow") or "workflow"),
        outputs,
        overwrite=overwrite,
    )
    saved: list[dict[str, Any]] = []
    for output, destination in zip(outputs, destinations):
        download_output(
            config,
            output,
            destination,
            overwrite=overwrite,
        )
        saved.append(
            {
                "path": str(destination),
                "filename": output.get("filename"),
                "subfolder": output.get("subfolder", ""),
                "type": output.get("type", "output"),
                "node_id": output.get("node_id"),
                "output_index": output.get("output_index"),
            }
        )
    manifest["outputs"] = saved
    manifest["updated_at"] = utc_now()
    write_manifest(manifest_path, manifest)
    return [item["path"] for item in saved]


def command_fetch(args: argparse.Namespace) -> int:
    profile_path = config_path(args.config)
    config = load_config(profile_path)
    manifest_path = resolve_job_manifest(profile_path, args.job)
    manifest = load_manifest(manifest_path)
    files = fetch_manifest_outputs(
        manifest,
        manifest_path,
        config,
        requested_out=args.out,
        overwrite=args.overwrite,
    )
    payload = result_payload(
        ok=True,
        state="completed",
        prompt_id=manifest.get("prompt_id"),
        files=files,
        job_id=manifest.get("job_id"),
        manifest=str(manifest_path),
    )
    emit_payload(
        args,
        payload,
        [f"Saved: {path}" for path in files],
    )
    return EXIT_OK


def command_run(args: argparse.Namespace) -> int:
    profile_path = config_path(args.config)
    config, _, workflow_name, graph_path, plan, job_id = prepare_submission(
        args,
        profile_path,
    )
    if args.dry_run:
        files: list[str] = []
        if args.graph_out:
            destination = Path(args.graph_out).expanduser().resolve()
            write_json(destination, plan["graph"])
            files.append(str(destination))
        payload = result_payload(
            ok=True,
            state="planned",
            prompt_id=None,
            files=files,
            warnings=plan.get("warnings") or [],
            workflow=workflow_name,
            job_id=job_id,
            uploads=plan["uploads"],
        )
        if args.json:
            emit_payload(args, payload)
        elif args.graph_out:
            emit_payload(args, payload, [f"Resolved graph written: {files[0]}"])
        else:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
        return EXIT_OK

    manifest, manifest_path = submit_prepared(
        config,
        profile_path,
        workflow_name,
        graph_path,
        plan,
        job_id,
    )
    try:
        state, _ = wait_for_manifest(manifest, manifest_path, config)
    except UnknownStateError as exc:
        exc.details.setdefault("job_id", job_id)
        exc.details.setdefault("manifest", str(manifest_path))
        raise
    if state == "failed":
        raise ExecutionError(
            "ComfyUI execution failed",
            details={
                "job_id": job_id,
                "prompt_id": manifest.get("prompt_id"),
                "manifest": str(manifest_path),
            },
        )
    files = fetch_manifest_outputs(
        manifest,
        manifest_path,
        config,
        requested_out=args.out,
        overwrite=args.overwrite,
    )
    payload = result_payload(
        ok=True,
        state="completed",
        prompt_id=manifest.get("prompt_id"),
        files=files,
        warnings=plan.get("warnings") or [],
        workflow=workflow_name,
        job_id=job_id,
        manifest=str(manifest_path),
        node_errors=manifest.get("node_errors") or {},
        uploads=manifest.get("uploads") or [],
    )
    emit_payload(
        args,
        payload,
        [f"Saved: {file_path}" for file_path in files],
    )
    return EXIT_OK


def add_common_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        help="Local profile path. Defaults to config.local.json in the skill.",
    )


def add_job_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--job",
        required=True,
        help="Job ID or path to a job manifest",
    )


def add_submission_arguments(parser: argparse.ArgumentParser) -> None:
    add_common_config_argument(parser)
    parser.add_argument("--workflow", help="Configured workflow name")
    parser.add_argument("--prompt")
    parser.add_argument("--negative")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--cfg", type=float)
    parser.add_argument("--denoise", type=float)
    parser.add_argument("--sampler-name")
    parser.add_argument("--scheduler")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--resolution")
    parser.add_argument(
        "--reference",
        action="append",
        default=[],
        metavar="PATH",
        help="Reference image or video; repeat for multiple inputs",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NODE.INPUT=VALUE",
        help="Override any node input after semantic bindings",
    )
    parser.add_argument(
        "--job-id",
        help="Stable job ID; defaults to a generated UUID",
    )
    parser.add_argument(
        "--allow-draft",
        action="store_true",
        help="Allow an explicitly saved draft profile",
    )
    parser.add_argument(
        "--ignore-drift",
        action="store_true",
        help="Warn instead of refusing when workflow or descriptor hashes changed",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Discover and run ComfyUI workflows through a local profile."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup_parser = subparsers.add_parser(
        "setup",
        help="Detect the host and write a ready or draft local profile",
    )
    add_common_config_argument(setup_parser)
    setup_parser.add_argument("--server", help="ComfyUI base URL")
    setup_parser.add_argument("--comfy-root", help="ComfyUI installation root")
    setup_parser.add_argument("--python", help="Python executable to record")
    setup_parser.add_argument(
        "--workflow",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="API-format workflow to configure; repeat for multiple workflows",
    )
    setup_parser.add_argument(
        "--descriptor",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Explicit semantic descriptor for a workflow",
    )
    setup_parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="WORKFLOW.ROLE=VALUE",
        help="Lock a specific local model filename",
    )
    setup_parser.add_argument(
        "--binding",
        action="append",
        default=[],
        metavar="WORKFLOW.ROLE=NODE.INPUT",
        help="Override a semantic parameter binding",
    )
    setup_parser.add_argument(
        "--default",
        action="append",
        default=[],
        metavar="WORKFLOW.KEY=VALUE",
        help="Override a stored workflow default",
    )
    setup_parser.add_argument(
        "--header",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="HTTP header for authenticated ComfyUI APIs",
    )
    setup_parser.add_argument("--timeout", type=float, help="HTTP timeout in seconds")
    setup_parser.add_argument(
        "--poll-interval",
        type=float,
        help="History polling interval in seconds",
    )
    setup_parser.add_argument(
        "--max-wait",
        type=float,
        help="Maximum workflow wait in seconds",
    )
    setup_parser.add_argument(
        "--no-discovery",
        action="store_true",
        help="Do not search common filesystem locations for ComfyUI",
    )
    setup_parser.add_argument(
        "--offline",
        action="store_true",
        help="Write a draft without contacting ComfyUI",
    )
    setup_parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Never prompt for model selection",
    )
    setup_parser.add_argument("--json", action="store_true")
    setup_parser.set_defaults(func=command_setup)

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Validate the local profile, server, nodes, and models",
    )
    add_common_config_argument(doctor_parser)
    doctor_parser.add_argument("--json", action="store_true")
    doctor_parser.add_argument("--require-cuda", action="store_true")
    doctor_parser.set_defaults(func=command_doctor)

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Print nodes and detected bindings for a workflow",
    )
    inspect_parser.add_argument("workflow")
    inspect_parser.add_argument("--server", help="Optional server for model discovery")
    inspect_parser.add_argument("--timeout", type=float, default=30)
    inspect_parser.set_defaults(func=command_inspect)

    run_parser = subparsers.add_parser(
        "run",
        help="Submit, wait for, and fetch a configured workflow",
    )
    add_submission_arguments(run_parser)
    run_parser.add_argument("--out", help="Output file or directory")
    run_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an explicit output path",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and validate without network writes",
    )
    run_parser.add_argument("--graph-out", help="Write resolved graph to this path")
    run_parser.add_argument("--json", action="store_true")
    run_parser.set_defaults(func=command_run)

    submit_parser = subparsers.add_parser(
        "submit",
        help="Validate, upload references, and submit a workflow",
    )
    add_submission_arguments(submit_parser)
    submit_parser.add_argument("--json", action="store_true")
    submit_parser.set_defaults(func=command_submit)

    status_parser = subparsers.add_parser(
        "status",
        help="Query a submitted job without resubmitting it",
    )
    add_common_config_argument(status_parser)
    add_job_argument(status_parser)
    status_parser.add_argument("--json", action="store_true")
    status_parser.set_defaults(func=command_status)

    wait_parser = subparsers.add_parser(
        "wait",
        help="Wait for a previously submitted job",
    )
    add_common_config_argument(wait_parser)
    add_job_argument(wait_parser)
    wait_parser.add_argument("--json", action="store_true")
    wait_parser.set_defaults(func=command_wait)

    fetch_parser = subparsers.add_parser(
        "fetch",
        help="Download outputs for a completed job",
    )
    add_common_config_argument(fetch_parser)
    add_job_argument(fetch_parser)
    fetch_parser.add_argument("--out", help="Output file or directory")
    fetch_parser.add_argument("--overwrite", action="store_true")
    fetch_parser.add_argument("--json", action="store_true")
    fetch_parser.set_defaults(func=command_fetch)
    return parser


def error_payload(exc: ToolError, command: str | None) -> dict[str, Any]:
    details = dict(exc.details)
    details.setdefault("command", command)
    return result_payload(
        ok=False,
        state=exc.state,
        error={
            "code": exc.code,
            "message": str(exc),
            "retryable": exc.retryable,
            **details,
        },
        prompt_id=details.get("prompt_id"),
        files=[],
        warnings=[],
        job_id=details.get("job_id"),
        manifest=details.get("manifest"),
    )


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    json_mode = "--json" in raw_argv
    parser = build_parser()
    args = parser.parse_args(raw_argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        if json_mode:
            print(
                json.dumps(
                    result_payload(
                        ok=False,
                        state="interrupted",
                        error={
                            "code": "INTERRUPTED",
                            "message": "Interrupted by user",
                            "retryable": False,
                        },
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            eprint("Interrupted.")
        return EXIT_INTERRUPTED
    except ToolError as exc:
        payload = error_payload(exc, getattr(args, "command", None))
        if json_mode:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            eprint(f"ERROR [{exc.code}]: {exc}")
        return exc.exit_code
    except (ValueError, TypeError, OSError) as exc:
        converted = ToolError(
            f"{type(exc).__name__}: {exc}",
            code="UNEXPECTED_INPUT_ERROR",
        )
        payload = error_payload(converted, getattr(args, "command", None))
        if json_mode:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            eprint(f"ERROR [{converted.code}]: {converted}")
        return converted.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
