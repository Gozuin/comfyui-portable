#!/usr/bin/env python3
"""Discover and run ComfyUI workflows through a machine-local profile."""

from __future__ import annotations

import argparse
import copy
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
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
SKILL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = SKILL_ROOT / "config.local.json"
DEFAULT_SERVER = "http://127.0.0.1:8188"

MODEL_INPUT_ROLES = {
    "ckpt_name": "checkpoint",
    "unet_name": "unet",
    "clip_name": "clip",
    "clip_name1": "clip_1",
    "clip_name2": "clip_2",
    "clip_name3": "clip_3",
    "vae_name": "vae",
    "lora_name": "lora",
    "control_net_name": "controlnet",
    "style_model_name": "style_model",
    "clip_vision_name": "clip_vision",
    "ipadapter_file": "ipadapter",
    "upscale_model_name": "upscale_model",
}

MODEL_INPUT_EXCLUSIONS = {
    "sampler_name",
    "scheduler",
    "device",
    "dtype",
    "weight_dtype",
    "filename",
    "filename_prefix",
    "image",
    "mask",
    "text",
    "prompt",
    "negative_prompt",
}

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

POSITIVE_TEXT_INPUTS = (
    "prompt",
    "text",
    "positive",
    "positive_prompt",
    "text_g",
)

NEGATIVE_TEXT_INPUTS = (
    "negative_prompt",
    "negative",
    "text_l",
    "text",
)


class ToolError(RuntimeError):
    """A user-facing command error."""


def eprint(*parts: Any) -> None:
    print(*parts, file=sys.stderr)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ToolError(f"JSON file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ToolError(f"Invalid JSON in {path}: {exc}") from exc


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def config_path(value: str | None) -> Path:
    configured = value or os.environ.get("COMFYUI_SKILL_CONFIG")
    if configured:
        return Path(configured).expanduser().resolve()
    return DEFAULT_CONFIG_PATH


def load_config(path: Path, required: bool = True) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise ToolError(
                f"Local profile not found: {path}\n"
                f"Run: python {Path(__file__).name} setup --help"
            )
        return {}
    value = read_json(path)
    if not isinstance(value, dict):
        raise ToolError(f"Local profile must contain a JSON object: {path}")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ToolError(
            f"Unsupported schema_version in {path}: "
            f"{value.get('schema_version')!r}; expected {SCHEMA_VERSION}"
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
    config: dict[str, Any], extra: dict[str, str] | None = None
) -> dict[str, str]:
    headers = {str(key): str(value) for key, value in (config.get("headers") or {}).items()}
    api_key = os.environ.get("COMFYUI_API_KEY")
    if api_key and not any(key.lower() == "authorization" for key in headers):
        headers["Authorization"] = f"Bearer {api_key}"
    if extra:
        headers.update(extra)
    return headers


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
            f"HTTP {exc.code} from {url}: {detail[:2000]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise ToolError(f"Cannot reach {url}: {exc.reason}") from exc

    if raw:
        return body
    if not body.strip():
        return {}
    try:
        return json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ToolError(f"Non-JSON response from {url}: {body[:500]!r}") from exc


def get_system_stats(config: dict[str, Any]) -> dict[str, Any]:
    timeout = float(config.get("timeout_seconds", 30))
    data = http_request(
        server_url(config) + "/system_stats",
        headers=merged_headers(config),
        timeout=timeout,
    )
    if not isinstance(data, dict):
        raise ToolError("/system_stats did not return a JSON object")
    return data


def get_object_info(config: dict[str, Any]) -> dict[str, Any]:
    timeout = float(config.get("timeout_seconds", 30))
    data = http_request(
        server_url(config) + "/object_info",
        headers=merged_headers(config),
        timeout=timeout,
    )
    if not isinstance(data, dict):
        raise ToolError("/object_info did not return a JSON object")
    return data


def option_values(spec: Any) -> list[str] | None:
    if isinstance(spec, list) and spec:
        values = spec[0]
        if isinstance(values, list) and all(isinstance(item, str) for item in values):
            return list(values)
    if isinstance(spec, dict):
        values = spec.get("options")
        if isinstance(values, list) and all(isinstance(item, str) for item in values):
            return list(values)
    return None


def model_options_from_object_info(
    object_info: dict[str, Any],
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for class_name, details in object_info.items():
        if not isinstance(details, dict):
            continue
        input_groups = details.get("input") or {}
        for group_name in ("required", "optional"):
            group = input_groups.get(group_name) or {}
            if not isinstance(group, dict):
                continue
            for input_name, spec in group.items():
                values = option_values(spec)
                if values:
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
        raise ToolError(f"Not a ComfyUI root: {explicit}")

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
        raise ToolError(f"Workflow must be a non-empty API-format JSON object: {path}")
    for node_id, node in graph.items():
        if not isinstance(node, dict) or "class_type" not in node:
            raise ToolError(
                f"Node {node_id!r} in {path} is not API format "
                "(expected class_type and inputs)"
            )
        if not isinstance(node.get("inputs"), dict):
            raise ToolError(f"Node {node_id!r} in {path} has no inputs object")
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


def model_role_for_input(
    class_name: str,
    input_name: str,
    model_options: dict[str, list[str]],
) -> str | None:
    if input_name in MODEL_INPUT_EXCLUSIONS:
        return None
    if f"{class_name}.{input_name}" in model_options:
        return MODEL_INPUT_ROLES.get(input_name, input_name.removesuffix("_name"))
    if input_name in MODEL_INPUT_ROLES:
        return MODEL_INPUT_ROLES[input_name]
    return None


def unique_key(base: str, used: set[str]) -> str:
    candidate = base
    index = 2
    while candidate in used:
        candidate = f"{base}_{index}"
        index += 1
    used.add(candidate)
    return candidate


def find_model_bindings(
    graph: dict[str, Any],
    model_options: dict[str, list[str]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    used: set[str] = set()
    for node_id in sorted_node_ids(graph):
        node = graph[node_id]
        class_name = node_class(node)
        for input_name, value in (node.get("inputs") or {}).items():
            if not isinstance(value, str):
                continue
            role = model_role_for_input(class_name, input_name, model_options)
            if not role:
                continue
            result[unique_key(role, used)] = {
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


def text_binding_for(
    graph: dict[str, Any],
    target_id: str,
    *,
    negative: bool,
    same_node_as_other: bool = False,
) -> dict[str, str] | None:
    inputs = (graph[target_id].get("inputs") or {})
    if negative and same_node_as_other:
        for key in ("negative_prompt", "negative"):
            if key in inputs:
                return {"node": target_id, "input": key}
    candidates = NEGATIVE_TEXT_INPUTS if negative else POSITIVE_TEXT_INPUTS
    for key in candidates:
        if key in inputs:
            return {"node": target_id, "input": key}
    return None


def inspect_workflow(
    graph: dict[str, Any],
    model_options: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    model_options = model_options or {}
    bindings: dict[str, Any] = {}
    defaults: dict[str, Any] = {}

    sampler_ids = [
        node_id
        for node_id in sorted_node_ids(graph)
        if is_sampler_node(
            node_class(graph[node_id]),
            graph[node_id].get("inputs") or {},
        )
    ]
    if sampler_ids:
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
            for key in candidates:
                if key in sampler_inputs:
                    bindings[role] = {"node": sampler_id, "input": key}
                    if role != "seed":
                        defaults[role] = sampler_inputs[key]
                    break

        positive_target = link_target(graph, sampler_inputs.get("positive"))
        negative_target = link_target(graph, sampler_inputs.get("negative"))
        if positive_target:
            positive_id = positive_target[0]
            same_node = bool(negative_target and negative_target[0] == positive_id)
            binding = text_binding_for(
                graph, positive_id, negative=False, same_node_as_other=same_node
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
            defaults["width"] = inputs["width"]
            defaults["height"] = inputs["height"]
            if "batch_size" in inputs:
                bindings["batch_size"] = {"node": node_id, "input": "batch_size"}
                defaults["batch_size"] = inputs["batch_size"]
            break

    if "resolution" not in bindings:
        for node_id in sorted_node_ids(graph):
            inputs = graph[node_id].get("inputs") or {}
            if "resolution" in inputs:
                bindings["resolution"] = {"node": node_id, "input": "resolution"}
                defaults["resolution"] = inputs["resolution"]
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

    return {
        "bindings": bindings,
        "models": find_model_bindings(graph, model_options),
        "defaults": defaults,
        "required_nodes": sorted(
            {node_class(node) for node in graph.values() if node_class(node)}
        ),
    }


def merge_descriptor(
    automatic: dict[str, Any],
    descriptor: dict[str, Any] | None,
) -> dict[str, Any]:
    merged = copy.deepcopy(automatic)
    if not descriptor:
        return merged
    for role, binding in (descriptor.get("bindings") or {}).items():
        merged["bindings"][role] = binding
    for role, binding in (descriptor.get("models") or {}).items():
        model_binding = dict(binding)
        model_binding.pop("required", None)
        merged["models"][role] = model_binding
    merged["defaults"].update(descriptor.get("defaults") or {})
    if descriptor.get("required_nodes"):
        merged["required_nodes"] = list(descriptor["required_nodes"])
    return merged


def binding_items(binding: Any) -> list[dict[str, Any]]:
    if isinstance(binding, list):
        return [item for item in binding if isinstance(item, dict)]
    if isinstance(binding, dict):
        return [binding]
    return []


def validate_binding(graph: dict[str, Any], binding: Any, label: str) -> None:
    for item in binding_items(binding):
        node_id = str(item.get("node", ""))
        input_name = item.get("input")
        if node_id not in graph:
            raise ToolError(f"{label} points to missing node {node_id!r}")
        if not isinstance(input_name, str) or input_name not in graph[node_id]["inputs"]:
            raise ToolError(
                f"{label} points to missing input {input_name!r} on node {node_id}"
            )


def set_node_input(
    graph: dict[str, Any],
    binding: dict[str, Any],
    value: Any,
) -> None:
    node_id = str(binding["node"])
    input_name = str(binding["input"])
    if node_id not in graph:
        raise ToolError(f"Binding points to missing node {node_id!r}")
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
            raise ToolError(f"{label} must use NAME=VALUE: {value!r}")
        key, item = value.split("=", 1)
        if not key or not item:
            raise ToolError(f"{label} must use NAME=VALUE: {value!r}")
        result[key] = item
    return result


def parse_scoped_values(values: list[str], label: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for value in values:
        if "=" not in value or "." not in value.split("=", 1)[0]:
            raise ToolError(
                f"{label} must use WORKFLOW.ROLE=VALUE: {value!r}"
            )
        scope, item = value.split("=", 1)
        workflow, key = scope.split(".", 1)
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
            raise ToolError(f"Workflow must use NAME=PATH: {value!r}")
        result[name] = Path(raw_path).expanduser().resolve()
    return result


def parse_descriptor_arguments(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ToolError(f"Descriptor must use NAME=PATH: {value!r}")
        name, raw_path = value.split("=", 1)
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
    print(f"\n{label}: {current!r} is not installed.")
    for index, option in enumerate(options, start=1):
        print(f"  {index}. {option}")
    while True:
        answer = input("Select a model number, or press Enter to keep the value: ").strip()
        if not answer:
            return str(current)
        try:
            selected = int(answer)
        except ValueError:
            print("Enter a number or press Enter.")
            continue
        if 1 <= selected <= len(options):
            return options[selected - 1]
        print("Selection is out of range.")


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
            "*.api.json files in the workflows directory."
        )

    result = []
    for name, path in explicit_workflows.items():
        if not path.is_file():
            raise ToolError(f"Workflow file not found: {path}")
        descriptor = explicit_descriptors.get(name) or default_descriptor_for(path)
        if name in explicit_descriptors and not explicit_descriptors[name].is_file():
            raise ToolError(f"Descriptor file not found: {explicit_descriptors[name]}")
        result.append((name, path, descriptor))
    return result


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
            raise ToolError(f"Not a ComfyUI root: {args.comfy_root}")
    elif not args.no_discovery:
        root = discover_comfy_root()

    python_path: Path | None = None
    if args.python:
        python_path = Path(args.python).expanduser().resolve()
        if not python_path.is_file():
            raise ToolError(f"Python executable not found: {python_path}")
    else:
        python_path = find_comfy_python(root)

    temporary_config: dict[str, Any] = {
        **existing,
        "schema_version": SCHEMA_VERSION,
        "server": server,
        "timeout_seconds": int(
            args.timeout or existing.get("timeout_seconds", 30)
        ),
        "poll_interval_seconds": float(
            args.poll_interval
            if args.poll_interval is not None
            else existing.get("poll_interval_seconds", 2)
        ),
        "max_wait_seconds": int(
            args.max_wait or existing.get("max_wait_seconds", 900)
        ),
        "headers": headers,
    }

    object_info: dict[str, Any] | None = None
    system_stats: dict[str, Any] | None = None
    if not args.offline:
        try:
            system_stats = get_system_stats(temporary_config)
            object_info = get_object_info(temporary_config)
        except ToolError as exc:
            eprint(f"WARN: server validation skipped: {exc}")

    model_options = (
        model_options_from_object_info(object_info) if object_info else {}
    )
    interactive = not args.non_interactive and sys.stdin.isatty()
    model_overrides = parse_scoped_values(args.model, "--model")
    binding_overrides = parse_scoped_values(args.binding, "--binding")
    default_overrides = parse_scoped_values(args.default, "--default")
    workflow_args = parse_workflow_arguments(args.workflow)
    descriptor_args = parse_descriptor_arguments(args.descriptor)
    workflows = workflows_to_configure(workflow_args, descriptor_args)

    configured_workflows: dict[str, Any] = dict(existing.get("workflows") or {})
    for name, workflow_path, descriptor_path in workflows:
        graph = load_graph(workflow_path)
        automatic = inspect_workflow(graph, model_options)
        descriptor = read_json(descriptor_path) if descriptor_path else None
        if descriptor is not None and not isinstance(descriptor, dict):
            raise ToolError(f"Descriptor must be a JSON object: {descriptor_path}")
        merged = merge_descriptor(automatic, descriptor)

        for role, raw_binding in binding_overrides.get(name, {}).items():
            if "." not in raw_binding:
                raise ToolError(
                    f"--binding {name}.{role} must use NODE.INPUT: {raw_binding!r}"
                )
            node_id, input_name = raw_binding.split(".", 1)
            merged["bindings"][role] = {"node": node_id, "input": input_name}

        final_models: dict[str, Any] = {}
        for role, binding in merged.get("models", {}).items():
            node_id = str(binding.get("node", ""))
            input_name = str(binding.get("input", ""))
            if node_id not in graph or input_name not in graph[node_id]["inputs"]:
                raise ToolError(
                    f"Model binding {name}.{role} points to missing "
                    f"{node_id}.{input_name}"
                )
            current = graph[node_id]["inputs"].get(input_name)
            desired = model_overrides.get(name, {}).get(role, current)
            class_name = node_class(graph[node_id])
            options = model_options.get(f"{class_name}.{input_name}", [])
            if desired and options and desired not in options:
                if interactive:
                    desired = prompt_model_choice(
                        f"{name}.{role}", desired, options
                    )
                else:
                    eprint(
                        f"WARN: {name}.{role}={desired!r} is not exposed by "
                        f"{class_name}.{input_name} on this server"
                    )
            final_models[role] = {
                "node": node_id,
                "input": input_name,
                "value": desired,
            }

        final_defaults = dict(merged.get("defaults") or {})
        for key, value in default_overrides.get(name, {}).items():
            final_defaults[key] = parse_cli_value(value)

        missing_overrides = (
            set(model_overrides.get(name, {}))
            - set(merged.get("models", {}))
        )
        if missing_overrides:
            raise ToolError(
                f"Unknown model bindings for {name}: "
                f"{', '.join(sorted(missing_overrides))}"
            )
        missing_binding = (
            set(binding_overrides.get(name, {}))
            - set(merged.get("bindings", {}))
        )
        if missing_binding:
            raise ToolError(
                f"Unknown bindings for {name}: "
                f"{', '.join(sorted(missing_binding))}"
            )

        configured_workflows[name] = {
            "file": store_path(workflow_path),
            "descriptor": store_path(descriptor_path) if descriptor_path else None,
            "bindings": merged.get("bindings") or {},
            "models": final_models,
            "defaults": final_defaults,
            "required_nodes": merged.get("required_nodes") or [],
        }

    config = {
        **temporary_config,
        "comfyui": {
            "root": str(root) if root else None,
            "python": str(python_path) if python_path else None,
            "input_dir": str(root / "input") if root else None,
            "output_dir": str(root / "output") if root else None,
        },
        "workflows": configured_workflows,
    }
    write_json(path, config)

    print(f"Profile written: {path}")
    print(f"Server: {server}")
    print(f"ComfyUI root: {root or 'not detected (HTTP-only mode)'}")
    print(f"Python: {python_path or 'not detected'}")
    if system_stats:
        devices = system_stats.get("devices") or []
        names = [
            str(device.get("name") or device.get("type") or "unknown")
            for device in devices
            if isinstance(device, dict)
        ]
        print(f"Devices: {', '.join(names) if names else 'reported, no names'}")
    else:
        print("Server validation: skipped")
    for name, definition in configured_workflows.items():
        model_count = len(definition.get("models") or {})
        binding_count = len(definition.get("bindings") or {})
        print(
            f"Workflow {name}: {model_count} model binding(s), "
            f"{binding_count} parameter binding(s)"
        )
    return 0


def add_check(
    checks: list[dict[str, str]],
    level: str,
    message: str,
) -> None:
    checks.append({"level": level, "message": message})


def command_doctor(args: argparse.Namespace) -> int:
    path = config_path(args.config)
    config = load_config(path)
    checks: list[dict[str, str]] = []
    add_check(checks, "PASS", f"Profile loaded: {path}")

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
            isinstance(device, dict)
            and str(device.get("type", "")).lower() == "cuda"
            for device in devices
        ):
            add_check(checks, "FAIL", "No CUDA device reported")
        object_info = get_object_info(config)
        add_check(checks, "PASS", f"Node registry: {len(object_info)} classes")
    except ToolError as exc:
        add_check(checks, "FAIL", str(exc))

    model_options = (
        model_options_from_object_info(object_info) if object_info else {}
    )
    workflows = config.get("workflows") or {}
    if not workflows:
        add_check(checks, "FAIL", "No workflows configured")

    for name, definition in workflows.items():
        try:
            graph_path = resolve_stored_path(definition["file"])
            graph = load_graph(graph_path)
        except (KeyError, ToolError) as exc:
            add_check(checks, "FAIL", f"{name}: {exc}")
            continue

        add_check(checks, "PASS", f"{name}: workflow graph loaded")
        for role, binding in (definition.get("bindings") or {}).items():
            try:
                validate_binding(graph, binding, f"{name}.{role}")
            except ToolError as exc:
                add_check(checks, "FAIL", str(exc))
        for role, binding in (definition.get("models") or {}).items():
            try:
                validate_binding(graph, binding, f"{name}.{role}")
            except ToolError as exc:
                add_check(checks, "FAIL", str(exc))
                continue
            class_name = node_class(graph[str(binding["node"])])
            option_key = f"{class_name}.{binding['input']}"
            value = binding.get("value")
            options = model_options.get(option_key)
            if options and value not in options:
                add_check(
                    checks,
                    "FAIL",
                    f"{name}.{role}: {value!r} is not installed for {option_key}",
                )
            elif options:
                add_check(checks, "PASS", f"{name}.{role}: {value}")
            else:
                add_check(
                    checks,
                    "WARN",
                    f"{name}.{role}: server exposes no option list for {option_key}",
                )

        if object_info:
            for class_name in definition.get("required_nodes") or []:
                if class_name in object_info:
                    add_check(checks, "PASS", f"{name}: node {class_name}")
                else:
                    add_check(
                        checks, "FAIL", f"{name}: missing node class {class_name}"
                    )

    failed = any(check["level"] == "FAIL" for check in checks)
    if args.json:
        print(
            json.dumps(
                {"ok": not failed, "checks": checks},
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        for check in checks:
            print(f"[{check['level']}] {check['message']}")
        print("Doctor result:", "FAIL" if failed else "PASS")
    return 1 if failed else 0


def command_inspect(args: argparse.Namespace) -> int:
    path = Path(args.workflow).expanduser().resolve()
    graph = load_graph(path)
    model_options: dict[str, list[str]] = {}
    if args.server:
        config = {"server": args.server, "timeout_seconds": args.timeout}
        model_options = model_options_from_object_info(get_object_info(config))
    automatic = inspect_workflow(graph, model_options)
    summary = {
        "workflow": str(path),
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
    return 0


def convert_parameter(role: str, value: Any) -> Any:
    if value is None:
        return None
    if role in {"seed", "steps", "width", "height", "batch_size", "resolution"}:
        return int(value)
    if role in {"cfg", "denoise"}:
        return float(value)
    return str(value)


def upload_reference(
    config: dict[str, Any],
    file_path: Path,
) -> str:
    if not file_path.is_file():
        raise ToolError(f"Reference file not found: {file_path}")

    boundary = "----comfyui-portable-" + uuid.uuid4().hex
    content_type = (
        mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    )
    file_bytes = file_path.read_bytes()
    chunks: list[bytes] = []

    def add_field(name: str, value: str) -> None:
        chunks.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n"
            ).encode("utf-8")
        )

    add_field("type", "input")
    add_field("overwrite", "true")
    escaped_filename = file_path.name.replace("\\", "\\\\").replace('"', '\\"')
    chunks.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; '
            f'filename="{escaped_filename}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode("utf-8")
    )
    chunks.append(file_bytes)
    chunks.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    payload = b"".join(chunks)

    result = http_request(
        server_url(config) + "/upload/image",
        method="POST",
        payload=payload,
        headers=merged_headers(
            config,
            {"Content-Type": f"multipart/form-data; boundary={boundary}"},
        ),
        timeout=float(config.get("timeout_seconds", 30)),
    )
    if not isinstance(result, dict) or not result.get("name"):
        raise ToolError(f"Upload response did not contain a filename: {result!r}")
    name = str(result["name"])
    subfolder = str(result.get("subfolder") or "").strip("/")
    return f"{subfolder}/{name}" if subfolder else name


def gather_outputs(history_item: dict[str, Any]) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for node_output in (history_item.get("outputs") or {}).values():
        if not isinstance(node_output, dict):
            continue
        for values in node_output.values():
            if not isinstance(values, list):
                continue
            for item in values:
                if isinstance(item, dict) and item.get("filename"):
                    outputs.append(item)
    return outputs


def download_output(
    config: dict[str, Any],
    output: dict[str, Any],
    destination: Path,
) -> None:
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
    destination.write_bytes(data)


def submit_prompt(graph: dict[str, Any], config: dict[str, Any]) -> str:
    client_id = str(uuid.uuid4())
    result = http_request(
        server_url(config) + "/prompt",
        method="POST",
        payload={"prompt": graph, "client_id": client_id},
        headers=merged_headers(config),
        timeout=float(config.get("timeout_seconds", 30)),
    )
    if not isinstance(result, dict):
        raise ToolError(f"Unexpected /prompt response: {result!r}")
    if result.get("node_errors"):
        raise ToolError(
            "ComfyUI rejected the graph: "
            + json.dumps(result["node_errors"], ensure_ascii=False)
        )
    prompt_id = result.get("prompt_id")
    if not prompt_id:
        raise ToolError(f"/prompt did not return prompt_id: {result!r}")
    return str(prompt_id)


def wait_for_outputs(
    prompt_id: str,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    timeout = float(config.get("timeout_seconds", 30))
    interval = max(0.2, float(config.get("poll_interval_seconds", 2)))
    deadline = time.monotonic() + float(config.get("max_wait_seconds", 900))
    while time.monotonic() < deadline:
        history = http_request(
            server_url(config) + "/history/" + urllib.parse.quote(prompt_id),
            headers=merged_headers(config),
            timeout=timeout,
        )
        if isinstance(history, dict) and prompt_id in history:
            item = history[prompt_id]
            status = (item.get("status") or {}) if isinstance(item, dict) else {}
            status_name = str(status.get("status_str") or "").lower()
            if status_name in {"error", "failed"}:
                raise ToolError(
                    "ComfyUI execution failed: "
                    + json.dumps(item, ensure_ascii=False)[:4000]
                )
            if status_name == "success":
                outputs = gather_outputs(item)
                if not outputs:
                    raise ToolError(
                        "ComfyUI completed successfully but returned no outputs: "
                        + json.dumps(item, ensure_ascii=False)[:2000]
                    )
                return outputs
        time.sleep(interval)
    raise ToolError(
        f"Timed out after {config.get('max_wait_seconds', 900)} seconds "
        f"waiting for {prompt_id}"
    )


def select_workflow(
    config: dict[str, Any],
    requested: str | None,
) -> tuple[str, dict[str, Any]]:
    workflows = config.get("workflows") or {}
    if requested:
        if requested not in workflows:
            raise ToolError(
                f"Unknown workflow {requested!r}. "
                f"Available: {', '.join(sorted(workflows)) or 'none'}"
            )
        return requested, workflows[requested]
    if len(workflows) == 1:
        name = next(iter(workflows))
        return name, workflows[name]
    raise ToolError(
        "Multiple workflows are configured; pass --workflow. "
        f"Available: {', '.join(sorted(workflows))}"
    )


def output_destinations(
    requested: str | None,
    workflow_name: str,
    outputs: list[dict[str, Any]],
) -> list[Path]:
    if requested:
        base = Path(requested).expanduser().resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = SKILL_ROOT / "outputs" / f"{workflow_name}-{timestamp}"

    if len(outputs) == 1 and base.suffix:
        return [base]
    if base.suffix:
        base = base.with_suffix("")
    base.mkdir(parents=True, exist_ok=True)
    destinations = []
    used: set[str] = set()
    for index, output in enumerate(outputs, start=1):
        filename = Path(str(output.get("filename") or f"output-{index}.png")).name
        candidate = filename
        if candidate in used:
            candidate = f"{Path(filename).stem}-{index}{Path(filename).suffix}"
        used.add(candidate)
        destinations.append(base / candidate)
    return destinations


def command_run(args: argparse.Namespace) -> int:
    path = config_path(args.config)
    config = load_config(path)
    workflow_name, definition = select_workflow(config, args.workflow)
    graph_path = resolve_stored_path(definition["file"])
    graph = load_graph(graph_path)
    bindings = definition.get("bindings") or {}

    for role, binding in (definition.get("models") or {}).items():
        validate_binding(graph, binding, f"{workflow_name}.{role}")
        set_node_input(graph, binding, binding.get("value"))

    parameters = {
        "prompt": args.prompt,
        "negative": args.negative,
        "seed": args.seed,
        "steps": args.steps,
        "cfg": args.cfg,
        "denoise": args.denoise,
        "sampler_name": args.sampler_name,
        "scheduler": args.scheduler,
        "width": args.width,
        "height": args.height,
        "batch_size": args.batch_size,
        "resolution": args.resolution,
    }
    for role, value in (definition.get("defaults") or {}).items():
        if role in PARAMETER_NAMES and parameters.get(role) is None:
            parameters[role] = value

    if "prompt" in bindings and parameters["prompt"] is None:
        raise ToolError("--prompt is required for this workflow")

    for role in sorted(PARAMETER_NAMES):
        value = parameters.get(role)
        if value is None:
            continue
        if role not in bindings:
            raise ToolError(
                f"--{role.replace('_', '-')} was provided, but workflow "
                f"{workflow_name!r} has no {role!r} binding"
            )
        validate_binding(graph, bindings[role], f"{workflow_name}.{role}")
        for binding in binding_items(bindings[role]):
            set_node_input(graph, binding, convert_parameter(role, value))

    reference_bindings = binding_items(bindings.get("reference"))
    if args.reference and not reference_bindings:
        raise ToolError(
            f"Workflow {workflow_name!r} has no reference input binding"
        )
    if len(args.reference) > len(reference_bindings):
        raise ToolError(
            f"Workflow {workflow_name!r} accepts at most "
            f"{len(reference_bindings)} reference file(s)"
        )
    for reference, binding in zip(args.reference, reference_bindings):
        uploaded_name = upload_reference(config, Path(reference).expanduser().resolve())
        set_node_input(graph, binding, uploaded_name)

    for override in args.set:
        if "." not in override or "=" not in override:
            raise ToolError(f"--set must use NODE.INPUT=VALUE: {override!r}")
        node_input, raw_value = override.split("=", 1)
        node_id, input_name = node_input.split(".", 1)
        if node_id not in graph or input_name not in graph[node_id]["inputs"]:
            raise ToolError(f"--set points to missing {node_id}.{input_name}")
        graph[node_id]["inputs"][input_name] = parse_cli_value(raw_value)

    if args.dry_run:
        if args.graph_out:
            destination = Path(args.graph_out).expanduser().resolve()
            write_json(destination, graph)
            print(f"Resolved graph written: {destination}")
        else:
            print(json.dumps(graph, ensure_ascii=False, indent=2))
        return 0

    try:
        http_request(
            server_url(config) + "/free",
            method="POST",
            payload={"unload_models": True, "free_memory": True},
            headers=merged_headers(config),
            timeout=float(config.get("timeout_seconds", 30)),
        )
    except ToolError as exc:
        eprint(f"WARN: VRAM free request skipped: {exc}")

    prompt_id = submit_prompt(graph, config)
    outputs = wait_for_outputs(prompt_id, config)
    destinations = output_destinations(args.out, workflow_name, outputs)
    for output, destination in zip(outputs, destinations):
        download_output(config, output, destination)
        print(f"Saved: {destination}")

    if args.json:
        print(
            json.dumps(
                {
                    "workflow": workflow_name,
                    "prompt_id": prompt_id,
                    "files": [str(item) for item in destinations],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0


def add_common_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        help="Local profile path. Defaults to config.local.json in the skill.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Discover and run ComfyUI workflows through a local profile."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup_parser = subparsers.add_parser(
        "setup", help="Detect the host and write config.local.json"
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
        "--poll-interval", type=float, help="History polling interval in seconds"
    )
    setup_parser.add_argument(
        "--max-wait", type=float, help="Maximum workflow wait in seconds"
    )
    setup_parser.add_argument(
        "--no-discovery",
        action="store_true",
        help="Do not search common filesystem locations for ComfyUI",
    )
    setup_parser.add_argument(
        "--offline",
        action="store_true",
        help="Do not contact ComfyUI while writing the profile",
    )
    setup_parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Never prompt for model selection",
    )
    setup_parser.set_defaults(func=command_setup)

    doctor_parser = subparsers.add_parser(
        "doctor", help="Validate the local profile, server, nodes, and models"
    )
    add_common_config_argument(doctor_parser)
    doctor_parser.add_argument("--json", action="store_true")
    doctor_parser.add_argument("--require-cuda", action="store_true")
    doctor_parser.set_defaults(func=command_doctor)

    inspect_parser = subparsers.add_parser(
        "inspect", help="Print nodes and detected bindings for a workflow"
    )
    inspect_parser.add_argument("workflow")
    inspect_parser.add_argument("--server", help="Optional server for model discovery")
    inspect_parser.add_argument("--timeout", type=float, default=30)
    inspect_parser.set_defaults(func=command_inspect)

    run_parser = subparsers.add_parser(
        "run", help="Resolve and submit a configured workflow"
    )
    add_common_config_argument(run_parser)
    run_parser.add_argument("--workflow", help="Configured workflow name")
    run_parser.add_argument("--prompt")
    run_parser.add_argument("--negative")
    run_parser.add_argument("--seed", type=int)
    run_parser.add_argument("--steps", type=int)
    run_parser.add_argument("--cfg", type=float)
    run_parser.add_argument("--denoise", type=float)
    run_parser.add_argument("--sampler-name")
    run_parser.add_argument("--scheduler")
    run_parser.add_argument("--width", type=int)
    run_parser.add_argument("--height", type=int)
    run_parser.add_argument("--batch-size", type=int)
    run_parser.add_argument("--resolution", type=int)
    run_parser.add_argument(
        "--reference",
        action="append",
        default=[],
        metavar="PATH",
        help="Reference image or video; repeat for multiple inputs",
    )
    run_parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NODE.INPUT=VALUE",
        help="Override any node input after semantic bindings",
    )
    run_parser.add_argument("--out", help="Output file or directory")
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the graph without submitting it",
    )
    run_parser.add_argument("--graph-out", help="Write resolved graph to this path")
    run_parser.add_argument("--json", action="store_true")
    run_parser.set_defaults(func=command_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        eprint("Interrupted.")
        return 130
    except ToolError as exc:
        eprint(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
