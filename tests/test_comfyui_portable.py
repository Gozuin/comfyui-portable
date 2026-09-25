from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import socket
import sys
import tempfile
import threading
import unittest
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "comfyui_portable.py"
spec = importlib.util.spec_from_file_location("comfyui_portable", MODULE_PATH)
assert spec and spec.loader
comfyui_portable = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = comfyui_portable
spec.loader.exec_module(comfyui_portable)


OBJECT_INFO = {
    "CheckpointLoaderSimple": {
        "input": {
            "required": {
                "ckpt_name": [["model-a.safetensors", "model-b.safetensors"], {}]
            }
        }
    },
    "CLIPTextEncode": {
        "input": {
            "required": {
                "text": ["STRING", {}],
                "clip": ["CLIP", {}],
            }
        }
    },
    "KSampler": {
        "input": {
            "required": {
                "seed": ["INT", {}],
                "steps": ["INT", {}],
                "cfg": ["FLOAT", {}],
                "sampler_name": [["euler", "dpmpp_2m"], {}],
                "scheduler": [["normal", "karras"], {}],
                "denoise": ["FLOAT", {}],
                "model": ["MODEL", {}],
                "positive": ["CONDITIONING", {}],
                "negative": ["CONDITIONING", {}],
                "latent_image": ["LATENT", {}],
            }
        }
    },
    "EmptyLatentImage": {
        "input": {
            "required": {
                "width": ["INT", {}],
                "height": ["INT", {}],
                "batch_size": ["INT", {}],
            }
        }
    },
    "LoadImage": {
        "input": {
            "required": {
                "image": ["STRING", {}],
            }
        }
    },
    "VAEEncode": {
        "input": {
            "required": {
                "pixels": ["IMAGE", {}],
                "vae": ["VAE", {}],
            }
        }
    },
    "VAEDecode": {
        "input": {
            "required": {
                "samples": ["LATENT", {}],
                "vae": ["VAE", {}],
            }
        }
    },
    "SaveImage": {
        "input": {
            "required": {
                "filename_prefix": ["STRING", {}],
                "images": ["IMAGE", {}],
            }
        }
    },
    "UNETLoader": {
        "input": {
            "required": {
                "unet_name": [["unet.safetensors"], {}],
            }
        }
    },
    "CLIPLoader": {
        "input": {
            "required": {
                "clip_name": [["clip.safetensors"], {}],
                "type": [["qwen_image", "sdxl"], {}],
            }
        }
    },
    "VAELoader": {
        "input": {
            "required": {
                "vae_name": [["vae.safetensors"], {}],
            }
        }
    },
}


class MockComfyState:
    def __init__(self) -> None:
        self.object_info = copy.deepcopy(OBJECT_INFO)
        self.last_graph: dict | None = None
        self.uploads: list[dict] = []
        self.requests: list[str] = []
        self.prompt_response: dict = {"prompt_id": "pid-1", "number": 1}
        self.history: dict = {}
        self.queue: dict = {"pending": [], "running": []}
        self.close_on_prompt = False
        self.free_calls = 0
        self.prompt_calls = 0
        self.reset_defaults()

    def reset_defaults(self) -> None:
        self.history = {
            "pid-1": {
                "status": {"status_str": "success", "completed": True},
                "outputs": {
                    "7": {
                        "images": [
                            {
                                "filename": "result.png",
                                "subfolder": "",
                                "type": "output",
                            }
                        ]
                    }
                },
            }
        }
        self.queue = {"pending": [], "running": []}


class MockComfyHandler(BaseHTTPRequestHandler):
    server_version = "MockComfy/1.0"

    def log_message(self, format: str, *args: object) -> None:
        return

    def send_json(self, value: object, status: int = 200) -> None:
        payload = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        self.server.state.requests.append(f"GET {parsed.path}")
        if parsed.path == "/system_stats":
            self.send_json(
                {
                    "devices": [
                        {"name": "Mock CUDA", "type": "cuda"},
                    ]
                }
            )
            return
        if parsed.path == "/object_info":
            self.send_json(self.server.state.object_info)
            return
        if parsed.path.startswith("/history/"):
            prompt_id = urllib.parse.unquote(parsed.path.rsplit("/", 1)[-1])
            item = self.server.state.history.get(prompt_id)
            self.send_json({prompt_id: item} if item is not None else {})
            return
        if parsed.path == "/queue":
            self.send_json(self.server.state.queue)
            return
        if parsed.path == "/view":
            payload = b"\x89PNG\r\n\x1a\nmock"
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        self.server.state.requests.append(f"POST {self.path}")
        if self.path == "/free":
            self.server.state.free_calls += 1
            self.send_json({})
            return
        if self.path == "/prompt":
            if self.server.state.close_on_prompt:
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
                return
            payload = json.loads(body.decode("utf-8"))
            self.server.state.prompt_calls += 1
            self.server.state.last_graph = payload["prompt"]
            self.send_json(self.server.state.prompt_response)
            return
        if self.path == "/upload/image":
            text = body.decode("utf-8", errors="replace")
            fields: dict[str, str] = {}
            for name in ("type", "overwrite", "subfolder"):
                marker = f'name="{name}"'
                start = text.find(marker)
                if start >= 0:
                    start = text.find("\r\n\r\n", start) + 4
                    end = text.find("\r\n", start)
                    fields[name] = text[start:end]
            filename_marker = 'filename="'
            start = text.find(filename_marker)
            filename = "reference.png"
            if start >= 0:
                start += len(filename_marker)
                end = text.find('"', start)
                filename = text[start:end]
            fields["filename"] = filename
            self.server.state.uploads.append(fields)
            self.send_json(
                {
                    "name": filename,
                    "subfolder": fields.get("subfolder", ""),
                    "type": "input",
                }
            )
            return
        self.send_json({"error": "not found"}, status=404)


class ComfyuiPortableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), MockComfyHandler)
        cls.server.state = MockComfyState()
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def setUp(self) -> None:
        self.server.state = MockComfyState()

    def run_cli(self, *arguments: str) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exit_code = comfyui_portable.main(list(arguments))
        return exit_code, stdout.getvalue(), stderr.getvalue()

    def setup_profile(
        self,
        base: Path,
        *,
        workflow: str = "txt2img",
        workflow_path: Path | None = None,
        descriptor_path: Path | None = None,
        model: str | None = "model-a.safetensors",
        offline: bool = False,
        extra: list[str] | None = None,
    ) -> tuple[int, str, str, Path]:
        profile = base / "config.local.json"
        selected_workflow = workflow_path or ROOT / "examples" / f"{workflow}.api.json"
        selected_descriptor = (
            descriptor_path or ROOT / "examples" / f"{workflow}.descriptor.json"
        )
        arguments = [
            "setup",
            "--config",
            str(profile),
            "--server",
            self.url,
            "--workflow",
            f"{workflow}={selected_workflow}",
            "--descriptor",
            f"{workflow}={selected_descriptor}",
            "--non-interactive",
            "--poll-interval",
            "0.01",
            "--max-wait",
            "5",
            "--json",
        ]
        if model is not None:
            arguments.extend(["--model", f"{workflow}.checkpoint={model}"])
        if offline:
            arguments.append("--offline")
        if extra:
            arguments.extend(extra)
        code, stdout, stderr = self.run_cli(*arguments)
        return code, stdout, stderr, profile

    def write_mock_server_failure(self) -> None:
        self.server.state.history = {}
        self.server.state.queue = {"pending": [["1", "pid-1"]], "running": []}

    def test_setup_doctor_and_run_json_are_machine_clean(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            setup_code, setup_out, _, profile = self.setup_profile(base)
            self.assertEqual(setup_code, 0)
            setup_payload = json.loads(setup_out)
            self.assertEqual(setup_payload["state"], "ready")
            self.assertEqual(self.server.state.free_calls, 0)

            code, stdout, stderr = self.run_cli(
                "doctor",
                "--config",
                str(profile),
                "--json",
            )
            self.assertEqual(code, 0, stderr)
            self.assertTrue(json.loads(stdout)["ok"])

            code, stdout, stderr = self.run_cli(
                "run",
                "--config",
                str(profile),
                "--workflow",
                "txt2img",
                "--prompt",
                "a red apple",
                "--negative",
                "blurry",
                "--steps",
                "24",
                "--out",
                str(base / "result.png"),
                "--json",
            )
            self.assertEqual(code, 0, stderr)
            payload = json.loads(stdout)
            self.assertEqual(payload["state"], "completed")
            self.assertEqual(payload["prompt_id"], "pid-1")
            self.assertEqual(self.server.state.prompt_calls, 1)
            self.assertEqual(self.server.state.free_calls, 0)
            graph = self.server.state.last_graph
            self.assertEqual(graph["2"]["inputs"]["text"], "a red apple")
            self.assertEqual(graph["3"]["inputs"]["text"], "blurry")
            self.assertEqual(graph["4"]["inputs"]["steps"], 24)

    def test_dry_run_has_zero_remote_side_effects(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            reference = base / "reference.png"
            reference.write_bytes(b"\x89PNG\r\n\x1a\nreference")
            setup_code, _, _, profile = self.setup_profile(
                base,
                workflow="img2img",
                model="model-b.safetensors",
            )
            self.assertEqual(setup_code, 0)
            self.server.state.uploads = []

            code, stdout, stderr = self.run_cli(
                "run",
                "--config",
                str(profile),
                "--workflow",
                "img2img",
                "--prompt",
                "replace the background",
                "--reference",
                str(reference),
                "--dry-run",
                "--json",
            )
            self.assertEqual(code, 0, stderr)
            payload = json.loads(stdout)
            self.assertEqual(payload["state"], "planned")
            self.assertEqual(payload["uploads"][0]["source"], str(reference.resolve()))
            self.assertEqual(self.server.state.uploads, [])
            self.assertEqual(self.server.state.prompt_calls, 0)
            self.assertEqual(self.server.state.free_calls, 0)

    def test_uploads_are_namespaced_unique_and_never_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            graph = comfyui_portable.load_graph(ROOT / "examples" / "img2img.api.json")
            graph["8"] = {
                "class_type": "LoadImage",
                "inputs": {"image": "second.png"},
            }
            graph_path = base / "two-references.api.json"
            graph_path.write_text(json.dumps(graph), encoding="utf-8")
            descriptor = json.loads(
                (ROOT / "examples" / "img2img.descriptor.json").read_text(
                    encoding="utf-8"
                )
            )
            descriptor["bindings"]["reference"] = [
                {"node": "2", "input": "image"},
                {"node": "8", "input": "image"},
            ]
            descriptor_path = base / "two-references.descriptor.json"
            descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")
            first_dir = base / "first"
            second_dir = base / "second"
            first_dir.mkdir()
            second_dir.mkdir()
            first = first_dir / "same.png"
            second = second_dir / "same.png"
            first.write_bytes(b"\x89PNG\r\n\x1a\none")
            second.write_bytes(b"\x89PNG\r\n\x1a\ntwo")

            setup_code, _, _, profile = self.setup_profile(
                base,
                workflow="two",
                workflow_path=graph_path,
                descriptor_path=descriptor_path,
                model="model-b.safetensors",
            )
            self.assertEqual(setup_code, 0)
            self.server.state.uploads = []

            code, stdout, stderr = self.run_cli(
                "submit",
                "--config",
                str(profile),
                "--workflow",
                "two",
                "--prompt",
                "edit",
                "--reference",
                str(first),
                "--reference",
                str(second),
                "--job-id",
                "job-uploads",
                "--json",
            )
            self.assertEqual(code, 0, stderr)
            payload = json.loads(stdout)
            self.assertEqual(payload["state"], "accepted")
            uploads = self.server.state.uploads
            self.assertEqual(len(uploads), 2)
            self.assertNotEqual(uploads[0]["filename"], uploads[1]["filename"])
            self.assertEqual(uploads[0]["overwrite"], "false")
            self.assertEqual(uploads[0]["subfolder"], "comfyui-portable/job-uploads")
            self.assertEqual(
                self.server.state.last_graph["2"]["inputs"]["image"],
                f"{uploads[0]['subfolder']}/{uploads[0]['filename']}",
            )
            self.assertEqual(
                self.server.state.last_graph["8"]["inputs"]["image"],
                f"{uploads[1]['subfolder']}/{uploads[1]['filename']}",
            )

    def test_noninteractive_missing_model_writes_draft_and_run_refuses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            setup_code, setup_out, _, profile = self.setup_profile(base, model=None)
            self.assertEqual(setup_code, 2)
            setup_payload = json.loads(setup_out)
            self.assertEqual(setup_payload["state"], "draft")
            self.assertEqual(
                setup_payload["error"]["code"],
                "NEEDS_MODEL_SELECTION",
            )
            config = json.loads(profile.read_text(encoding="utf-8"))
            self.assertEqual(config["validation_state"], "draft")

            code, stdout, _ = self.run_cli(
                "run",
                "--config",
                str(profile),
                "--workflow",
                "txt2img",
                "--prompt",
                "test",
                "--json",
            )
            self.assertEqual(code, comfyui_portable.EXIT_CONFIG)
            self.assertEqual(json.loads(stdout)["error"]["code"], "PROFILE_NOT_READY")

    def test_empty_model_enum_is_not_treated_as_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            self.server.state.object_info["CheckpointLoaderSimple"]["input"][
                "required"
            ]["ckpt_name"] = [[], {}]
            code, stdout, _, _ = self.setup_profile(Path(temporary))
            self.assertEqual(code, comfyui_portable.EXIT_CONFIG)
            payload = json.loads(stdout)
            self.assertEqual(payload["state"], "draft")
            self.assertEqual(payload["error"]["code"], "NEEDS_MODEL_SELECTION")
            self.assertEqual(payload["error"]["candidates"], [])

    def test_descriptor_required_nodes_are_additive(self) -> None:
        graph = comfyui_portable.load_graph(ROOT / "examples" / "txt2img.api.json")
        automatic = comfyui_portable.inspect_workflow(graph, OBJECT_INFO)
        merged = comfyui_portable.merge_descriptor(
            automatic,
            {"required_nodes": ["KSampler"]},
        )
        self.assertIn("EmptyLatentImage", merged["required_nodes"])
        self.assertIn("KSampler", merged["required_nodes"])

    def test_binding_validation_rejects_strings_and_malformed_lists(self) -> None:
        graph = comfyui_portable.load_graph(ROOT / "examples" / "txt2img.api.json")
        with self.assertRaises(comfyui_portable.ToolError):
            comfyui_portable.validate_binding(graph, "2.text", "txt2img.prompt")
        with self.assertRaises(comfyui_portable.ToolError):
            comfyui_portable.validate_binding(
                graph,
                [{"node": "2", "input": "text"}, "invalid"],
                "txt2img.prompt",
            )

    def test_sdxl_dual_text_inputs_are_bound_as_channels(self) -> None:
        graph = {
            "1": {
                "class_type": "CLIPTextEncodeSDXL",
                "inputs": {
                    "text_g": "positive",
                    "text_l": "positive",
                    "clip": ["2", 0],
                },
            },
            "2": {
                "class_type": "CLIPTextEncodeSDXL",
                "inputs": {
                    "text_g": "negative",
                    "text_l": "negative",
                    "clip": ["3", 0],
                },
            },
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": 1,
                    "steps": 20,
                    "positive": ["1", 0],
                    "negative": ["2", 0],
                    "latent_image": ["4", 0],
                },
            },
            "4": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": 1024, "height": 1024},
            },
        }
        result = comfyui_portable.inspect_workflow(graph)
        self.assertEqual(
            result["bindings"]["prompt"],
            [
                {"node": "1", "input": "text_g"},
                {"node": "1", "input": "text_l"},
            ],
        )
        self.assertEqual(
            result["bindings"]["negative"],
            [
                {"node": "2", "input": "text_g"},
                {"node": "2", "input": "text_l"},
            ],
        )

    def test_rebound_parameter_uses_new_target_default(self) -> None:
        graph = {
            "1": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": 1,
                    "steps": 20,
                    "sampler_name": "euler",
                },
            },
            "2": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": 2,
                    "steps": 40,
                    "sampler_name": "euler",
                },
            },
        }
        automatic = comfyui_portable.inspect_workflow(graph)
        merged = comfyui_portable.merge_descriptor(
            automatic,
            {"bindings": {"steps": {"node": "2", "input": "steps"}}},
        )
        defaults = comfyui_portable.derive_defaults(graph, merged["bindings"])
        self.assertEqual(defaults["steps"], 40)

    def test_linked_input_is_not_overwritten_by_parameter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            graph = comfyui_portable.load_graph(ROOT / "examples" / "txt2img.api.json")
            graph["9"] = {
                "class_type": "PrimitiveInt",
                "inputs": {"value": 12},
            }
            graph["4"]["inputs"]["steps"] = ["9", 0]
            graph_path = base / "linked.api.json"
            graph_path.write_text(json.dumps(graph), encoding="utf-8")
            descriptor_path = ROOT / "examples" / "txt2img.descriptor.json"
            setup_code, _, _, profile = self.setup_profile(
                base,
                workflow="linked",
                workflow_path=graph_path,
                descriptor_path=descriptor_path,
                offline=True,
            )
            self.assertEqual(setup_code, 0)
            code, stdout, _ = self.run_cli(
                "run",
                "--config",
                str(profile),
                "--workflow",
                "linked",
                "--prompt",
                "test",
                "--steps",
                "30",
                "--allow-draft",
                "--json",
            )
            self.assertEqual(code, comfyui_portable.EXIT_CONFIG)
            self.assertEqual(json.loads(stdout)["error"]["code"], "LINKED_INPUT")

    def test_string_enum_resolution_is_not_cast_to_integer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            graph = comfyui_portable.load_graph(ROOT / "examples" / "txt2img.api.json")
            graph["5"]["inputs"]["resolution"] = "1024x1024"
            graph_path = base / "resolution.api.json"
            graph_path.write_text(json.dumps(graph), encoding="utf-8")
            self.server.state.object_info["EmptyLatentImage"]["input"]["required"][
                "resolution"
            ] = [["1024x1024", "512x512"], {}]
            setup_code, _, _, profile = self.setup_profile(
                base,
                workflow="resolution",
                workflow_path=graph_path,
                descriptor_path=ROOT / "examples" / "txt2img.descriptor.json",
            )
            self.assertEqual(setup_code, 0)

            code, _, stderr = self.run_cli(
                "run",
                "--config",
                str(profile),
                "--workflow",
                "resolution",
                "--prompt",
                "test",
                "--resolution",
                "1024x1024",
                "--out",
                str(base / "result.png"),
                "--json",
            )
            self.assertEqual(code, 0, stderr)
            self.assertEqual(
                self.server.state.last_graph["5"]["inputs"]["resolution"],
                "1024x1024",
            )

    def test_prompt_id_is_kept_when_node_errors_are_partial(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            setup_code, _, _, profile = self.setup_profile(base)
            self.assertEqual(setup_code, 0)
            self.server.state.prompt_response = {
                "prompt_id": "pid-partial",
                "node_errors": {"8": {"errors": [{"message": "unused branch"}]}},
            }
            code, stdout, stderr = self.run_cli(
                "submit",
                "--config",
                str(profile),
                "--workflow",
                "txt2img",
                "--prompt",
                "test",
                "--job-id",
                "partial-job",
                "--json",
            )
            self.assertEqual(code, 0, stderr)
            payload = json.loads(stdout)
            self.assertEqual(payload["state"], "accepted_with_node_errors")
            self.assertEqual(payload["prompt_id"], "pid-partial")
            self.assertTrue(Path(payload["manifest"]).exists())

    def test_submit_transport_failure_is_unknown_and_not_retried(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            setup_code, _, _, profile = self.setup_profile(base)
            self.assertEqual(setup_code, 0)
            self.server.state.close_on_prompt = True
            code, stdout, _ = self.run_cli(
                "submit",
                "--config",
                str(profile),
                "--workflow",
                "txt2img",
                "--prompt",
                "test",
                "--job-id",
                "unknown-job",
                "--json",
            )
            self.assertEqual(code, comfyui_portable.EXIT_UNKNOWN)
            payload = json.loads(stdout)
            self.assertEqual(payload["error"]["code"], "SUBMISSION_UNKNOWN")
            manifest = base / ".comfyui-portable" / "jobs" / "unknown-job.json"
            self.assertEqual(
                json.loads(manifest.read_text(encoding="utf-8"))["state"],
                "unknown",
            )
            self.assertEqual(self.server.state.prompt_calls, 0)

    def test_status_wait_and_fetch_recover_the_same_job(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            setup_code, _, _, profile = self.setup_profile(base)
            self.assertEqual(setup_code, 0)
            code, submit_stdout, stderr = self.run_cli(
                "submit",
                "--config",
                str(profile),
                "--workflow",
                "txt2img",
                "--prompt",
                "test",
                "--job-id",
                "recoverable",
                "--json",
            )
            self.assertEqual(code, 0, stderr)
            self.assertEqual(json.loads(submit_stdout)["prompt_id"], "pid-1")

            code, status_stdout, stderr = self.run_cli(
                "status",
                "--config",
                str(profile),
                "--job",
                "recoverable",
                "--json",
            )
            self.assertEqual(code, 0, stderr)
            self.assertEqual(json.loads(status_stdout)["state"], "completed")

            code, wait_stdout, stderr = self.run_cli(
                "wait",
                "--config",
                str(profile),
                "--job",
                "recoverable",
                "--json",
            )
            self.assertEqual(code, 0, stderr)
            self.assertEqual(json.loads(wait_stdout)["state"], "completed")

            output_dir = base / "recovered"
            code, fetch_stdout, stderr = self.run_cli(
                "fetch",
                "--config",
                str(profile),
                "--job",
                "recoverable",
                "--out",
                str(output_dir),
                "--json",
            )
            self.assertEqual(code, 0, stderr)
            payload = json.loads(fetch_stdout)
            self.assertEqual(len(payload["files"]), 1)
            self.assertTrue(Path(payload["files"][0]).is_file())

    def test_output_destinations_do_not_collide_after_renaming(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            first = base / "7-1-result.png"
            first.write_bytes(b"existing")
            outputs = [
                {
                    "filename": "result.png",
                    "node_id": "7",
                    "output_index": 0,
                },
                {
                    "filename": "result-3.png",
                    "node_id": "7",
                    "output_index": 1,
                },
                {
                    "filename": "result.png",
                    "node_id": "7",
                    "output_index": 2,
                },
            ]
            destinations = comfyui_portable.output_destinations(
                str(base),
                "txt2img",
                outputs,
            )
            names = [path.name for path in destinations]
            self.assertEqual(len(names), len(set(names)))
            self.assertNotIn("7-1-result.png", names)
            self.assertEqual(len(destinations), 3)

    def test_model_detection_ignores_clip_loader_type_enum(self) -> None:
        graph = {
            "1": {
                "class_type": "CLIPLoader",
                "inputs": {
                    "clip_name": "clip.safetensors",
                    "type": "qwen_image",
                },
            }
        }
        result = comfyui_portable.inspect_workflow(graph, OBJECT_INFO)
        self.assertIn("clip", result["models"])
        self.assertNotIn("type", result["models"])
        self.assertEqual(
            result["models"]["clip"]["input"],
            "clip_name",
        )

    def test_descriptor_can_extend_a_custom_model_loader(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            self.server.state.object_info["CustomLoader"] = {
                "input": {
                    "required": {
                        "custom_name": [["custom-a.safetensors"], {}],
                    }
                }
            }
            graph = {
                "1": {
                    "class_type": "CustomLoader",
                    "inputs": {"custom_name": "custom-a.safetensors"},
                }
            }
            graph_path = base / "custom.api.json"
            graph_path.write_text(json.dumps(graph), encoding="utf-8")
            descriptor_path = base / "custom.descriptor.json"
            descriptor_path.write_text(
                json.dumps(
                    {
                        "models": {
                            "checkpoint": {
                                "node": "1",
                                "input": "custom_name",
                            }
                        },
                        "required_nodes": ["CustomLoader"],
                    }
                ),
                encoding="utf-8",
            )
            code, stdout, stderr, _ = self.setup_profile(
                base,
                workflow="custom",
                workflow_path=graph_path,
                descriptor_path=descriptor_path,
                model="custom-a.safetensors",
            )
            self.assertEqual(code, 0, stderr)
            self.assertEqual(json.loads(stdout)["state"], "ready")

    def test_descriptor_rejects_unconsumed_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            graph = {
                "1": {
                    "class_type": "CheckpointLoaderSimple",
                    "inputs": {"ckpt_name": "model-a.safetensors"},
                }
            }
            graph_path = base / "unused.api.json"
            graph_path.write_text(json.dumps(graph), encoding="utf-8")
            descriptor_path = base / "unused.descriptor.json"
            descriptor_path.write_text(
                json.dumps({"defaults": {"steps": 20}}),
                encoding="utf-8",
            )
            code, stdout, _, _ = self.setup_profile(
                base,
                workflow="unused",
                workflow_path=graph_path,
                descriptor_path=descriptor_path,
                model="model-a.safetensors",
                offline=True,
            )
            self.assertEqual(code, comfyui_portable.EXIT_CONFIG)
            self.assertEqual(
                json.loads(stdout)["error"]["code"],
                "UNCONSUMED_DEFAULT",
            )

    def test_json_error_is_parseable_and_uses_stable_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing.json"
            code, stdout, stderr = self.run_cli(
                "run",
                "--config",
                str(missing),
                "--workflow",
                "txt2img",
                "--prompt",
                "test",
                "--json",
            )
            self.assertEqual(code, comfyui_portable.EXIT_CONFIG)
            payload = json.loads(stdout)
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["error"]["code"], "PROFILE_NOT_FOUND")
            self.assertEqual(stderr, "")

    def test_download_failure_does_not_replace_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "result.png"
            destination.write_bytes(b"keep-me")
            output = {
                "filename": "result.png",
                "subfolder": "",
                "type": "output",
            }
            with (
                mock.patch.object(
                    comfyui_portable,
                    "http_request",
                    side_effect=comfyui_portable.TransportError("offline"),
                ),
                self.assertRaises(comfyui_portable.ToolError),
            ):
                comfyui_portable.download_output(
                    {"server": self.url},
                    output,
                    destination,
                    overwrite=True,
                )
            self.assertEqual(destination.read_bytes(), b"keep-me")


if __name__ == "__main__":
    unittest.main()
