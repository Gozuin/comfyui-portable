from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import threading
import unittest
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


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
}


class MockComfyState:
    def __init__(self) -> None:
        self.last_graph: dict | None = None
        self.uploads: list[str] = []


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
            self.send_json(OBJECT_INFO)
            return
        if parsed.path == "/history/pid-1":
            self.send_json(
                {
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
            )
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
        if self.path == "/free":
            self.send_json({})
            return
        if self.path == "/prompt":
            payload = json.loads(body.decode("utf-8"))
            self.server.state.last_graph = payload["prompt"]
            self.send_json({"prompt_id": "pid-1", "number": 1})
            return
        if self.path == "/upload/image":
            text = body.decode("utf-8", errors="replace")
            marker = 'filename="'
            start = text.find(marker)
            filename = "reference.png"
            if start >= 0:
                start += len(marker)
                end = text.find('"', start)
                filename = text[start:end]
            self.server.state.uploads.append(filename)
            self.send_json({"name": filename, "subfolder": "", "type": "input"})
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
        self.server.state.last_graph = None
        self.server.state.uploads = []

    def run_cli(self, *arguments: str) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exit_code = comfyui_portable.main(list(arguments))
        return exit_code, stdout.getvalue(), stderr.getvalue()

    def test_inspect_detects_common_txt2img_bindings(self) -> None:
        graph = comfyui_portable.load_graph(ROOT / "examples" / "txt2img.api.json")
        result = comfyui_portable.inspect_workflow(
            graph,
            comfyui_portable.model_options_from_object_info(OBJECT_INFO),
        )
        self.assertIn("prompt", result["bindings"])
        self.assertIn("negative", result["bindings"])
        self.assertIn("seed", result["bindings"])
        self.assertIn("checkpoint", result["models"])
        self.assertEqual(result["defaults"]["steps"], 20)

    def test_inspect_handles_same_node_positive_and_negative(self) -> None:
        graph = {
            "1": {
                "class_type": "UNETLoader",
                "inputs": {"unet_name": "model.safetensors"},
            },
            "2": {
                "class_type": "CLIPLoader",
                "inputs": {"clip_name": "encoder.safetensors", "type": "qwen_image"},
            },
            "3": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": "vae.safetensors"},
            },
            "4": {
                "class_type": "TextEncodeQwenImage21",
                "inputs": {
                    "clip": ["2", 0],
                    "vae": ["3", 0],
                    "prompt": "positive",
                    "negative_prompt": "negative",
                    "resolution": 1024,
                },
            },
            "5": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": 1,
                    "steps": 20,
                    "cfg": 1.0,
                    "denoise": 1.0,
                    "model": ["1", 0],
                    "positive": ["4", 0],
                    "negative": ["4", 1],
                    "latent_image": ["4", 2],
                },
            },
            "6": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["5", 0], "vae": ["3", 0]},
            },
            "7": {
                "class_type": "SaveImage",
                "inputs": {"filename_prefix": "portable", "images": ["6", 0]},
            },
        }
        result = comfyui_portable.inspect_workflow(graph)
        self.assertEqual(
            result["bindings"]["prompt"], {"node": "4", "input": "prompt"}
        )
        self.assertEqual(
            result["bindings"]["negative"],
            {"node": "4", "input": "negative_prompt"},
        )
        self.assertEqual(
            result["bindings"]["resolution"], {"node": "4", "input": "resolution"}
        )

    def test_setup_doctor_and_run_txt2img(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            profile = base / "config.local.json"
            output = base / "result.png"
            exit_code, _, _ = self.run_cli(
                "setup",
                "--config",
                str(profile),
                "--server",
                self.url,
                "--workflow",
                f"txt2img={ROOT / 'examples' / 'txt2img.api.json'}",
                "--descriptor",
                f"txt2img={ROOT / 'examples' / 'txt2img.descriptor.json'}",
                "--model",
                "txt2img.checkpoint=model-a.safetensors",
                "--offline",
                "--non-interactive",
            )
            self.assertEqual(exit_code, 0)
            config = json.loads(profile.read_text(encoding="utf-8"))
            self.assertEqual(
                config["workflows"]["txt2img"]["models"]["checkpoint"]["value"],
                "model-a.safetensors",
            )

            exit_code, _, _ = self.run_cli(
                "doctor", "--config", str(profile), "--json"
            )
            self.assertEqual(exit_code, 0)

            exit_code, _, _ = self.run_cli(
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
                str(output),
            )
            self.assertEqual(exit_code, 0)
            self.assertEqual(output.read_bytes(), b"\x89PNG\r\n\x1a\nmock")
            graph = self.server.state.last_graph
            self.assertEqual(graph["2"]["inputs"]["text"], "a red apple")
            self.assertEqual(graph["3"]["inputs"]["text"], "blurry")
            self.assertEqual(graph["4"]["inputs"]["steps"], 24)

    def test_run_uploads_img2img_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            profile = base / "config.local.json"
            reference = base / "reference.png"
            output = base / "edited.png"
            reference.write_bytes(b"\x89PNG\r\n\x1a\nreference")
            exit_code, _, _ = self.run_cli(
                "setup",
                "--config",
                str(profile),
                "--server",
                self.url,
                "--workflow",
                f"img2img={ROOT / 'examples' / 'img2img.api.json'}",
                "--descriptor",
                f"img2img={ROOT / 'examples' / 'img2img.descriptor.json'}",
                "--model",
                "img2img.checkpoint=model-b.safetensors",
                "--offline",
                "--non-interactive",
            )
            self.assertEqual(exit_code, 0)
            exit_code, _, _ = self.run_cli(
                "run",
                "--config",
                str(profile),
                "--workflow",
                "img2img",
                "--prompt",
                "replace the background",
                "--reference",
                str(reference),
                "--out",
                str(output),
            )
            self.assertEqual(exit_code, 0)
            self.assertEqual(self.server.state.uploads, ["reference.png"])
            self.assertEqual(output.read_bytes(), b"\x89PNG\r\n\x1a\nmock")
            self.assertEqual(
                self.server.state.last_graph["2"]["inputs"]["image"],
                "reference.png",
            )

    def test_doctor_rejects_missing_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            profile = base / "config.local.json"
            exit_code, _, _ = self.run_cli(
                "setup",
                "--config",
                str(profile),
                "--server",
                self.url,
                "--workflow",
                f"txt2img={ROOT / 'examples' / 'txt2img.api.json'}",
                "--descriptor",
                f"txt2img={ROOT / 'examples' / 'txt2img.descriptor.json'}",
                "--model",
                "txt2img.checkpoint=missing.safetensors",
                "--offline",
                "--non-interactive",
            )
            self.assertEqual(exit_code, 0)
            exit_code, _, _ = self.run_cli(
                "doctor", "--config", str(profile), "--json"
            )
            self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
