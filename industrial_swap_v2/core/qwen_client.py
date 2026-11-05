"""Qianwen DashScope multimodal editing; one durable POST, resumable downloads."""
import base64
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urlsplit
import requests
from PIL import Image
from .context import read_json, write_json, sha256
from .credentials import api_key

ENDPOINT = "https://maas.qianwenaiapi.com/api/v1/services/aigc/multimodal-generation/generation"
DOC = "https://platform.qianwenai.com/docs/api-reference/image-generation/qwen-image-editing"


def payload(config, images, prompt):
    if config["model"] != "qwen-image-2.1-pro":
        raise ValueError("This experiment fixes qwen-image-2.1-pro")
    if config["candidates"] != 1 or config["max_api_requests"] != 1:
        raise ValueError("User authorized one request producing one image")
    content = []
    for path in images:
        path = Path(path)
        assert path.stat().st_size <= 10 * 1024 * 1024
        with Image.open(path) as image:
            assert 1 <= min(image.size) and max(image.size) <= 8000
        content.append({"image": "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")})
    assert 1 <= len(images) <= 10
    content.append({"text": prompt})
    return {"model": config["model"], "input": {"messages": [{"role": "user", "content": content}]},
            "parameters": {"n": 1, "size": config["output_size"], "seed": config["seed"],
                           "prompt_extend": config["prompt_extend"], "watermark": False}}


def image_urls(response):
    urls = []
    for choice in response.get("output", {}).get("choices", []):
        if choice.get("finish_reason") not in (None, "stop"):
            raise RuntimeError("Image request did not finish normally")
        urls.extend(item["image"] for item in choice.get("message", {}).get("content", []) if "image" in item)
    if len(urls) != 1:
        raise RuntimeError(f"Expected exactly one generated image; received {len(urls)}")
    return urls


def generate(ctx, images, prompt, session=None):
    config = ctx.cfg["m2"]
    request = payload(config, images, prompt)
    digest = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    out = ctx.stage("m2")
    response_path = out / "api_response.json"
    request_path = out / "api_request.json"
    lock_path = ctx.records / "qwen_generation_once.lock"
    client = session or requests.Session()
    if response_path.exists():
        if read_json(request_path)["payload_sha256"] != digest:
            raise RuntimeError("Cached Qwen request differs; refusing a second API call")
        response = read_json(response_path)
    else:
        key = api_key(config.get("api_key_file"))
        endpoint = config.get("endpoint", ENDPOINT)
        if urlsplit(endpoint).scheme != "https":
            raise ValueError("API credentials require HTTPS")
        record = {"model": config["model"], "endpoint": endpoint, "documentation": DOC,
                  "parameters": request["parameters"], "prompt": prompt,
                  "reference_images": [{"path": str(p), "sha256": sha256(p)} for p in images],
                  "payload_sha256": digest, "authentication": "DASHSCOPE_API_KEY or explicitly configured protected file; value never saved",
                  "depth_role": "visual reference, not native ControlNet conditioning"}
        # An exclusive durable claim prevents retries, concurrent calls and a second
        # submission after a timeout, crash, HTTP failure or configuration change.
        try:
            with lock_path.open("x", encoding="utf-8") as stream:
                json.dump({"payload_sha256": digest, "submitted_at_unix": time.time(), "max_requests": 1}, stream)
        except FileExistsError as error:
            raise RuntimeError("One Qwen submission has already been claimed. No automatic resubmission is permitted; inspect saved response/status.") from error
        write_json(request_path, record)
        ctx.receipt("qwen_request", "submitted_once", payload_sha256=digest, requested_images=1)
        try:
            result = client.post(endpoint, headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
                                 json=request, timeout=(30, config["timeout_seconds"]), allow_redirects=False)
            if result.status_code != 200:
                try:
                    failure = result.json()
                except ValueError:
                    failure = {}
                # Keep only diagnostic identifiers; never log arbitrary response text.
                ctx.receipt("qwen_request", "http_error_no_retry", http_status=result.status_code,
                            api_error_code=failure.get("code"), request_id=failure.get("request_id"))
                raise RuntimeError(f"Qwen returned HTTP {result.status_code}; no second request made")
            response = result.json()
            write_json(response_path, response)
            image_urls(response)
            ctx.receipt("qwen_request", "response_saved", request_id=response.get("request_id"), usage=response.get("usage"))
        except (requests.RequestException, ValueError):
            ctx.receipt("qwen_request", "unknown_or_invalid_response_no_retry", note="Submission may have been processed; the durable claim remains")
            raise RuntimeError("Qwen response unavailable or invalid; automatic resubmission disabled") from None
    urls = image_urls(response)
    destination = out / "edited_scene.png"
    if not destination.exists():
        if urlsplit(urls[0]).scheme != "https":
            raise ValueError("Unexpected output image URL")
        # Separate unauthenticated GET. Resuming this download never submits generation.
        download = requests.get(urls[0], timeout=(30, 120)) if session is None else session.get(urls[0], timeout=(30, 120))
        download.raise_for_status()
        temporary = out / "edited_scene.download"
        temporary.write_bytes(download.content)
        with Image.open(temporary) as image:
            image.verify()
        temporary.replace(destination)
    with Image.open(destination) as image:
        actual_size = list(image.size)
    record = read_json(request_path)
    record.update(request_id=response.get("request_id"), usage=response.get("usage"),
                  output_sha256=sha256(destination), actual_size=actual_size,
                  successful_generation_requests=1, candidate_count=1,
                  submission_attempts=1+len(list(ctx.records.glob('qwen_attempt_*.lock'))))
    write_json(out / "generation.json", record)
    return destination
