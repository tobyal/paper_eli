"""Read only the explicitly configured credential; never log its value."""
import os
from pathlib import Path
import re
from urllib.parse import urlsplit


def api_key(path=None):
    value = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if value:
        return value
    if path and Path(path).is_file():
        text = Path(path).read_text(encoding="utf-8-sig")
        lines = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith('#')]
        if len(lines) != 1:
            raise ValueError('Configured file must contain exactly one key or DASHSCOPE_API_KEY assignment')
        value = lines[0]
        assignment = re.fullmatch(r'(?:export\s+)?DASHSCOPE_API_KEY\s*=\s*(.+)', value)
        if assignment:
            value = assignment.group(1).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('\"', "'"):
            value = value[1:-1]
        # Treat credentials as opaque. In particular, current Qianwen keys contain
        # dots; extracting an alphanumeric substring silently corrupts the key.
        if len(value) < 16 or not re.fullmatch(r'[A-Za-z0-9._+/=-]+', value) or value.startswith(('http:', 'https:')):
            raise ValueError('Key must be a single complete ASCII token; ambiguous files are rejected')
        return value
    raise RuntimeError("DASHSCOPE_API_KEY is not configured")


def inspect(path):
    text = Path(path).read_text(encoding="utf-8-sig")
    matches = set(re.findall(r"(?<![A-Za-z0-9._-])sk-[A-Za-z0-9._-]{16,}", text))
    return {"recognized_key_count": len(matches),
            "line_shapes": [{"length": len(line), "has_equals": "=" in line,
                             "has_colon": ":" in line,
                             "long_token_lengths": [len(x) for x in re.findall(r"[A-Za-z0-9_-]{20,}", line)]}
                            for line in text.splitlines()],
            "endpoint_hosts_in_file": [urlsplit(u).hostname for u in re.findall(r"https?://[^\s\"']+", text)],
            "environment_names": re.findall(r"\b([A-Z][A-Z0-9_]*KEY)\s*=", text)}
