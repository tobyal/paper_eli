"""Execute a UTF-8 shell script over the configured SSH connection."""
import argparse
import subprocess
from pathlib import Path

ROOT = '/mnt/sda/ubuntu/lkx/papers/paper_eli'
parser = argparse.ArgumentParser()
parser.add_argument('script', type=Path)
args = parser.parse_args()
text = args.script.read_text(encoding='utf-8-sig').replace('\r\n', '\n')
prefix = f'cd {ROOT}\nexport http_proxy=http://127.0.0.1:17891 https_proxy=http://127.0.0.1:17891 HTTP_PROXY=http://127.0.0.1:17891 HTTPS_PROXY=http://127.0.0.1:17891\nexport no_proxy=localhost,127.0.0.1,::1 NO_PROXY=localhost,127.0.0.1,::1\n'
result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ClearAllForwardings=yes', 'ubuntuA6000', 'bash -s'], input=(prefix + text).encode('utf-8'))
raise SystemExit(result.returncode)
