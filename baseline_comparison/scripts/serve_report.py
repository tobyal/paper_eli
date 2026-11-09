"""Loopback-only read-only result server; no model, environment or secret access."""
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import unquote,urlsplit
from pathlib import Path
BASE=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli/baseline_comparison')
class Handler(SimpleHTTPRequestHandler):
 def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(BASE),**kwargs)
 def send_head(self):
  path=unquote(urlsplit(self.path).path)
  parts=Path(path).parts
  allowed=(path in ['/', '/prompts.json','/README.md'] or path.startswith('/comparison/') or
   (len(parts)>=4 and parts[1] in ['gaussianeditor','dge','gsle'] and parts[2] in ['case00','case02','case04','case07','case10'] and parts[3] in ['logs','records','scene_final.ply','task.json','config.json']) or
   (len(parts)==4 and parts[1]=='inputs' and parts[2] in ['case00','case02','case04','case07','case10'] and parts[3]=='ours_scene.ply'))
  if not allowed or '..' in parts:self.send_error(403);return None
  if path=='/':self.send_response(302);self.send_header('Location','/comparison/index.html');self.end_headers();return None
  return super().send_head()
ThreadingHTTPServer(('127.0.0.1',8092),Handler).serve_forever()
