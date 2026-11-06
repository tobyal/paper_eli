"""Server lifecycle. Only manages PIDs started by this application's own launcher."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from . import store as s
from . import service


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=['init','start','restart','status'])
    parser.add_argument('--port',type=int,default=18086); args=parser.parse_args()
    if args.command=='init':
        service.bootstrap(); print('Database initialized:',s.DATA/'database/app.db'); return
    if args.command=='restart':
        # Existing GPU subprocesses have their own sessions. The new worker
        # reconciles their PID/success marker instead of launching a duplicate.
        for name in ('server','worker'):
            pidfile=s.DATA/'service'/(name+'.pid')
            if not pidfile.exists(): continue
            pid=int(pidfile.read_text()); proc=Path(f'/proc/{pid}/cmdline')
            expected=b'industrial_swap_v2.webapp.'+(b'app:app' if name=='server' else b'worker')
            if proc.exists() and expected in proc.read_bytes():
                os.kill(pid,signal.SIGTERM)
                for _ in range(50):
                    if not proc.exists() or not proc.read_bytes(): break
                    time.sleep(.1)
    if args.command in ('start','restart'):
        service.bootstrap()
        for name,command in [('worker',[sys.executable,'-m','industrial_swap_v2.webapp.worker']),
            ('server',[sys.executable,'-m','uvicorn','industrial_swap_v2.webapp.app:app','--host','127.0.0.1','--port',str(args.port),'--workers','1'])]:
            pidfile=s.DATA/'service'/(name+'.pid')
            if pidfile.exists():
                pid=int(pidfile.read_text()); proc=Path(f'/proc/{pid}/cmdline')
                if proc.exists() and b'industrial_swap_v2.webapp' in proc.read_bytes():
                    print(name,'already running',pid); continue
            with (s.DATA/'service'/(name+'.log')).open('ab') as log:
                child=subprocess.Popen(command,cwd=s.ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            pidfile.write_text(str(child.pid)); print(name,'started',child.pid)
    for name in ('worker','server'):
        file=s.DATA/'service'/(name+'.pid'); print(name,'pid:',file.read_text() if file.exists() else 'not started')


if __name__=='__main__': main()
