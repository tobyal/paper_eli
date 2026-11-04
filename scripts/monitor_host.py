"""Record shared-host load while one known experiment process remains alive."""
import argparse,json,os,time
from datetime import datetime,timezone
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('pid',type=int);p.add_argument('output',type=Path);a=p.parse_args()
proc=Path('/proc')/str(a.pid)
def identity():return (proc/'stat').read_text().split()[21]
start=identity()
with a.output.open('a') as file:
 while proc.exists():
  try:
   if identity()!=start:break
   status=(proc/'status').read_text().splitlines()
   rss=next((line.split()[1] for line in status if line.startswith('VmRSS:')),None)
   row={'utc':datetime.now(timezone.utc).isoformat(),'load_average_1_5_15':os.getloadavg(),'logical_cpus':os.cpu_count(),'experiment_rss_kib':int(rss) if rss else None}
   file.write(json.dumps(row)+'\n');file.flush()
  except FileNotFoundError:break
  time.sleep(15)
