#!/usr/bin/env python3
"""Read-only DRM residency census, without GPU libraries or command lines."""
import json
from pathlib import Path

devices = []
for path in sorted(Path('/sys/class/drm').glob('renderD*')):
    pci = (path/'device').resolve().name
    devices.append({'render':path.name,'pci':pci})
clients = []
seen = set()
inaccessible_processes = 0
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        comm = (proc/'comm').read_text().strip()
        for info in (proc/'fdinfo').iterdir():
            text = info.read_text()
            if 'drm-driver:' not in text:
                continue
            values = {k.strip():v.strip() for line in text.splitlines() if ':' in line for k,v in [line.split(':',1)] if k.startswith('drm-')}
            ident = (proc.name,values.get('drm-client-id'),values.get('drm-pdev'))
            if ident in seen:
                continue
            seen.add(ident)
            clients.append({'pid':int(proc.name),'comm':comm,**values})
    except (OSError,PermissionError):
        if proc.exists():
            inaccessible_processes += 1
        continue
print(json.dumps({'devices':devices,'accessible_clients':clients,
                 'inaccessible_processes':inaccessible_processes,
                 'coverage':'accessible process fdinfo only; an empty list is not proof of a free GPU'},indent=2))
