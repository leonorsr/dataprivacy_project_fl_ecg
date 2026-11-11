"""Route reports from the already-running, pre-rename notebook writer."""
import fcntl
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def route_once(root=ROOT):
    offline = root / 'Privacy Techniques/LDP'
    clinical = root / 'Privacy Techniques/LDP + soft labels'
    clinical.mkdir(parents=True, exist_ok=True)
    refresh_needed = False
    for name in ('experiencias_treino_ldp.ipynb', 'comparation_ldp.ipynb', 'experiencias_logits.ipynb'):
        source = offline / name
        try:
            notebook = json.loads(source.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        if notebook.get('metadata', {}).get('offline_ldp_study'):
            continue
        if name != 'experiencias_logits.ipynb':
            source.replace(clinical / name)
        refresh_needed = True
    if refresh_needed:
        from importlib import reload
        from . import offline_notebooks
        reload(offline_notebooks).refresh()

def main():
    base = ROOT / 'artifacts/ldp_logits'
    with (base / 'folder_layout.lock').open('a') as own_lock:
        try:
            fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        while True:
            route_once()
            with (base / 'closed_loop/run.lock').open('a') as training_lock:
                try:
                    fcntl.flock(training_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    running = True
                else:
                    running = False
                    fcntl.flock(training_lock, fcntl.LOCK_UN)
            if not running:
                route_once()
                return
            time.sleep(10)

if __name__ == '__main__':
    main()
