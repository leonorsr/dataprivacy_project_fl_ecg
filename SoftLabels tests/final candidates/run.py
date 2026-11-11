"""Run training, then apply the user-approved local-dev analysis amendment."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from support.final_candidates import run,refresh_notebook
if __name__=='__main__':
    run()
    from support.local_decision import finalize
    from support.local_priority import refresh
    if finalize():refresh_notebook()
    refresh()
