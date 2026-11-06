"""Run paired per-anchor MC mean/std experiments and update comparation_softlabels."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from support.softlabels_moments import run
if __name__=='__main__':run()
