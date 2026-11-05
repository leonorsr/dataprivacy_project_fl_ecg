"""Run all paired SA communication/training experiments and refresh the notebooks."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from support.sa_communication import run
if __name__=='__main__':run()
