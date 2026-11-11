from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from support.federated_global.validation import validate_soft
if __name__=='__main__':print(validate_soft())
