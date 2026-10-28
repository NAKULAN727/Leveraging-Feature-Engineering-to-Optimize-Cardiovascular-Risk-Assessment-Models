import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the implementation from scratch/run_expert_review.py
from scratch.run_expert_review import inspect_and_evaluate

if __name__ == "__main__":
    inspect_and_evaluate()
