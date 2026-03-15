"""
Restore model variables after a kernel restart.
Run with:  %run -i fix_session.py
The -i flag runs in the notebook's own namespace so variables land globally.
"""
import sys, os
from pathlib import Path

# Ensure ml-models/ is on sys.path
_here = Path(os.path.abspath(__file__)).parent.parent   # ml-models/
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from models.root_cause.classifier import RootCauseClassifier
classifier = RootCauseClassifier()
classifier.load()
print("[fix_session] classifier ready")

# test split (requires splits from cell-7)
test_windows = splits["test"]["windows"]
test_metrics  = splits["test"]["metrics"]
print("[fix_session] test_windows / test_metrics ready")
