"""Compatibility import for the shared linear-work event memory."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sleeping_machines.event_memory import affine_prefix, linear_memory as segmented_memory
