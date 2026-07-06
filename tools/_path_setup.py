"""
_path_setup.py  —  ensure sibling tools/ modules and root packages are importable.

Import this FIRST in any tools/*.py script before other local imports.
"""
import sys, os
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_TOOLS_DIR)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)
