# pytest conftest -- ensure workspace src/ takes precedence over installed package.
#
# The editable install may point to a different clone (C:/Temp/bga-invention-engine/src).
# This conftest inserts the workspace src/ at the front of sys.path so that
# new modules written in this session are importable during test runs.
import sys
import os

# Insert workspace src at position 0 so it shadows the installed editable package
_workspace_src = os.path.join(os.path.dirname(__file__), "src")
if _workspace_src not in sys.path:
    sys.path.insert(0, _workspace_src)
