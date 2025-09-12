import os
import sys

# Add optical_flow/RAFT paths so core modules and models are discoverable when
# importing from within the Mie_Postprocessing_Py tree.
_root = os.path.abspath(os.path.join(__file__, '..', '..', '..'))
_raft_dir = os.path.join(_root, 'optical_flow', 'RAFT')
_raft_core = os.path.join(_raft_dir, 'core')
for _p in (_raft_dir, _raft_core):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)

# Also add mie_postprocessing (sibling package) if not already present
_mpp = os.path.join(_root, 'Mie_Postprocessing_Py', 'mie_postprocessing')
if os.path.isdir(_mpp) and _mpp not in sys.path:
    sys.path.append(_mpp)

__all__ = []

