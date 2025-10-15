import os, sys, runpy

_root = os.path.abspath(os.path.join(__file__, '..', '..', '..'))
_raft_dir = os.path.join(_root, 'optical_flow', 'RAFT')

if not os.path.isdir(_raft_dir):
    raise FileNotFoundError(f"Expected RAFT at {_raft_dir}")

_core = os.path.join(_raft_dir, 'core')
for _p in (_raft_dir, _core):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)

_demo_path = os.path.join(_raft_dir, 'demo.py')
runpy.run_path(_demo_path, run_name='__main__')

