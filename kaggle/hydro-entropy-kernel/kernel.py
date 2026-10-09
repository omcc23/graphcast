import sys
from pathlib import Path

INPUT = Path("/kaggle/input")
if INPUT.is_dir():
    for path in sorted(INPUT.rglob("*")):
        print("input", path, flush=True)
else:
    print("no /kaggle/input", flush=True)

matches = list(INPUT.rglob("entropy.py")) if INPUT.is_dir() else []
if not matches:
    raise SystemExit("entropy.py was not mounted under /kaggle/input")
CODE = matches[0].parent
print("code", CODE, flush=True)
sys.path.insert(0, str(CODE))

from run_session import main

if __name__ == "__main__":
    main()
