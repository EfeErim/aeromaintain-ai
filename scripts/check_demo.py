"""Render the hosted demo with only the packages in ``requirements.txt``.

CI installs ``requirements.txt`` into a clean environment and runs this script, so a
dependency the demo imports but the hosting file omits fails the build instead of
the live site.
"""

from __future__ import annotations

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=120)
    app.run()
    if app.exception:
        for error in app.exception:
            print(error.message, file=sys.stderr)
            print("\n".join(error.stack_trace), file=sys.stderr)
        return 1
    if not app.title and not app.header and not app.markdown:
        print("The demo rendered no content", file=sys.stderr)
        return 1
    print("Demo rendered without errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
