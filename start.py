from __future__ import annotations

import os
import subprocess


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
STREAMLIT = "/Users/svennatterer/miniconda3/bin/streamlit"


def main() -> None:
    subprocess.run(
        [STREAMLIT, "run", "app.py", "--server.port", "8502"],
        cwd=PROJECT_DIR,
        check=True,
    )


if __name__ == "__main__":
    main()
