"""
The computational core must not depend on plotnine
"""

import subprocess
import sys


def test_core_does_not_import_plotnine():
    code = (
        "import sys\n"
        "import plotnine_ggdist._core.intervals, plotnine_ggdist._core.density\n"
    )
    # importing the subpackage runs plotnine_ggdist/__init__.py, so check the
    # core modules' own imports instead
    import ast
    from pathlib import Path

    import plotnine_ggdist._core as core

    for path in Path(core.__file__).parent.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any(n.startswith("plotnine") for n in names), f"{path.name} imports {names}"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
