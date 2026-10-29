"""Checks for agent.guard: blocked operations are caught, safe code passes (no kernel, no API)."""

import sys
from pathlib import Path

# Make the project root importable when running this file as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.guard import HINT_CHART, HINT_EXPORT, HINT_OTHER, HINT_READ, check_code  # noqa: E402

# (code, hint the error must contain)
BLOCKED = [
    ("import subprocess", HINT_OTHER),
    ("import subprocess as sp\nsp.run(['dir'])", HINT_OTHER),
    ("import os\nos.system('dir')", HINT_OTHER),
    ("from os import system\nsystem('dir')", HINT_OTHER),
    ("import os as o\no.popen('dir')", HINT_OTHER),
    ("import os\nos.remove('data.csv')", HINT_OTHER),
    ("import os\nos.unlink('data.csv')", HINT_OTHER),
    ("import os\nos.rmdir('outputs')", HINT_OTHER),
    ("import shutil\nshutil.rmtree('outputs')", HINT_OTHER),
    ("from shutil import move\nmove('a.csv', 'b.csv')", HINT_OTHER),
    ("import requests", HINT_OTHER),
    ("import urllib.request", HINT_OTHER),
    ("from http import client", HINT_OTHER),
    ("import socket", HINT_OTHER),
    ("eval('1 + 1')", HINT_OTHER),
    ("exec('x = 1')", HINT_OTHER),
    ("compile('x = 1', '<s>', 'exec')", HINT_OTHER),
    ("__import__('os')", HINT_OTHER),
    ("open('out.txt', 'w')", HINT_EXPORT),
    ("open('out.txt', mode='a')", HINT_EXPORT),
    ("open('out.txt', 'r+')", HINT_EXPORT),
    ("mode = 'w'\nopen('out.txt', mode)", HINT_EXPORT),
    ("df.to_csv('out.csv')", HINT_EXPORT),
    ("df.to_csv(path_or_buf='out.csv', index=False)", HINT_EXPORT),
    ("df.to_excel('out.xlsx')", HINT_EXPORT),
    ("df.to_parquet('out.parquet')", HINT_EXPORT),
    ("df.to_json('out.json')", HINT_EXPORT),
    ("df.to_pickle('out.pkl')", HINT_EXPORT),
    ("from pathlib import Path\nPath('out.txt').write_text('x')", HINT_EXPORT),
    ("from pathlib import Path\nPath('out.txt').open('w')", HINT_EXPORT),
    ("np.save('out.npy', df.values)", HINT_EXPORT),
    ("plt.savefig('chart.png')", HINT_CHART),
    ("fig, ax = plt.subplots()\nfig.savefig('chart.png')", HINT_CHART),
    # Reading is allowed only inside data/ and uploads/ of the project
    ("open('evals/questions.json').read()", HINT_READ),
    ("import json\njson.load(open('evals/questions.json'))", HINT_READ),
    ("pd.read_csv('data/../evals/results/x.json')", HINT_READ),
    ("pd.read_csv('../other_project/data.csv')", HINT_READ),
    ("open('C:/Windows/win.ini')", HINT_READ),
    ("np.load('/etc/passwd')", HINT_READ),
    ("from pathlib import Path\nPath('agent/prompts.py').read_text()", HINT_READ),
    ("path = 'data/x.csv'\npd.read_csv(path)", HINT_READ),
    ("pd.read_excel(f'data/{name}.xlsx')", HINT_READ),
    ("from pathlib import Path\np = Path('data/x.csv')\np.read_text()", HINT_READ),
    ("import os\nos.listdir('data')", HINT_READ),
    ("import glob\nglob.glob('*.csv')", HINT_READ),
    ("import os\nfor root, dirs, files in os.walk('.'):\n    pass", HINT_READ),
    ("from pathlib import Path\nlist(Path('.').iterdir())", HINT_READ),
    ("import os\nos.scandir('.')", HINT_READ),
]

ALLOWED = [
    "df.groupby('Category')['Sales'].sum()",
    "df['Sales'].plot.hist(title='Sales')",
    "with open('data/x.csv') as f:\n    print(f.readline())",
    "open('data/x.csv', 'r', encoding='utf-8')",
    "open('data/x.csv', mode='rb')",
    "print('to_csv')",
    "print('df.to_csv(\"out.csv\")')",
    "text = df.head().to_csv(index=False)\nprint(text)",
    "import os\nprint(os.getcwd())",
    "import numpy as np\nnp.mean([1, 2, 3])",
    "from scipy import stats\nstats.ttest_ind([1, 2, 3], [2, 3, 4])",
    "def broken(:",  # a syntax error is left for Python to report
    "pd.read_csv('data/Titanic-Dataset.csv')",
    "pd.read_csv('uploads/my_file.csv')",
    "from pathlib import Path\nPath('data/x.csv').read_text()",
    "from pathlib import Path\npd.read_csv(Path('data') / 'x.csv')",
    "import io\npd.read_csv(io.StringIO('a,b\\n1,2'))",
]

passed = 0
failed = 0


def report(name: str, ok: bool, details: str) -> None:
    global passed, failed
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"       {details}")


def main() -> int:
    for code, hint in BLOCKED:
        error = check_code(code)
        first = code.splitlines()[-1]
        report(f"blocked: {first}", error is not None and hint in error and "nothing was executed" in error,
               (error or "NOT BLOCKED").splitlines()[1] if error else "NOT BLOCKED")

    for code in ALLOWED:
        error = check_code(code)
        report(f"allowed: {code.splitlines()[0]}", error is None, error or "passes")

    # The full message: every problem with its line, and each hint once
    error = check_code("import os\nos.remove('a')\ndf.to_csv('a.csv')\nplt.savefig('a.png')\ndf.to_excel('b.xlsx')") or ""
    report("message lists problems and hints",
           "line 2" in error and "line 3" in error and "line 4" in error
           and error.count(HINT_EXPORT) == 1 and HINT_CHART in error and HINT_OTHER in error,
           error.replace("\n", " | "))

    print(f"\nTotal: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
