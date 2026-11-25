#!/usr/bin/env python3
"""Single entry point for reproducing the numerical results of the article

    J. Kluska, M. Goral, T. Maczka: "Calibration-aware posture certificates
    for humanoid ultra-wideband localization".

Run from the repository root:

    python reproduce.py                   # task "check" (about 2 min)
    python reproduce.py --task sphere     # exact sphere example, Table 1, Fig. 1 data
    python reproduce.py --task searches   # the five static G1 searches (about 8 min)
    python reproduce.py --task estimator  # estimator and calibration experiments (about 4 min)
    python reproduce.py --task standing   # standing, diffraction and local-profile studies (about 10 min)
    python reproduce.py --task audit      # recomputation audit of the stored results
    python reproduce.py --task all        # everything above, then "check"

"check" verifies the bundled G1 model against its SHA-256 manifest, runs the
unit tests and compares every number printed in the article with the stored
results and replay reports (one PASS/FAIL line per claim). The replay tasks
recompute the experiments on this machine and write difference reports next
to the stored results; they do not overwrite the authors' stored files.
See REPRODUCTION.md for details.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
POC = ROOT/'poc'
RESULTS = POC/'results'
CHECKERS = ['check_article_sphere.py', 'check_article_g1_static.py',
            'check_article_estimator.py', 'check_article_sensitivity.py']


def environment():
    env = dict(os.environ)
    env.setdefault('G1_XML', str(ROOT/'g1_model'/'g1.xml'))
    shim = str(POC/'menagerie_shim')
    env['PYTHONPATH'] = shim+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env.setdefault('PYTHONIOENCODING', 'utf-8')
    return env


def run(*args):
    """Run a Python program from the repository root; stop on failure."""
    print('\n>>> python '+' '.join(str(a) for a in args), flush=True)
    started = time.perf_counter()
    code = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, env=environment()).returncode
    print('<<< exit %d, %.1f s' % (code, time.perf_counter()-started), flush=True)
    if code:
        sys.exit(code)


def verify_model():
    """Compare the bundled Menagerie G1 files with the pinned manifest."""
    manifest = json.loads((POC/'model_manifest.json').read_text())
    root = Path(environment()['G1_XML']).resolve().parent
    bad = []
    for item in manifest['files']:
        path = root/item['path']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            bad.append(item['path'])
    print('G1 model: %d files, revision %s, %d mismatches'
          % (len(manifest['files']), manifest['requested_revision'][:7], len(bad)), flush=True)
    if bad:
        print('Missing or modified: '+', '.join(bad))
        print('Restore them with: python poc/menagerie_shim/fetch_model.py poc/model_manifest.json g1_model')
        sys.exit(1)


def task_check():
    verify_model()
    run('-m', 'pytest', '-q', 'poc')
    for checker in CHECKERS:
        run(POC/checker)


def task_sphere():
    run(POC/'verified_sphere.py')
    run(POC/'paper_example.py', '--output', RESULTS/'paper_validation.json')
    for name in ['bnb', 'excluded', 'hard', 'inconclusive']:
        shutil.copyfile(RESULTS/'paper_validation.json', RESULTS/name/'paper_validation.json')
    run(ROOT/'figsrc'/'make_figs.py', '--data-only')
    run(POC/'check_article_sphere.py')


def task_searches():
    verify_model()
    run(POC/'replay_g1_static.py')
    run(POC/'check_article_g1_static.py')


def task_estimator():
    verify_model()
    run(POC/'replay_article_estimator.py')
    run(POC/'check_article_estimator.py')


def task_standing():
    verify_model()
    run(POC/'replay_sensitivity.py')
    run(POC/'check_article_sensitivity.py')


def task_audit():
    verify_model()
    run(POC/'audit_reviewer.py')


TASKS = dict(check=task_check, sphere=task_sphere, searches=task_searches,
             estimator=task_estimator, standing=task_standing, audit=task_audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--task', default='check', choices=[*TASKS, 'all'])
    task = parser.parse_args().task
    started = time.perf_counter()
    for name in (['sphere', 'searches', 'estimator', 'standing', 'audit', 'check'] if task == 'all' else [task]):
        print('\n===== task: %s =====' % name, flush=True)
        TASKS[name]()
    print('\nAll steps of task "%s" finished successfully in %.0f s.' % (task, time.perf_counter()-started))


if __name__ == '__main__':
    main()
