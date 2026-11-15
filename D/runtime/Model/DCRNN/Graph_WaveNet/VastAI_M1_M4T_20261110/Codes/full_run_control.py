"""Process-safe tmux launcher and read-only full-checkpoint discovery.

Only experiment orchestration changes. The original trainer validates payload,
data, graph, library and preprocessing contracts again before restoring state.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid

VERSION = 'gwn-managed-full-v3-stop-reduction'
HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
MODEL = 'GraphWaveNet_distance_fixed_plus_adaptive'
SCHEMA = 'immutable-epoch-current-and-best-v1'
CONDITIONS = ['M1', 'M2', 'M3', 'M4', 'M3T', 'M4T']
STANDARD_OUTPUTS = ['metrics.csv', 'seed_metric_summary.csv', 'tests.csv',
                    'conditions.csv', 'score_rows.parquet', 'scoring_audit.json']
ENVIRONMENT_KEYS = ['PATH', 'RCLONE_CONFIG', 'TSA_SYSTEM_PYTHON', 'TSA_GWN_PYTHON',
                    'TSA_GWN_ENV_ROOT', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                    'CUDA_VISIBLE_DEVICES']


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_settings(path):
    if Path(path).suffix.lower() in ('.yaml', '.yml'):
        import yaml
        result = yaml.safe_load(Path(path).read_text(encoding='utf-8-sig'))
    else:
        result = read_json(path)
    if not isinstance(result, dict):
        raise ValueError('Experiment configuration must be an object/mapping.')
    return result


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class FileLease:
    """OS-owned nonblocking lock. Never delete the persistent lock inode."""
    def __init__(self, path):
        self.path = Path(path)
        self.stream = None

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open('a+b')
        try:
            if os.name == 'nt':
                import msvcrt
                if self.path.stat().st_size == 0:
                    stream.write(b'0')
                    stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            stream.close()
            return False
        self.stream = stream
        return True

    def release(self):
        if self.stream is None:
            return
        try:
            if os.name == 'nt':
                import msvcrt
                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        finally:
            self.stream.close()
            self.stream = None

    def __enter__(self):
        if not self.acquire():
            raise RuntimeError('Another process owns ' + str(self.path))
        return self

    def __exit__(self, *_):
        self.release()


def control_dir(output):
    return Path(output) / 'Control'


@contextmanager
def runner_lease(argv):
    """Guard all new pilot/full entry points, including direct run_full.sh."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--project-root', type=Path, default=Path('/workspace/TSA'))
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--mode', default='pilot')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    args, _ = parser.parse_known_args(argv)
    if args.self_test or args.validate_only or '--help' in argv or '-h' in argv:
        yield True
        return
    output = (args.output_root or args.project_root / 'Result/GraphWaveNet/VastAI').resolve()
    lease = FileLease(control_dir(output) / 'training.lock')
    if not lease.acquire():
        print('[duplicate blocked] Another training process holds the project lock. No new training started.', flush=True)
        yield False
        return
    try:
        atomic_json(control_dir(output) / 'training_owner.json', {
            'pid': os.getpid(), 'mode': args.mode, 'project_root': str(args.project_root.resolve()),
            'started_utc': datetime.now(timezone.utc).isoformat(), 'controller_version': VERSION})
        yield True
    finally:
        lease.release()


def validate_seed_checkpoint(seed_folder):
    """Check hashes without unpickling .pt. Return latest valid generation."""
    seed_folder = Path(seed_folder)
    root = seed_folder / 'Checkpoints'
    candidates = sorted(root.glob('epoch_*'), reverse=True) if root.is_dir() else []
    expected = read_json(seed_folder / 'contract.json') if (seed_folder / 'contract.json').is_file() else None
    errors = []
    for folder in candidates:
        if not folder.is_dir():
            continue
        try:
            if folder.is_symlink():
                raise ValueError('Symlink checkpoint generation')
            commit = read_json(folder / 'COMMITTED.json')
            manifest = read_json(folder / 'manifest.json')
            if commit.get('schema_version') != SCHEMA or manifest.get('schema_version') != SCHEMA:
                raise ValueError('Unexpected checkpoint schema')
            if commit.get('generation') != folder.name or manifest.get('generation') != folder.name:
                raise ValueError('Generation identity mismatch')
            if commit.get('manifest_sha256') != sha256(folder / 'manifest.json'):
                raise ValueError('Manifest checksum mismatch')
            if expected is not None and manifest.get('contract_sha256') != canonical_hash(expected):
                raise RuntimeError('Checkpoint contract mismatch: ' + str(folder))
            state = folder / 'state.pt'
            if state.is_symlink() or state.stat().st_size != manifest.get('state_bytes'):
                raise ValueError('State size mismatch')
            if sha256(state) != manifest.get('state_sha256'):
                raise ValueError('State checksum mismatch')
            if (int(manifest['completed_epochs']) < 1
                    or int(manifest['epoch']) + 1 != int(manifest['completed_epochs'])
                    or not 1 <= int(manifest['best_epoch']) <= int(manifest['completed_epochs'])):
                raise ValueError('Invalid completed epoch')
            return {'seed_folder': str(seed_folder), 'generation': folder.name,
                    'completed_epochs': manifest['completed_epochs'], 'rejected_generations': errors}
        except RuntimeError:
            raise
        except (OSError, ValueError, TypeError, KeyError) as error:
            errors.append({'generation': folder.name, 'reason': str(error)})
    artifacts = root.is_dir() and any(root.iterdir())
    if artifacts or (seed_folder / 'last.pt').exists() or (seed_folder / 'best.pt').exists():
        raise RuntimeError('Checkpoint artifacts exist but no valid committed generation remains: '
                           + str(seed_folder) + '; ' + json.dumps(errors))
    return None


def inspect_run(run):
    records = []
    for seed in sorted((Path(run) / 'Models' / MODEL).glob('seed*')):
        if seed.is_dir():
            result = validate_seed_checkpoint(seed)
            if result is not None:
                records.append(result)
    return records


def _lineage_parent(source, record):
    value = record.get('resume_source')
    if not value:
        return None
    parent = Path(value).expanduser()
    if not parent.is_absolute():
        parent = source.parent / parent
    if not parent.is_dir():
        # Recovered Drive batches can have a different absolute project root.
        relocated = source.parent / str(value).replace('\\', '/').rstrip('/').split('/')[-1]
        if relocated.is_dir():
            parent = relocated
        else:
            raise RuntimeError('Previous resume batch is missing; restore it before continuing: ' + str(parent))
    return parent.resolve()


def resolve_condition_source(source, condition, selected_count, _seen=None):
    """Follow the batch lineage separately for each condition; never borrow another arm."""
    if source is None:
        return None
    source = Path(source).expanduser().resolve()
    seen = set() if _seen is None else set(_seen)
    if source in seen:
        raise RuntimeError('Resume lineage cycle: ' + str(source))
    seen.add(source)
    if not source.is_dir():
        raise FileNotFoundError('Resume source directory missing: ' + str(source))
    if (source / 'Models').is_dir():
        if selected_count != 1:
            raise ValueError('A condition run requires exactly one selected condition.')
        info = read_json(source / 'experiment_config.json')
        if info.get('training', {}).get('input_condition') != condition:
            raise ValueError('Resume condition mismatch: ' + condition)
        return source if inspect_run(source) else None
    record = read_json(source / 'batch_manifest.json')
    if record.get('mode') not in ('full', 'pilot'):
        raise ValueError('Resume source has no recognized batch mode: ' + str(source))
    folder = source / 'Conditions' / condition
    runs = sorted((p for p in folder.glob('*') if p.is_dir() and (p / 'Models').is_dir()), reverse=True)
    for run in runs:
        if inspect_run(run):
            return run
    parent = _lineage_parent(source, record)
    if parent is not None:
        if (parent / 'Models').is_dir():
            info = read_json(parent / 'experiment_config.json')
            if info.get('training', {}).get('input_condition') != condition:
                return None
            return resolve_condition_source(parent, condition, 1, seen)
        return resolve_condition_source(parent, condition, selected_count, seen)
    return None


def discover_latest_full(output_root):
    skipped = []
    full = Path(output_root) / 'full'
    for batch in sorted((p for p in full.glob('*') if p.is_dir()), reverse=True):
        manifest = batch / 'batch_manifest.json'
        if not manifest.is_file():
            if any(batch.glob('Conditions/*/*/Models/*/seed*/Checkpoints/epoch_*')):
                raise RuntimeError('A batch has checkpoints but its manifest is missing: ' + str(batch))
            skipped.append({'batch': str(batch), 'reason': 'no batch manifest/checkpoints'})
            continue
        record = read_json(manifest)
        if record.get('mode') != 'full' or record.get('validate_only'):
            skipped.append({'batch': str(batch), 'reason': 'not a training full batch'})
            continue
        found = [resolve_condition_source(batch, c, len(CONDITIONS)) for c in CONDITIONS]
        if any(p is not None for p in found):
            return batch.resolve(), skipped
        skipped.append({'batch': str(batch), 'reason': 'no completed checkpoint; empty/preflight failure'})
    return None, skipped


def _same_training_configuration(source_config, current, seeds):
    from stopping_policy import stopping_reduction
    changes = {}
    for section in ('data', 'model', 'training'):
        if source_config.get(section, {}) != current.get(section, {}):
            if section == 'training':
                try:
                    changes = stopping_reduction(source_config.get(section, {}), current.get(section, {}))
                except ValueError as error:
                    raise RuntimeError('Latest full uses a different training configuration: ' + str(error)) from error
                continue
            raise RuntimeError('Latest full uses a different ' + section + ' configuration. '
                               'Use --fresh only for an intentional new experiment.')
    previous = source_config.get('seeds_resolved', source_config.get('execution', {}).get('seeds', [11]))
    if list(map(int, previous)) != list(map(int, seeds)):
        raise RuntimeError('Latest full uses different seeds. Use --fresh for a separate experiment.')
    return changes


def build_resume_plan(output_root, config, conditions, seeds, final_test=False, fresh=False):
    batch, skipped = (None, []) if fresh else discover_latest_full(output_root)
    plan = {'controller_version': VERSION, 'action': 'launch', 'resume_source': None,
            'condition_sources': {}, 'checkpoint_states': {}, 'skipped_batches': skipped,
            'selected_conditions': conditions, 'seeds': seeds, 'final_test': bool(final_test)}
    if batch is None:
        return plan
    record = read_json(batch / 'batch_manifest.json')
    changes = _same_training_configuration(read_json(batch / 'resolved_configuration.json'), config, seeds)
    plan['stopping_policy_changes'] = changes
    plan['resume_source'] = str(batch)
    for condition in conditions:
        source = resolve_condition_source(batch, condition, len(conditions))
        if source is not None:
            source_config = read_json(source / 'experiment_config.json')
            if source_config.get('training', {}).get('mode') != 'full':
                raise RuntimeError('Auto full resume cannot use a pilot checkpoint: ' + str(source))
            plan['condition_sources'][condition] = str(source)
            plan['checkpoint_states'][condition] = inspect_run(source)
    # A completed invocation is a no-op unless final-test still needs to run.
    splits = ('validation', 'test') if final_test else ('validation',)
    complete = (not changes and record.get('status') == 'passed'
                and (not final_test or record.get('final_test'))
                and all(record.get('conditions', {}).get(c, {}).get('status') == 'passed' for c in conditions)
                and list(map(int, record.get('seeds', []))) == list(map(int, seeds)))
    scoring = record.get('combined_scoring', {})
    if complete and all(split in scoring and all((Path(scoring[split]) / name).is_file()
                                                for name in STANDARD_OUTPUTS) for split in splits):
        plan['action'] = 'completed'
    return plan


def running_experiments(project, proc_root=Path('/proc')):
    """Find already-running legacy/current processes without exposing full argv."""
    if not proc_root.is_dir():
        return []
    found = []
    for folder in proc_root.iterdir():
        if not folder.name.isdigit() or int(folder.name) == os.getpid():
            continue
        try:
            args = [v.decode(errors='replace') for v in (folder / 'cmdline').read_bytes().split(b'\0') if v]
            if not any(Path(arg).name == 'run_experiments.py' for arg in args):
                continue
            marker = args.index('--project-root') if '--project-root' in args else None
            same = marker is not None and Path(args[marker + 1]).resolve() == Path(project).resolve()
            if marker is None:
                same = any(str(BUNDLE) in arg for arg in args)
            if same:
                found.append({'pid': int(folder.name), 'mode': args[args.index('--mode') + 1]
                              if '--mode' in args else 'pilot'})
        except (OSError, ValueError, IndexError):
            continue
    return found


def process_ancestors(pid):
    result = set()
    while pid > 1 and pid not in result:
        result.add(pid)
        try:
            values = Path('/proc', str(pid), 'status').read_text().splitlines()
            pid = int(next(line for line in values if line.startswith('PPid:')).split()[1])
        except (OSError, ValueError, StopIteration):
            break
    return result


def tmux_call(*args, check=False):
    return subprocess.run(['tmux', *map(str, args)], text=True, capture_output=True, check=check)


def matching_session(processes):
    panes = tmux_call('list-panes', '-a', '-F', '#{session_name}|#{pane_pid}')
    ancestry = set().union(*(process_ancestors(p['pid']) for p in processes))
    for line in panes.stdout.splitlines():
        try:
            session, pid = line.rsplit('|', 1)
            if int(pid) in ancestry:
                return session
        except ValueError:
            continue
    return None


def ensure_tmux():
    if shutil.which('tmux'):
        return
    if os.name != 'posix' or not hasattr(os, 'geteuid') or os.geteuid() != 0 or not shutil.which('apt-get'):
        raise RuntimeError('tmux is missing. Install tmux on this Linux server first.')
    print('[setup] Installing tmux once using apt-get.', flush=True)
    subprocess.run(['apt-get', 'update'], check=True)
    subprocess.run(['apt-get', 'install', '-y', 'tmux'], check=True,
                   env=dict(os.environ, DEBIAN_FRONTEND='noninteractive'))


def attach(session, detached):
    print('[tmux] Reconnect: tmux attach -t ' + shlex.quote(session), flush=True)
    if not detached and sys.stdin.isatty() and sys.stdout.isatty():
        command = 'switch-client' if os.environ.get('TMUX') else 'attach-session'
        os.execvp('tmux', ['tmux', command, '-t', session])


def is_linux_server():
    return os.name == 'posix' and Path('/proc').is_dir()


def request_environment(request):
    env = dict(os.environ, TSA_PROJECT_ROOT=request['project_root'], TSA_OUTPUT_ROOT=request['output_root'])
    env.update(request.get('environment', {}))
    if 'RCLONE_CONFIG' not in env:
        candidate = Path.home() / '.config/rclone/tsa-vast.conf'
        if candidate.is_file():
            env['RCLONE_CONFIG'] = str(candidate)
    return env


def wait_for_training(command, cwd, env, project, on_stopping=None):
    """Isolate tee/Bash from terminal SIGINT; stop only our actual Python runner.

    The runner's existing SIGTERM handler performs interruption bookkeeping and
    backup. Do not return until its wrapper/tee pipeline has fully finished.
    """
    stopped = False
    notified = False
    signaled = set()
    previous = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}

    def request_stop(_number, _frame):
        nonlocal stopped
        stopped = True

    for number in previous:
        signal.signal(number, request_stop)
    try:
        child = subprocess.Popen(command, cwd=cwd, env=env, start_new_session=True)
        while True:
            if stopped:
                if not notified:
                    print('[stopping] Waiting for checkpoint bookkeeping/Drive backup. Please wait; no new training will start.', flush=True)
                    if on_stopping is not None:
                        try:
                            on_stopping()
                        except Exception as error:
                            print('[stopping status warning] ' + str(error), file=sys.stderr, flush=True)
                    notified = True
                # A stop received during startup is kept until the runner appears.
                for process in running_experiments(project):
                    pid = process['pid']
                    if pid not in signaled and child.pid in process_ancestors(pid):
                        try:
                            os.kill(pid, signal.SIGTERM)
                            signaled.add(pid)
                        except ProcessLookupError:
                            pass
            try:
                code = child.wait(timeout=0.5)
                return (130 if stopped else code), stopped
            except subprocess.TimeoutExpired:
                continue
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def configure_tmux_view(session):
    # Session scope only: do not change unrelated tmux users/sessions.
    tmux_call('set-option', '-t', session, 'mouse', 'on')
    tmux_call('set-option', '-t', session, 'history-limit', '50000')


def interactive_worker(request_path):
    """Keep a usable shell after completion, interruption or a visible error."""
    request = read_json(request_path)
    try:
        code = supervise(request_path)
    except KeyboardInterrupt:
        atomic_json(request['status_path'], dict(request, state='interrupted', exit_code=130))
        code = 130
    resume = ['bash', str(BUNDLE / 'start_full.sh')]
    if request.get('final_test'):
        resume.append('--final-test')
    print('\n[worker] ' + ('Interrupted; review the backup status above.' if code == 130 else 'Finished with exit code ' + str(code) + '.')
          + ' This pane now returns to a Bash prompt.\n'
          + 'Resume/reconnect: ' + shlex.join(resume) + '\n'
          + 'Scroll: mouse wheel or Ctrl+B then [; press q to leave copy mode.', flush=True)
    if sys.stdin.isatty() and sys.stdout.isatty():
        os.chdir(BUNDLE)
        os.execvpe('/bin/bash', ['bash', '--noprofile', '--norc', '-i'], request_environment(request))
    return code


def supervise(request_path):
    request = read_json(request_path)
    status_path = Path(request['status_path'])
    try:
        plan = build_resume_plan(request['output_root'], read_settings(request['config']),
                                 request['conditions'], request['seeds'], request['final_test'], request['fresh'])
        atomic_json(Path(request_path).with_name('resume_plan.json'), plan)
        if plan['action'] == 'completed':
            print('[completed] Latest full already finished. No training repeated. ' + str(plan['resume_source']), flush=True)
            atomic_json(status_path, dict(request, state='completed', plan=plan, exit_code=0))
            return 0
        if plan['resume_source']:
            print('[auto-resume] ' + plan['resume_source'], flush=True)
            for condition in request['conditions']:
                states = plan['checkpoint_states'].get(condition, [])
                print('  ' + condition + ': ' + (', '.join('epoch ' + str(s['completed_epochs']) for s in states)
                                                 if states else 'not started; fresh condition'), flush=True)
        else:
            print('[fresh] No previous full checkpoint. Starting the first full experiment.', flush=True)
        command = ['bash', str(BUNDLE / 'run_full.sh'), '--config', request['config'],
                   '--conditions', *request['conditions'], '--seeds', *map(str, request['seeds'])]
        if plan['resume_source']:
            command += ['--resume-source', plan['resume_source']]
        if request['final_test']:
            command += ['--final-test']
        atomic_json(status_path, dict(request, state='running', plan=plan, supervisor_pid=os.getpid()))
        def stopping():
            atomic_json(status_path, dict(request, state='stopping', plan=plan, supervisor_pid=os.getpid()))
        code, stopped = wait_for_training(command, BUNDLE, request_environment(request),
                                         request['project_root'], on_stopping=stopping)
        atomic_json(status_path, dict(request, state='interrupted' if stopped else ('finished' if code == 0 else 'failed'),
                                     plan=plan, exit_code=code))
        return code
    except KeyboardInterrupt:
        atomic_json(status_path, dict(request, state='interrupted', exit_code=130))
        print('[interrupted] Stopped before training startup; no new model was launched.', flush=True)
        return 130
    except Exception as error:
        atomic_json(status_path, dict(request, state='failed', exit_code=1, error=str(error)))
        print('[controller ERROR] ' + str(error), file=sys.stderr, flush=True)
        return 1


def start(args):
    if not is_linux_server():
        raise RuntimeError('Run start_full.sh inside the Vast.ai Linux terminal, not directly on Windows.')
    ensure_tmux()
    project = args.project_root.expanduser().resolve()
    output = (args.output_root or project / 'Result/GraphWaveNet/VastAI').expanduser().resolve()
    settings = read_settings(args.config)
    conditions = args.conditions or settings.get('execution', {}).get('conditions', CONDITIONS)
    seeds = args.seeds or settings.get('execution', {}).get('seeds', [11])
    conditions = [c for item in conditions for c in str(item).split(',')]
    seeds = [int(s) for item in seeds for s in str(item).split(',')]
    if not conditions or len(conditions) != len(set(conditions)) or not set(conditions) <= set(CONDITIONS):
        raise ValueError('Unknown or duplicate conditions.')
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('Duplicate/empty seeds.')
    key = hashlib.sha256(str(output).encode()).hexdigest()[:8]
    session = 'tsa-gwn-auto-' + key
    control = control_dir(output)
    lease = FileLease(control / 'launcher.lock')
    if not lease.acquire():
        print('[launch in progress] Another start command is preparing this project. Run the same command again shortly.', flush=True)
        return 0
    target_session = None
    try:
        processes = running_experiments(project)
        if processes:
            target_session = matching_session(processes)
            print('[already running] No duplicate training started. PIDs=' + ','.join(str(p['pid']) for p in processes), flush=True)
            if target_session is None:
                print('Existing training was started outside a discoverable tmux session; leave it running.', flush=True)
                return 0
        else:
            status_file = control / 'tmux_status.json'
            if status_file.is_file():
                owned = read_json(status_file)
                recorded_session = owned.get('session')
                if (owned.get('output_root') == str(output) and recorded_session
                        and tmux_call('has-session', '-t', recorded_session).returncode == 0):
                    session = recorded_session
            exists = tmux_call('has-session', '-t', session).returncode == 0
            if exists:
                owner = tmux_call('show-options', '-t', session, '-v', '@tsa_output').stdout.strip()
                if owner != str(output):
                    raise RuntimeError('tmux session name is owned by another task; refusing to replace it: ' + session)
                previous = read_json(control / 'tmux_status.json')
                pane = tmux_call('list-panes', '-t', session, '-F', '#{pane_dead}').stdout.strip()
                if previous.get('state') in ('launching', 'running', 'stopping') and pane == '0':
                    target_session = session
                else:
                    # Keep the old pane and its history; allocate a new owned name.
                    session += '-' + uuid.uuid4().hex[:6]
            if target_session is None:
                plan = build_resume_plan(output, settings, conditions, seeds, args.final_test, args.fresh)
                if plan['action'] == 'completed':
                    print('[completed] Latest full already finished. No new training started. ' + str(plan['resume_source']), flush=True)
                    return 0
                task_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8]
                task_dir = control / 'Invocations' / task_id
                task_dir.mkdir(parents=True, exist_ok=False)
                request = {'version': VERSION, 'project_root': str(project), 'output_root': str(output),
                           'config': str(args.config.resolve()), 'conditions': conditions, 'seeds': seeds,
                           'final_test': args.final_test, 'fresh': args.fresh, 'session': session,
                           'status_path': str(control / 'tmux_status.json'), 'state': 'launching',
                           'environment': {name: os.environ[name] for name in ENVIRONMENT_KEYS if name in os.environ}}
                request_path = task_dir / 'request.json'
                atomic_json(request_path, request)
                atomic_json(control / 'tmux_status.json', request)
                command = shlex.join([sys.executable, str(HERE / 'full_run_control.py'),
                                      'worker', '--request', str(request_path)])
                result = tmux_call('new-session', '-d', '-s', session, '-c', BUNDLE, command)
                if result.returncode:
                    atomic_json(control / 'tmux_status.json', dict(request, state='failed', error=result.stderr))
                    raise RuntimeError('tmux launch failed: ' + result.stderr)
                tmux_call('set-option', '-t', session, '@tsa_output', str(output), check=True)
                tmux_call('set-window-option', '-t', session, 'remain-on-exit', 'on')
                print('[started] ' + session + '\nControl record: ' + str(task_dir), flush=True)
                target_session = session
    finally:
        lease.release()
    if target_session:
        configure_tmux_view(target_session)
        attach(target_session, args.detached)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    launch = sub.add_parser('start')
    launch.add_argument('--project-root', type=Path, default=Path(os.environ.get('TSA_PROJECT_ROOT', '/workspace/TSA')))
    launch.add_argument('--output-root', type=Path, default=Path(os.environ['TSA_OUTPUT_ROOT']) if os.environ.get('TSA_OUTPUT_ROOT') else None)
    launch.add_argument('--config', type=Path, default=BUNDLE / 'configs/default.json')
    launch.add_argument('--conditions', nargs='+')
    launch.add_argument('--seeds', nargs='+')
    launch.add_argument('--final-test', action='store_true')
    launch.add_argument('--fresh', action='store_true')
    launch.add_argument('--detached', action='store_true')
    worker = sub.add_parser('supervise')
    worker.add_argument('--request', required=True, type=Path)
    interactive = sub.add_parser('worker')
    interactive.add_argument('--request', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.operation == 'start':
            return start(args)
        return interactive_worker(args.request) if args.operation == 'worker' else supervise(args.request)
    except Exception as error:
        print('[controller ERROR] ' + str(error), file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
