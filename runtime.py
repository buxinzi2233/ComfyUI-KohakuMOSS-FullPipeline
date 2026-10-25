"""One local trainer worker, with cancellation and GPU release for native workflows."""
import atexit
import json
import logging
import os
from pathlib import Path
import queue
import shutil
import subprocess
import signal
import tempfile
import threading
import uuid

import comfy.model_management as mm
import comfy.utils
import folder_paths

PREFIX = '@@KOHAMOSS_FULL@@'


class TrainerWorker:
    def __init__(self):
        self.process = None
        self.identity = None
        self.messages = queue.Queue()
        self.state_lock = threading.RLock()
        self.run_lock = threading.Lock()
        self.busy = False
        self.release_requested = False

    def _close(self):
        process, self.process = self.process, None
        self.identity = None
        if process is not None:
            if not self.busy and process.poll() is None:
                process.stdin.close()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            if process.poll() is None:
                if os.name == 'posix':
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if os.name == 'posix':
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait(timeout=5)
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    stream.close()

    def close(self):
        with self.state_lock:
            if self.busy:
                self.release_requested = True
            else:
                self._close()

    def before_prompt(self, data):
        graph = data.get('prompt', {})
        if not any(n.get('class_type') == 'KohakuMOSSFullPipelineSampler' for n in graph.values()):
            self.close()
        return data

    @staticmethod
    def _read(process, messages):
        try:
            for line in process.stdout:
                if line.startswith(PREFIX):
                    messages.put(json.loads(line[len(PREFIX):]))
                else:
                    logging.info('[KohakuMOSS worker] %s', line.rstrip())
        finally:
            messages.put({'type': 'closed'})

    def _start(self, model):
        identity = (model['trainer_root'], model['python'], model.get('cuda_home'), model.get('cxx'), model.get('worker_ram_gib'))
        if self.process is not None and (self.process.poll() is not None or identity != self.identity):
            self._close()
        if self.process is not None:
            return
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        env['HF_HUB_OFFLINE'] = '1'
        if model.get('cuda_home'):
            env['CUDA_HOME'] = model['cuda_home']
            env['PATH'] = str(Path(model['cuda_home']) / 'bin') + os.pathsep + env.get('PATH', '')
        if model.get('cxx'):
            env['CXX'] = model['cxx']
        self.messages = queue.Queue()
        command = [model['python'], '-u', str(Path(__file__).with_name('worker.py')), model['trainer_root']]
        if model.get('worker_ram_gib', 0) > 0:
            systemd = shutil.which('systemd-run')
            if not systemd:
                raise RuntimeError('A worker RAM limit requires systemd-run; install it or explicitly set worker_ram_gib=0')
            unit = 'kohakumoss-full-' + uuid.uuid4().hex[:12]
            command = [systemd, '--user', '--scope', '--quiet', '--unit=' + unit,
                       '-p', f"MemoryMax={int(model['worker_ram_gib'] * 1024**3)}",
                       '-p', 'MemorySwapMax=2G', *command]
        self.process = subprocess.Popen(
            command, cwd=model['trainer_root'], env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1, start_new_session=(os.name == 'posix'),
        )
        self.identity = identity
        threading.Thread(target=self._read, args=(self.process, self.messages), daemon=True).start()

    def generate(self, model, job, keep_loaded):
        import numpy as np
        with self.run_lock:
            # Trainer owns its CUDA context. Release native ComfyUI allocations first.
            mm.unload_all_models()
            mm.soft_empty_cache()
            with self.state_lock:
                self.busy = True
            try:
                self._start(model)
                with tempfile.TemporaryDirectory(prefix='kohakumoss-', dir=folder_paths.get_temp_directory()) as temp:
                    image_path = str(Path(temp) / 'image.npy')
                    request_id = str(uuid.uuid4())
                    request = {'request_id': request_id, 'model': model, 'job': job, 'image_path': image_path}
                    self.process.stdin.write(json.dumps(request) + '\n')
                    self.process.stdin.flush()
                    progress = comfy.utils.ProgressBar(job['steps'])
                    while True:
                        mm.throw_exception_if_processing_interrupted()
                        try:
                            message = self.messages.get(timeout=.2)
                        except queue.Empty:
                            if self.process.poll() is not None:
                                raise RuntimeError(f'Trainer worker exited with code {self.process.returncode}; see console log')
                            continue
                        kind = message['type']
                        if kind == 'ready':
                            logging.info('[KohakuMOSS Full Pipeline] runtime %s', json.dumps(message['runtime']))
                            continue
                        if kind == 'closed':
                            raise RuntimeError('Trainer worker closed before completing the image; see console log')
                        if message.get('request_id') != request_id:
                            continue
                        if kind == 'progress':
                            progress.update_absolute(message['value'], message['max'])
                        elif kind == 'error':
                            logging.error('[KohakuMOSS Full Pipeline] %s', message['traceback'])
                            raise RuntimeError(message['error'])
                        elif kind == 'result':
                            images = np.load(image_path, allow_pickle=False).copy()
                            report = message['report']
                            logging.info('[KohakuMOSS Full Pipeline] generated in %.2fs, model preparation %.2fs, reused=%s',
                                         report['generate_seconds'], report['model_prepare_seconds'], report['pipeline_reused'])
                            return images, report
            except BaseException:
                with self.state_lock:
                    self._close()
                raise
            finally:
                with self.state_lock:
                    self.busy = False
                    if self.release_requested or not keep_loaded:
                        self._close()
                    self.release_requested = False


WORKER = TrainerWorker()
atexit.register(WORKER.close)
