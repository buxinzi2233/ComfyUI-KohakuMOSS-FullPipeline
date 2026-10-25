"""Run the unmodified KohakuMossTrainer inference API in its own Python environment."""
import contextlib
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

PROTOCOL = sys.stdout
PREFIX = '@@KOHAMOSS_FULL@@'


def emit(kind, **data):
    PROTOCOL.write(PREFIX + json.dumps({'type': kind, **data}, default=str) + '\n')
    PROTOCOL.flush()


def main():
    root = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(root / 'src'))
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        import torch
        from kohakumoss.inference import MossPipeline, knobs
        from kohakumoss.inference.placement_report import forward_tokens
        from kohakumoss.logging.log import setup
        from kohakumoss.model.numerics import fix_problems
        from kohakumoss.offload.accel import configure_allocator, release_cache
        from kohakumoss.encoders.vae import resolve_vae_dtype
        from kohakumoss_kernels import compile_setup
        setup()
        configure_allocator('cuda')
        compile_setup.apply()
    revision = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    runtime = {'python': sys.executable, 'torch': torch.__version__, 'cuda': torch.version.cuda,
               'gpu': torch.cuda.get_device_name(), 'trainer_root': str(root), 'trainer_revision': revision}
    if Path('/proc/self/cgroup').is_file():
        group = Path('/proc/self/cgroup').read_text().strip().split('::')[-1]
        limit = Path('/sys/fs/cgroup') / group.lstrip('/') / 'memory.max'
        runtime['worker_cgroup'] = group
        if limit.is_file():runtime['memory_limit_bytes'] = limit.read_text().strip()
    emit('ready', runtime=runtime)
    pipe = None
    loaded_key = None
    prepared_tokens = 0
    for line in sys.stdin:
        request = json.loads(line)
        ident = request['request_id']
        try:
            started = time.perf_counter()
            model, job = request['model'], request['job']
            vae_dtype = resolve_vae_dtype(model['vae_precision'], 'cuda')
            sizes = [(job['height'], job['width'])] * job['batch_size']
            tokens = forward_tokens(sizes, job['batch_size'], job['cfg'], job['cfg_mode'])
            key = json.dumps(model, sort_keys=True)
            reused = pipe is not None and key == loaded_key and tokens <= prepared_tokens
            with contextlib.redirect_stdout(sys.stderr):
                if not reused:
                    pipe = None
                    gc.collect()
                    release_cache(torch.device('cuda'))
                    dtype = knobs.compute_dtype(model['precision'], 'cuda')
                    fix_kw = {'c': model['fp16_scale']} if model['fp16_fix'] == 'online' else {}
                    problems = fix_problems(model['fp16_fix'], fix_kw, dtype == torch.float16,
                                            knobs.carries_fix(model['transformer']), 'FP16 compute', 'transformer')
                    if problems:
                        raise ValueError('; '.join(problems))
                    pipe = MossPipeline.from_pretrained(
                        model['weights'], transformer=model['transformer'],
                        text_encoder_file=model['text_encoder'], vae_file=model['vae'],
                        dtype=dtype, device='cuda',
                        adapters=[(a['path'], a['strength']) for a in model['loras']],
                        adapter_mode=model['adapter_mode'],
                        text_plan={'memory': model['text_memory']},
                        vae_plan={'memory': model['vae_memory']},
                        memory=model['memory_profile'],
                        fp16_fix=model['fp16_fix'], fp16_fix_kwargs=fix_kw,
                        tune=model['tune'], vae_dtype=vae_dtype,
                        forward_tokens=tokens,
                    )
                    loaded_key = key
                    prepared_tokens = tokens
                load_s = time.perf_counter() - started
                generate_start = time.perf_counter()
                result = pipe.generate(
                    prompts=[job['positive']] * job['batch_size'],
                    negative_prompts=[job['negative']] * job['batch_size'],
                    sizes=sizes, seeds=[job['seed'] + i for i in range(job['batch_size'])],
                    steps=job['steps'], cfg=job['cfg'], shift=job['shift'],
                    skip_layers=knobs.skip_layers(job['skip_layers']),
                    cfg_mode=job['cfg_mode'], batch=job['batch_size'], output='tensor',
                    progress=True,
                    step_callback=lambda i, _: emit('progress', request_id=ident, value=i + 1, max=job['steps']),
                )
                generate_s = time.perf_counter() - generate_start
                array = np.concatenate([
                    ((image.detach().float().cpu().permute(0, 2, 3, 1).numpy() + 1) / 2).clip(0, 1)
                    for image in result.images
                ], axis=0)
                if not np.isfinite(array).all():
                    raise RuntimeError('Trainer returned non-finite image pixels')
                np.save(request['image_path'], array, allow_pickle=False)
            report = {'runtime': runtime, 'pipeline_reused': reused, 'model_prepare_seconds': load_s,
                      'generate_seconds': generate_s, 'total_worker_seconds': time.perf_counter() - started,
                      'execution_api': 'kohakumoss.inference.MossPipeline.generate',
                      'vae_dtype': str(vae_dtype),
                      'vae_placement': pipe.place.vae.placement.name,
                      'text_placement': pipe.place.text_encoder.placement.name,
                      'weight_streaming': pipe.weight_stream is not None,
                      'actual_steps_per_batch': [len(stats['rows']) for stats in result.settings['step_stats']],
                      'job': job, 'model': model, 'trainer_settings': result.settings}
            emit('result', request_id=ident, report=report)
        except Exception as error:
            emit('error', request_id=ident, error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
            pipe = None
            loaded_key = None
            gc.collect()


if __name__ == '__main__':
    main()
