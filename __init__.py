"""ComfyUI nodes for the original KohakuMossTrainer inference pipeline."""
import copy
import json
import os
from pathlib import Path

import torch
import folder_paths
from server import PromptServer

from .runtime import WORKER

TRAINER_ROOT = os.environ.get('KOHAKU_MOSS_TRAINER_ROOT', '')
CUDA_HOME = os.environ.get('KOHAKU_MOSS_CUDA_HOME', '')


def files(kind, preferred):
    choices = folder_paths.get_filename_list(kind)
    return (choices, {'default': preferred}) if preferred in choices else (choices,)


class KohakuMOSSFullPipelineLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'unet_name': files('diffusion_models', 'moss_image_base_fp16_fixed.safetensors'),
            'text_encoder_name': files('text_encoders', 'qwen3.5_9b_int8_convrot.safetensors'),
            'vae_name': files('vae', 'flux2-vae_small.safetensors'),
            'precision': (['auto', 'fp16', 'bf16', 'fp32'], {'default': 'auto'}),
            'memory_profile': (['auto', 'speed', 'balanced', 'low', 'minimal'], {'default': 'auto'}),
        }, 'optional': {
            'trainer_root': ('STRING', {'default': TRAINER_ROOT, 'advanced': True}),
            'python_path': ('STRING', {'default': '', 'advanced': True, 'tooltip': 'Empty: use trainer_root/.venv/bin/python'}),
            'tune': (['off', 'table'], {'default': 'off', 'advanced': True}),
            'adapter_mode': (['merge', 'bypass'], {'default': 'bypass', 'advanced': True}),
            'text_memory': (['auto', 'ring', 'offload', 'resident', 'cpu'], {'default': 'auto', 'advanced': True}),
            'vae_precision': (['auto', 'fp16', 'bf16', 'fp32'], {'default': 'auto', 'advanced': True}),
            'vae_memory': (['auto', 'resident', 'offload', 'ring', 'cpu'], {'default': 'auto', 'advanced': True}),
            'fp16_fix': (['none', 'online', 'fp32_residual'], {'default': 'none', 'advanced': True}),
            'fp16_scale': ('FLOAT', {'default': .0625, 'min': .000001, 'max': 1., 'advanced': True}),
            'cuda_home': ('STRING', {'default': CUDA_HOME, 'advanced': True}),
            'worker_ram_gib': ('FLOAT', {'default': 12., 'min': 0., 'max': 256., 'advanced': True}),
            'cxx': ('STRING', {'default': os.environ.get('CXX', ''), 'advanced': True}),
        }}
    RETURN_TYPES = ('KOHAKU_MOSS_PIPELINE',)
    RETURN_NAMES = ('pipeline',)
    FUNCTION = 'load'
    CATEGORY = 'KohakuMOSS/Full Pipeline'
    DESCRIPTION = 'Use the trainer MossPipeline in its own Python environment. Does not patch native ComfyUI models or replace desktop PyTorch.'

    def load(self, unet_name, text_encoder_name, vae_name, precision='auto', memory_profile='auto',
             trainer_root=TRAINER_ROOT, python_path='', tune='off', adapter_mode='bypass',
             text_memory='auto', vae_precision='auto', vae_memory='auto', fp16_fix='none',
             fp16_scale=.0625, cuda_home='', worker_ram_gib=12., cxx=''):
        if not trainer_root.strip():
            raise ValueError('Set trainer_root or KOHAKU_MOSS_TRAINER_ROOT to your KohakuMossTrainer checkout')
        root = Path(trainer_root).expanduser().resolve()
        if not (root / 'src/kohakumoss/inference/pipeline.py').is_file():
            raise ValueError('trainer_root must be a KohakuMossTrainer checkout containing src/kohakumoss/inference/pipeline.py')
        # Keep the venv executable path: resolving its symlink would select base Python.
        python = Path(python_path).expanduser().absolute() if python_path else root / '.venv/bin/python'
        if not python.is_file():
            raise ValueError(f'Trainer Python not found: {python}')
        transformer = folder_paths.get_full_path_or_raise('diffusion_models', unet_name)
        model = dict(trainer_root=str(root), python=str(python),
                     weights=str(Path(transformer).resolve().parent.parent), transformer=transformer,
                     text_encoder=folder_paths.get_full_path_or_raise('text_encoders', text_encoder_name),
                     vae=folder_paths.get_full_path_or_raise('vae', vae_name),
                     precision=precision, memory_profile=memory_profile, tune=tune,
                     adapter_mode=adapter_mode, text_memory=text_memory,
                     vae_precision=vae_precision, vae_memory=vae_memory, fp16_fix=fp16_fix,
                     fp16_scale=fp16_scale, cuda_home=cuda_home, worker_ram_gib=worker_ram_gib, cxx=cxx, loras=[])
        return (model,)


class KohakuMOSSFullPipelineLoRA:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'model': ('KOHAKU_MOSS_PIPELINE',),
                             'lora_name': (folder_paths.get_filename_list('loras'),),
                             'strength': ('FLOAT', {'default': 1., 'min': -10., 'max': 10., 'step': .05})}}
    RETURN_TYPES = ('KOHAKU_MOSS_PIPELINE',)
    RETURN_NAMES = ('pipeline',)
    FUNCTION = 'load'
    CATEGORY = 'KohakuMOSS/Full Pipeline'
    DESCRIPTION = 'Add a trainer-compatible LoRA. Uses trainer adapter loading, fused QKV conversion and FP16 fold handling. Chain nodes for multiple LoRAs.'

    def load(self, model, lora_name, strength):
        result = copy.deepcopy(model)
        path = folder_paths.get_full_path_or_raise('loras', lora_name)
        if strength != 0:
            result['loras'].append({'path': path, 'strength': strength})
        return (result,)


class KohakuMOSSFullPipelineSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'model': ('KOHAKU_MOSS_PIPELINE',),
            'positive': ('STRING', {'multiline': True, 'dynamicPrompts': True}),
            'negative': ('STRING', {'multiline': True, 'default': ''}),
            'width': ('INT', {'default': 1024, 'min': 16, 'max': 16384, 'step': 16}),
            'height': ('INT', {'default': 1024, 'min': 16, 'max': 16384, 'step': 16}),
            'batch_size': ('INT', {'default': 1, 'min': 1, 'max': 64}),
            'seed': ('INT', {'default': 114514, 'min': 0, 'max': 0xffffffffffffffff, 'control_after_generate': True}),
            'steps': ('INT', {'default': 50, 'min': 1, 'max': 10000}),
            'cfg': ('FLOAT', {'default': 3., 'min': 0., 'max': 100., 'step': .1}),
            'shift': ('FLOAT', {'default': 5., 'min': .001, 'max': 100., 'step': .1}),
            'cfg_mode': (['packed', 'two_pass'], {'default': 'packed'}),
            'skip_layers': ('STRING', {'default': 'default', 'tooltip': 'Trainer default skips negative block 22. Use none or comma-separated layer indices.'}),
        }, 'optional': {
            'keep_loaded': ('BOOLEAN', {'default': True, 'tooltip': 'Reuse the trainer pipeline. Native workflows automatically release this worker.'}),
            'force_run': ('BOOLEAN', {'default': True, 'tooltip': 'Actually sample again even with an unchanged seed. Disable to allow cached images.'}),
        }}
    RETURN_TYPES = ('IMAGE', 'STRING')
    RETURN_NAMES = ('images', 'trainer_report')
    FUNCTION = 'sample'
    CATEGORY = 'KohakuMOSS/Full Pipeline'
    DESCRIPTION = 'Original trainer Euler sampler, snapped sigma grid, negative skip layers, packed/two-pass CFG, per-image CUDA seeds and trainer dtype/kernel selection.'

    @classmethod
    def IS_CHANGED(cls, force_run=True, **kwargs):
        return float('nan') if force_run else False

    def sample(self, model, positive, negative, width, height, batch_size, seed, steps,
               cfg, shift, cfg_mode, skip_layers, keep_loaded=True, force_run=True):
        job = dict(positive=positive, negative=negative, width=width, height=height,
                   batch_size=batch_size, seed=seed, steps=steps, cfg=cfg, shift=shift,
                   cfg_mode=cfg_mode, skip_layers=skip_layers)
        images, report = WORKER.generate(model, job, keep_loaded)
        return (torch.from_numpy(images), json.dumps(report, ensure_ascii=False, indent=2))


class KohakuMOSSFullPipelineRelease:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'release': ('BOOLEAN', {'default': True})}}
    RETURN_TYPES = ('STRING',)
    FUNCTION = 'release'
    OUTPUT_NODE = True
    CATEGORY = 'KohakuMOSS/Full Pipeline'

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float('nan')

    def release(self, release=True):
        if release:
            WORKER.close()
        return ('Trainer model release requested' if release else 'Trainer model retained',)


NODE_CLASS_MAPPINGS = {c.__name__: c for c in [KohakuMOSSFullPipelineLoader, KohakuMOSSFullPipelineLoRA,
                                              KohakuMOSSFullPipelineSampler, KohakuMOSSFullPipelineRelease]}
NODE_DISPLAY_NAME_MAPPINGS = {
    'KohakuMOSSFullPipelineLoader': 'Kohaku MOSS Full Pipeline Loader',
    'KohakuMOSSFullPipelineLoRA': 'Kohaku MOSS Full Pipeline LoRA',
    'KohakuMOSSFullPipelineSampler': 'Kohaku MOSS Full Pipeline Sampler',
    'KohakuMOSSFullPipelineRelease': 'Kohaku MOSS Full Pipeline Release',
}
if getattr(PromptServer, 'instance', None) is not None:
    PromptServer.instance.add_on_prompt_handler(WORKER.before_prompt)
