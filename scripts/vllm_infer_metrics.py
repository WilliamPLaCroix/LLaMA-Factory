# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import json
from typing import Optional

import fire
from transformers import Seq2SeqTrainingArguments
from peft import PeftModel, PeftConfig
from transformers import LlamaTokenizer, LlamaForCausalLM

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.misc import check_version, get_device_count
from llamafactory.extras.packages import is_vllm_available
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

import numpy as np
from evaluate import load
sari = load("sari")
from bert_score import score
#perplexity = load("perplexity", module_type="metric")
#from readability import Readability

import textstat

import wandb


# ---------------------------------- degbugging vLLM multiprocessing issues ----------------------------------
import platform, sys, time, torch # for debugging vLLM multiprocessing issues
try:
    import vllm
    from vllm import envs as vllm_envs
except Exception:
    vllm = None
    vllm_envs = None

def _safe(obj, path, default=None):
    cur = obj
    for p in path.split("."):
        if cur is None:
            return default
        cur = getattr(cur, p, None)
    return cur if cur is not None else default

def _as_json(obj):
    try:
        return json.loads(json.dumps(obj, default=lambda x: str(x)))
    except Exception:
        return str(obj)

def print_repro_debug(llm=None, tok=None, sp=None, prompts=None, label="BEFORE"):
    print("\n" + "="*30 + f" DEBUG SNAPSHOT [{label}] " + "="*30)

    # 1) Environment and versions
    env_keys = [
        "CUDA_VISIBLE_DEVICES", "VLLM_ENABLE_V1_MULTIPROCESSING", "VLLM_WORKER_MULTIPROC_METHOD",
        "VLLM_ATTENTION_BACKEND", "VLLM_USE_RAY", "VLLM_USE_MODELSCOPE",
        "CUBLAS_WORKSPACE_CONFIG", "PYTORCH_CUDA_ALLOC_CONF",
        "TOKENIZERS_PARALLELISM", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        "CUDA_DEVICE_MAX_CONNECTIONS"
    ]
    env_report = {k: os.environ.get(k) for k in env_keys if k in os.environ}

    vers_report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", None),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
        "vllm": getattr(vllm, "__version__", None),
    }

    tf32 = {
        "matmul_allow_tf32": getattr(torch.backends.cuda.matmul, "allow_tf32", None),
        "cudnn_allow_tf32": getattr(torch.backends.cudnn, "allow_tf32", None),
        "deterministic_algs": getattr(torch, "are_deterministic_algorithms_enabled", lambda: None)(),
    }

    # 2) Tokenizer info
    tok_report = {}
    if tok is not None:
        tok_report = {
            "name_or_path": getattr(tok, "name_or_path", None),
            "pad_token_id": getattr(tok, "pad_token_id", None),
            "eos_token_id": getattr(tok, "eos_token_id", None),
            "bos_token_id": getattr(tok, "bos_token_id", None),
            "chat_template_present": hasattr(tok, "chat_template") and tok.chat_template is not None,
            "special_tokens_map": _as_json(getattr(tok, "special_tokens_map", {})),
        }

    # 3) Sampling params
    sp_report = {}
    if sp is not None:
        fields = [
            "temperature", "top_p", "top_k", "repetition_penalty",
            "max_tokens", "min_tokens", "n", "use_beam_search",
            "best_of", "stop", "stop_token_ids", "seed",
            "ignore_eos", "skip_special_tokens", "include_stop_str_in_output"
        ]
        sp_report = {f: _as_json(getattr(sp, f, None)) for f in fields}

    # 4) vLLM engine internals (best effort, version tolerant)
    engine_report = {}
    if llm is not None:
        eng = getattr(llm, "llm_engine", None)
        engine_report = {
            "model": _safe(eng, "model_config.model"),
            "dtype": str(_safe(eng, "model_config.dtype")),
            "tensor_parallel_size": _safe(eng, "parallel_config.tensor_parallel_size")
                                  or _safe(eng, "model_config.tensor_parallel_size"),
            "max_model_len": _safe(eng, "model_config.max_model_len"),
            "enforce_eager": _safe(eng, "engine_config.enforce_eager"),
            "max_num_seqs": _safe(eng, "scheduler_config.max_num_seqs"),
            "max_num_batched_tokens": _safe(eng, "scheduler_config.max_num_batched_tokens"),
            "kv_cache_dtype": str(_safe(eng, "cache_config.cache_dtype")),
            "gpu_memory_utilization": _safe(eng, "engine_config.gpu_memory_utilization"),
            "spec_decode_enabled": bool(_safe(eng, "spec_decode_config") or _safe(eng, "engine_config.speculative_config")),
        }

    # 5) vLLM envs snapshot
    vllm_env_report = {}
    if vllm_envs is not None:
        try:
            vllm_env_report = {
                "VLLM_ENABLE_V1_MULTIPROCESSING": getattr(vllm_envs, "VLLM_ENABLE_V1_MULTIPROCESSING", None),
                "VLLM_USE_RAY": getattr(vllm_envs, "VLLM_USE_RAY", None),
                "VLLM_ATTENTION_BACKEND": getattr(vllm_envs, "VLLM_ATTENTION_BACKEND", None),
            }
        except Exception as e:
            vllm_env_report = {"error": str(e)}

    # 6) Prompt shape quick check
    prompt_report = {}
    if prompts is not None:
        sample_p = prompts[0] if isinstance(prompts, (list, tuple)) and prompts else prompts
        prompt_report = {
            "num_prompts": len(prompts) if isinstance(prompts, (list, tuple)) else 1,
            "first_prompt_preview": str(sample_p)[:280].replace("\n", "\\n"),
            "first_prompt_len_chars": len(str(sample_p)),
        }

    # 7) Aggregate and pretty print
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "env": env_report,
        "versions": vers_report,
        "torch_flags": tf32,
        "tokenizer": tok_report,
        "sampling_params": sp_report,
        "vllm_engine": engine_report,
        "vllm_envs_parsed": vllm_env_report,
        "prompt_info": prompt_report,
    }
    print(json.dumps(report, indent=2))
    print("="*30 + " END DEBUG SNAPSHOT " + "="*30 + "\n")

# ---------------------------------- end debugging vLLM multiprocessing issues ----------------------------------


def vllm_infer(
    model_name_or_path: str,
    adapter_name_or_path: str = None,
    dataset: str = "alpaca_en_demo",
    dataset_dir: str = "data",
    template: str = "default",
    cutoff_len: int = 2048,
    max_samples: Optional[int] = None,
    vllm_config: str = "{}",
    save_name: str = "generated_predictions.jsonl",
    save_path: str = "./",
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = -1,
    max_new_tokens: int = 1024,
    repetition_penalty: float = 1.0,
    skip_special_tokens: bool = True,
    seed: Optional[int] = 42,
    pipeline_parallel_size: int = 1,
    image_max_pixels: int = 768 * 768,
    image_min_pixels: int = 32 * 32,
    grade: int = 7
):
    r"""Perform batch generation using vLLM engine, which supports tensor parallelism.

    Usage: python vllm_infer.py --model_name_or_path meta-llama/Llama-2-7b-hf --template llama --dataset alpaca_en_demo
    """
    check_version("vllm>=0.4.3,<=0.7.3")
    if pipeline_parallel_size > get_device_count():
        raise ValueError("Pipeline parallel size should be smaller than the number of gpus.")

    run_id = os.getenv("WANDB_RUN_ID") or None

    init_kwargs = dict(
                        project=os.environ.get("WANDB_PROJECT"),
                        #entity=os.environ.get("WANDB_ENTITY") or None,
                        id=run_id,
                        resume="allow" if run_id else "never",
                        name=os.environ.get("WANDB_NAME"), # do not set if id is present
                        group=os.environ.get("WANDB_RUN_GROUP"),
                        job_type=os.environ.get("WANDB_JOB_TYPE"),
                        dir=os.environ.get("WANDB_DIR"),
                        config={
                            "train_variant": os.environ.get("TRAIN_VARIANT", "cleaned"),
                            "infer_variant": os.environ.get("INFER_VARIANT", "cleaned"),
                            "grade": int(os.environ.get("INFER_GRADE", "0")),
                            "palette": {
                                "original": "#1f77b4",
                                "cleaned": "#2ca02c",
                                "augmented": "#d62728"
                                        },
                                },
                        settings=wandb.Settings(
                                                init_timeout=300,
                                                _service_wait=300,
                                                ),
                        )

    if run_id is None:
        init_kwargs["name"] = os.getenv("WANDB_NAME", f"{os.getenv('variation','var')}-baseline-")

    run = wandb.init(**init_kwargs)

    print(adapter_name_or_path)
    if adapter_name_or_path is not None:
        parent_id_path = os.path.join(adapter_name_or_path, "wandb_parent_id.txt")
        if os.path.exists(parent_id_path):
            with open(parent_id_path, 'r', encoding='utf-8') as f:
                parent_id = f.read().strip()
            # store as config for filtering and as summary for quick viewing
            wandb.config.update({"parent_run_id": parent_id}, allow_val_change=True)
            wandb.run.summary["parent_run_id"] = parent_id
    print(parent_id)

    model_args, data_args, _, generating_args = get_infer_args(
        dict(
            model_name_or_path=model_name_or_path,
            adapter_name_or_path=adapter_name_or_path,
            dataset=dataset,
            dataset_dir=dataset_dir,
            template=template,
            cutoff_len=cutoff_len,
            max_samples=max_samples,
            preprocessing_num_workers=16,
            vllm_config=vllm_config,
            temperature=0, # temperature,
            top_p=1.0, # top_p,
            top_k=-1, # top_k,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
    )

    training_args = Seq2SeqTrainingArguments(output_dir="dummy_dir")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)
    template_obj.mm_plugin.expand_mm_tokens = False  # for vllm generate
    dataset_module = get_dataset(template_obj, model_args, data_args, training_args, "ppo", **tokenizer_module)

    inputs, prompts, labels = [], [], []
    for sample in dataset_module["train_dataset"]:
        if sample["images"]:
            multi_modal_data = {
                "image": template_obj.mm_plugin._regularize_images(
                    sample["images"], image_max_pixels=image_max_pixels, image_min_pixels=image_min_pixels
                )
            }
        else:
            multi_modal_data = None

        inputs.append({"prompt_token_ids": sample["input_ids"], "multi_modal_data": multi_modal_data})
        prompts.append(tokenizer.decode(sample["input_ids"], skip_special_tokens=skip_special_tokens))
        labels.append(
            tokenizer.decode(
                list(filter(lambda x: x != IGNORE_INDEX, sample["labels"])), skip_special_tokens=skip_special_tokens
            )
        )

    sampling_params = SamplingParams(
        repetition_penalty=generating_args.repetition_penalty or 1.0,  # repetition_penalty must > 0
        temperature=0, # generating_args.temperature,
        top_p=1.0, # generating_args.top_p or 1.0,  # top_p must > 0
        top_k=-1, # generating_args.top_k or -1,  # top_k must > 0
        stop_token_ids=template_obj.get_stop_token_ids(tokenizer),
        max_tokens=generating_args.max_new_tokens,
        skip_special_tokens=skip_special_tokens,
        seed=seed,
    )
    if model_args.adapter_name_or_path is not None:
        lora_request = LoRARequest("default", 1, model_args.adapter_name_or_path[0])
    else:
        lora_request = None

    engine_args = {
        "model": model_args.model_name_or_path,
        "trust_remote_code": True,
        "dtype": model_args.infer_dtype,
        "max_model_len": cutoff_len + max_new_tokens,
        "tensor_parallel_size": (get_device_count() // pipeline_parallel_size) or 1,
        "pipeline_parallel_size": pipeline_parallel_size,
        "disable_log_stats": True,
        "enable_lora": model_args.adapter_name_or_path is not None,
        "gpu_memory_utilization": 0.5,
    }
    if template_obj.mm_plugin.__class__.__name__ != "BasePlugin":
        engine_args["limit_mm_per_prompt"] = {"image": 4, "video": 2}

    if isinstance(model_args.vllm_config, dict):
        engine_args.update(model_args.vllm_config)


    score_dict = {"sari": [], "perplexity": [], "fkgl": [], "dfkgl": [], "bert_F1": []}


    # Debugging snapshot before generation
    print_repro_debug(LLM, tokenizer, sampling_params, prompts, label="BEFORE")
    results = LLM(**engine_args).generate(inputs, sampling_params, lora_request=lora_request)
    # Debugging snapshot after generation
    print_repro_debug(LLM, tokenizer, sampling_params, prompts, label="AFTER")


    preds = [result.outputs[0].text for result in results]
    
    system_prompt = f"user\n\nRewrite this Input sentence to make it easily understandable by students in Grade {grade}"
    sources = [prompt.removeprefix(system_prompt).removesuffix("assistant\n\n") for prompt in prompts]

    with open(f"{save_path}/generated_predictions/{save_name}_source-pred-label.jsonl", "w", encoding="utf-8") as f:
        for text, pred, label in zip(sources, preds, labels):
            sari_score = sari.compute(sources=[text], predictions=[pred], references=[[label]])
            score_dict["sari"].append(sari_score['sari'])
            f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")

    print("*" * 70)
    print(f"{len(prompts)} generated results have been saved at {save_path}/{save_name}.")
    print("*" * 70)

    metrics = {k: round(float(np.mean(v)), 2) for k, v in score_dict.items()}

    for _, (p, inp, out, lbl, src) in enumerate(zip(prompts, inputs, preds, labels, sources)):
        print(f"Prompt: {p}")
        print(f"Source: {src}")
        print(f"Input: {inp}")
        print(f"Pred: {out}")
        print(f"Label: {lbl}")
        print("-" * 80)

    text = f"\n".join(preds)
    metrics["fkgl"] = textstat.flesch_kincaid_grade(text)
    metrics["dfkgl"] = round(abs(float(metrics["fkgl"]) - float(grade)), 2)

    bert_precision, bert_recall, bert_F1 = score(preds, sources, lang='en', verbose=True)
    metrics["bert_F1"] = round(float(np.mean(bert_F1.numpy() * 100)), 2)

    print("*" * 70)
    #print("Readability results:", fk)
    print("Textstat FKGL:", metrics["fkgl"])
    print("*" * 70)

    # need to free up memory...? model -> cpu, call perplexity, which includes model -> gpu
    # import torch

    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    # else:
    #     device = torch.device("cpu")

    # allocated_memory = torch.cuda.memory_allocated(device) / (1024**2) # Convert to MB
    # reserved_memory = torch.cuda.memory_reserved(device) / (1024**2) # Convert to MB
    # max_reserved_memory = torch.cuda.max_memory_reserved(device) / (1024**2) # Convert to MB

    # print(f"Allocated memory: {allocated_memory:.2f} MB")
    # print(f"Reserved memory: {reserved_memory:.2f} MB")
    # print(f"Max reserved memory: {max_reserved_memory:.2f} MB")    
    
    # print("Attempting to load model+adapter")

    #if adapter_name_or_path is not None:
    #    config = PeftConfig.from_pretrained(adapter_name_or_path)
      
    
    #model = LlamaForCausalLM.from_pretrained(
        #model_name_or_path,
        #torch_dtype='auto',
        #device_map='auto',
        #offload_folder="offload", offload_state_dict = True
        #)
    
    #print("Base model loaded...")

    # Load the Lora model
    #if adapter_name_or_path is not None: 
        #model = PeftModel.from_pretrained(model, adapter_name_or_path)
    
        #print("Adapter successfully loaded!")

    #perplexity_results = perplexity.compute(predictions=labels, model=model, tokenizer=tokenizer)
    from stat_utils.cal_ppl import calculate_ppl
    ppl_save_path = save_path if adapter_name_or_path is None else adapter_name_or_path

    perplexity_results = calculate_ppl(model_name_or_path=model_name_or_path,
                                        adapter_name_or_path=adapter_name_or_path,
                                        save_path=ppl_save_path,
                                        dataset=dataset,
                                        template=template,
                                        )
    
    metrics["perplexity"] = round(perplexity_results, 2)

    def _read_global_step(output_dir: str) -> int | None:
        try:
            with open(os.path.join(output_dir, "trainer_state.json"), "r", encoding="utf-8") as f:
                state = json.load(f)
            return int(state.get("global_step") or state.get("global_steps") or 0)
        except Exception:
            return None

    def _py_scalar(x):
        # Coerce numpy types to Python scalars so summary accepts them cleanly
        if isinstance(x, (np.generic,)):
            return x.item()
        return x

    pref = f'infer/{run.config["infer_variant"]}/grade/{run.config["grade"]}'
    payload = {f"{pref}/{k}": v for k, v in metrics.items()}

    step = _read_global_step(adapter_name_or_path) if adapter_name_or_path else None
    wandb.log(payload, step=(step if step is not None else 0))

    # Keep a summary copy for quick table viewing
    #wandb.run.summary.update({f"{pref}/{k}": v for k, v in metrics.items()}
                             
    run.summary.update({f"{pref}/{k}": _py_scalar(v) for k, v in metrics.items()})

    train_v = run.config["train_variant"]; inver_v = run.config["infer_variant"]
    run.summary.update({f"matrix/{train_v}/{inver_v}/{k}": _py_scalar(v) for k, v in metrics.items()})

    if adapter_name_or_path is not None:
        predictions_path = os.path.join(adapter_name_or_path, save_name)
        if os.path.exists(predictions_path):
            wandb.save(predictions_path)

    run.finish()

    with open(f"{save_path}/{save_name}.metrics", "w", encoding="utf-8") as f:
        f.write(json.dumps(metrics))

    print("*" * 70)
    print(f'Metrics written to {save_path}/{save_name}.metrics:')
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("*" * 70)

if __name__ == "__main__":
    fire.Fire(vllm_infer)
