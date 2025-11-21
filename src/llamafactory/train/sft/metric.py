# Copyright 2025 HuggingFace Inc., THUDM, and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library and the THUDM's ChatGLM implementation.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
# https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/main.py
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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from ...extras.constants import IGNORE_INDEX
from ...extras.misc import numpify


if TYPE_CHECKING:
    from transformers import EvalPrediction, PreTrainedTokenizer

from evaluate import load
sari = load("sari")

import textstat
from bert_score import score

# import torch.nn as nn
# import math

### ----------------- Helpers for dumping predicitons -------------------------- ### 
import os, json
from pathlib import Path

def _is_main_process() -> bool:
    # works for single-GPU, DDP, FSDP
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True

def _decode_preds_like_hf(tokenizer, predictions: np.ndarray) -> list[str]:
    # Handle either token ids or logits
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    if predictions.ndim == 3:
        # [bsz, seq, vocab] -> greedy ids
        predictions = predictions.argmax(-1)
    predictions = np.where(predictions != IGNORE_INDEX, predictions, tokenizer.pad_token_id)
    return tokenizer.batch_decode(predictions, skip_special_tokens=True)

def _decode_labels(tokenizer, labels: np.ndarray) -> list[str]:
    labels = np.where(labels != IGNORE_INDEX, labels, tokenizer.pad_token_id)
    return tokenizer.batch_decode(labels, skip_special_tokens=True)

def _try_parse_json(s: str):
    try:
        return json.loads(s)
    except Exception:
        return s
##################################################################################


@dataclass
class ComputeAccuracy:
    r"""Compute accuracy and support `batch_eval_metrics`."""

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"accuracy": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)
        for i in range(len(preds)):
            pred, label = preds[i, :-1], labels[i, 1:]
            label_mask = label != IGNORE_INDEX
            self.score_dict["accuracy"].append(np.mean(pred[label_mask] == label[label_mask]))

        if compute_result:
            return self._dump()

@dataclass
class ComputeSimilarity:
    r"""Compute text similarity scores and support `batch_eval_metrics`.

    Wraps the tokenizer into metric functions, used in CustomSeq2SeqTrainer.
    """

    tokenizer: "PreTrainedTokenizer"

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}
            #result = self.score_dict
        self.score_dict = {"fkgl": [], "fkgl-delta": [], "sari": [], "dfkgl_sari": [], "bert_f1": []} # , "loss": [], "perplexity": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        # preds = eval_preds.predictions#[:, :-1, :].cpu().detach()
        # inputs = eval_preds.inputs#.cpu().detach()
        # labels = eval_preds.label_ids#[:, 1:].cpu().detach()

        # loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
        # self.score_dict["loss"] = loss_fn(preds.view(-1, preds.size(-1)), labels.view(-1)  ).cpu().detach().item()
        # self.score_dict["perplexity"] = math.exp(self.score_dict["loss"])

        #preds = np.argmax(preds, axis=-1)

        raw_inputs = np.where(numpify(eval_preds.inputs) != IGNORE_INDEX, numpify(eval_preds.inputs), self.tokenizer.pad_token_id)

        preds = numpify(eval_preds.predictions)
        labels = numpify(eval_preds.label_ids)
        inputs = numpify(eval_preds.inputs)

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)
        inputs = np.where(inputs != IGNORE_INDEX, inputs, self.tokenizer.pad_token_id)
        
        # self.tokenizer.padding_side = "left"
        preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        inputs = self.tokenizer.batch_decode(inputs, skip_special_tokens=True)

        sources = [source.split("\n")[3][:-9] for source in inputs] # remove the "assistant" on end of string
        grades = [int(source.split("\n")[2].split(" ")[-1].strip('.')) for source in inputs] # get the grade from the input prompt
        # print(preds)
        preds = [pred.removeprefix("assistant").removeprefix("\n").removeprefix("\n") for pred in preds] # remove the "assistant" at beginning of string
        labels = [[label] for label in labels]

        self.score_dict["sari"] = sari.compute(sources=sources, predictions=preds, references=labels)['sari']
        bert_precision, bert_recall, bert_F1 = score(preds, sources, lang='en', verbose=True)
        self.score_dict["bert_F1"] = round(float(np.mean(bert_F1.numpy() * 100)), 2)

        # for pred, label, source, raw_input in zip(preds, labels, inputs, raw_inputs):
        #     prompt = source
        #     source = source.split("\n") # source includes system prompt and input, need to separate
        #     grade = int(source[2].split(" ")[-1].strip('.')) # get the grade from the input prompt
        #     grades.append(grade)
        #     source = source[3][:-9] # remove the "assistant" on end of string
        #     pred = pred.split("\n\n")[1] # remove the "assistant" at beginning of string
        #     print("Prompt:", prompt, "\nSource:", source, "\nInput:", raw_input, "\nPred:", pred, "\nLabel:", label, "\n", "-"*80) # mimic vllm infer log output
        #     # print('{'+f'"prompt": "{source}", "predict": "{pred}", "label": "{label}"'+'}') # mimic jsonl format for analysis
        #     sari_score = sari.compute(sources=[source], predictions=[pred], references=[[label]])
        #     self.score_dict["sari"].append(sari_score['sari'])

        

        #self.score_dict = {k: float(np.mean(v)) for k, v in self.score_dict.items()}
        fkgl = textstat.flesch_kincaid_grade("\n".join(preds))
        self.score_dict["fkgl"].append(fkgl)
        target_grade = sum(grades) / len(grades)
        fkgl_delta = abs(fkgl - target_grade)
        self.score_dict["fkgl-delta"].append(fkgl_delta)

        def compute_fkgl_x_sari(fkgl_delta, fkgl_alpha=0.5):
            sari_mean = np.mean(self.score_dict["sari"])
            sari_beta = 1 - fkgl_alpha
            return 100 - sari_beta * (100 - sari_mean) - 10 * fkgl_alpha * fkgl_delta

        self.score_dict["dfkgl_sari"].append(compute_fkgl_x_sari(fkgl_delta, fkgl_alpha=0.5))

        # NEW: one-shot JSONL dump at the end of eval, main process only
        if compute_result and _is_main_process():
            dump_path = os.getenv("LF_DUMP_JSONL")  # e.g., export LF_DUMP_JSONL=$OUTPUT_DIR/eval_predictions.jsonl
            if dump_path:
                out = Path(dump_path)
                out.parent.mkdir(parents=True, exist_ok=True)
                with out.open("w", encoding="utf-8") as f:
                    for i, p in enumerate(preds):
                        row = {
                            "id": i,
                            "tgt_grade": grades[i],
                            "source": sources[i],
                            "pred": _try_parse_json(p),
                            "label": labels[i][0],
                        }
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")

        if compute_result:
            return self._dump()
