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

import torch.nn as nn
import math

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
            #result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}
            result = self.score_dict
        self.score_dict = {"sari": [], "fkgl": [], "fkgl-delta": [], "loss": [], "perplexity": []}
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

        preds, labels, inputs = numpify(eval_preds.predictions), numpify(eval_preds.label_ids), numpify(eval_preds.inputs)

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)
        inputs = np.where(inputs != IGNORE_INDEX, inputs, self.tokenizer.pad_token_id)
        
        self.tokenizer.padding_side = "left"
        preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        inputs = self.tokenizer.batch_decode(inputs, skip_special_tokens=True)

        for pred, label, source in zip(preds, labels, inputs):
            print(source)
            source = source[91:].split("\n")[0][:-9]
            sari_score = sari.compute(sources=[source], predictions=[pred], references=[[label]])
            self.score_dict["sari"].append(sari_score['sari'])

        self.score_dict = {k: float(np.mean(v)) for k, v in self.score_dict.items()}
        fkgl = textstat.flesch_kincaid_grade(" ".join(preds))
        self.score_dict["fkgl"].append(fkgl)

        # print(torch.cuda.memory_summary())

        # import gc

        # print(f"Memory allocated before cleanup: {torch.cuda.memory_allocated()} bytes")
        # print(f"Memory reserved before cleanup: {torch.cuda.memory_reserved()} bytes")

        # print("Before garbage collection:")
        # for obj in gc.get_objects():
        #     if torch.is_tensor(obj):
        #         print(type(obj), obj.size(), obj.device)

        # gc.collect()

        # print("After garbage collection:")
        # for obj in gc.get_objects():
        #     if torch.is_tensor(obj):
        #         print(type(obj), obj.size(), obj.device)
        # torch.cuda.empty_cache()
        
        # # Detach and delete tensors to free memory
        # del preds, preds, labels, inputs
        # del loss_fn, source


        # print(f"Memory allocated after cleanup: {torch.cuda.memory_allocated()} bytes")
        # print(f"Memory reserved after cleanup: {torch.cuda.memory_reserved()} bytes")
        if compute_result:
            return self._dump()
