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
        self.score_dict = {"sari": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        with torch.no_grad():
            self.score_dict = {"sari": []}
            eval_predictions = eval_preds.predictions[:, :-1, :].cpu().detach()
            predictions = eval_predictions.argmax(dim=-1)
            
            label_ids = eval_preds.label_ids[:, 1:].cpu().detach()

            preds, labels, inputs = numpify(predictions), numpify(label_ids), numpify(eval_preds.inputs).cpu().detach()
            preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
            labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)
            inputs = np.where(inputs != IGNORE_INDEX, inputs, self.tokenizer.pad_token_id)
            
            self.tokenizer.padding_side = "left"
            decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            decoded_inputs = self.tokenizer.batch_decode(inputs, skip_special_tokens=True)
            for pred, label, source in zip(decoded_preds, decoded_labels, decoded_inputs):
                source = source[91:].split("\n")[0][:-9]
                sari_score = sari.compute(sources=[source], predictions=[pred], references=[[label]])
                self.score_dict["sari"].append(round(sari_score['sari'], 2))

            self.score_dict = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

            text = " ".join(decoded_preds)
            self.score_dict["fkgl"] = textstat.flesch_kincaid_grade(text)

            loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
            self.score_dict["loss"] = loss_fn(eval_predictions.view(-1, eval_predictions.size(-1)), label_ids.view(-1)  ).cpu().detach().item()
            self.score_dict["perplexity"] = math.exp(self.score_dict["loss"])
            del eval_predictions, predictions, label_ids, preds, labels, inputs
            del decoded_preds, decoded_labels, decoded_inputs, loss_fn, loss
            print(torch.cuda.memory_summary())
            if compute_result:
                return self._dump()
