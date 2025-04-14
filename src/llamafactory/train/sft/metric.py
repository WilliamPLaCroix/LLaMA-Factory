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
from transformers.utils import is_jieba_available, is_nltk_available

from ...extras.constants import IGNORE_INDEX
from ...extras.misc import numpify
from ...extras.packages import is_rouge_available


if TYPE_CHECKING:
    from transformers import EvalPrediction, PreTrainedTokenizer

from evaluate import load
sari = load("sari")

import textstat

import torch.nn as nn

def compute_loss(logits, labels):

    logits = torch.tensor(logits, dtype=torch.float32).cpu().detach()
    logits = logits.view(-1, logits.size(-1))  # [batch_size * seq_len, vocab_size]
    labels = torch.tensor(labels, dtype=torch.long).cpu().detach()
    labels = labels.view(-1)                   # [batch_size * seq_len]
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
    loss = loss_fn(logits, labels)
    del logits, labels, loss_fn
    torch.cuda.empty_cache()
    return loss

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
        print("beginning of eval", torch.cuda.memory_summary())
        eval_predictions = eval_preds.predictions[:, :-1, :]
        #predictions = eval_predictions.argmax(dim=-1).cpu().detach()
        predictions = torch.tensor(eval_predictions).argmax(dim=-1).cpu().detach()
        
        label_ids = eval_preds.label_ids[:, 1:]

        preds, labels, inputs = numpify(predictions), numpify(label_ids), numpify(eval_preds.inputs)
        #print("after numpify", torch.cuda.memory_summary())
        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)
        inputs = np.where(inputs != IGNORE_INDEX, inputs, self.tokenizer.pad_token_id)
        
        self.tokenizer.padding_side = "left"
        #print("after unmasking", torch.cuda.memory_summary())
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_inputs = self.tokenizer.batch_decode(inputs, skip_special_tokens=True)
        #print("after decode", torch.cuda.memory_summary())
        for pred, label, source in zip(decoded_preds, decoded_labels, decoded_inputs):
            source = source[91:].split("\n")[0][:-9]
            sari_score = sari.compute(sources=[source], predictions=[pred], references=[[label]])
            self.score_dict["sari"].append(round(sari_score['sari'], 2))
        #print("after SARI", torch.cuda.memory_summary())

        self.score_dict = {k: float(np.mean(v)) for k, v in self.score_dict.items()}
        text = " ".join(decoded_preds)
        self.score_dict["fkgl"] = textstat.flesch_kincaid_grade(text)
        #print("before compute loss", torch.cuda.memory_summary())
        loss = compute_loss(eval_predictions, label_ids).to("cpu").item()
        #print("after compute loss", torch.cuda.memory_summary())
        self.score_dict["loss"] = loss
        self.score_dict["perplexity"] = torch.exp(loss)
        torch.cuda.empty_cache()
        print("after empty cache", torch.cuda.memory_summary())
        if compute_result:
            return self._dump()
