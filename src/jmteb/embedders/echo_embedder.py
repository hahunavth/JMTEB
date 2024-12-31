import json
import os
from pathlib import Path

import numpy as np
import torch
from accelerate import PartialState
from accelerate.utils import gather_object
from loguru import logger
from sentence_transformers.models import Pooling
from tqdm.autonotebook import trange
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from jmteb.embedders.base import TextEmbedder


class EchoEmbedder(TextEmbedder):
    def __init__(        
        self,
        model_name_or_path: str,
        batch_size: int = 32,
        device: str | None = None,
        # normalize_embeddings: bool = False,
        max_seq_length: int | None = None,
        # add_eos: bool = False,
        # truncate_dim: int | None = None,
        # pooling_config: str | None = "1_Pooling/config.json",
        # pooling_mode: str | None = None,
        model_kwargs: dict = {},
        tokenizer_kwargs: dict = {},
    ) -> None:
        # model_kwargs = self._model_kwargs_parser(model_kwargs)
        # self.model: PreTrainedModel = AutoModel.from_pretrained(
        #     model_name_or_path, trust_remote_code=True, **model_kwargs
        # )
        self.batch_size = batch_size
        if not device and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = device

        # TODO: distributed
        
        # self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
        # self.max_seq_length = getattr(self.model, "max_seq_length", None)
        # if max_seq_length:
        #     self.max_seq_length = max_seq_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token=os.environ['HF_TOKEN'])

        if "torch_dtype" in model_kwargs:
            self.set_output_tensor()
        else:
            self.set_output_numpy()

        # ECHO EMBEDDING
        from jmteb.embedders.echo_embedding import EchoParser, EchoPooling, EchoEmbeddingsGemma2b
        templates = {
            'query': '<s>Instruct:{!%%prompt%%,}\nQuery:{!%%text%%}\nQuery again:{%%text%%}{</s>}',
            'document': '<s>Document:{!%%text%%}\nDocument again:{%%text%%}{</s>}',
        }
        self.parser = EchoParser(tokenizer=self.tokenizer, templates=templates, max_length=max_seq_length)
        self.pooling = EchoPooling(strategy='last')
        self.echo_embeddings = EchoEmbeddingsGemma2b.from_pretrained(
            model_name_or_path,
            parser=self.parser, 
            pooling=self.pooling,
            trust_remote_code=True,
            token=os.environ['HF_TOKEN']
        )

    def get_output_dim(self) -> int:
        return self.echo_embeddings.model.config.hidden_size

    def encode(
        self,
        text: str | list[str],
        prefix: str | None = None,
        show_progress_bar: bool = True,
    ):
        if isinstance(text, str):
            text = [text]
            text_was_str = True
        else:
            text_was_str = False

        all_embeddings = []
        length_sorted_idx = np.argsort([-len(t) for t in text])
        text_sorted = [text[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(text), self.batch_size, desc="Batches", disable=not show_progress_bar):
            text_batch = text_sorted[start_index : start_index + self.batch_size]
            # if self.distributed_state:
            #     batch_embeddings = self._encode_batch_distributed(text_batch, prefix)
            # else:
            batch_embeddings = self._encode_batch(text_batch, prefix)
            all_embeddings.extend(batch_embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if len(all_embeddings):
            all_embeddings = torch.stack(all_embeddings)
        else:
            all_embeddings = torch.Tensor()

        if text_was_str:
            res = all_embeddings.view(-1)
        else:
            res = all_embeddings

        if self.convert_to_numpy:
            return res.cpu().numpy()
        else:
            return res.cpu()

    def _encode_batch(self, text: list[str], prefix: str | None = None) -> torch.Tensor:
        if prefix:
            # text = [prefix + t for t in text]
            raise NotImplementedError

        tokenized_data = self.parser.tokenize([('document', {'text': t}) for t in text])

        with torch.no_grad():
            embeddings = self.echo_embeddings(tokenized_data)['sentence_embedding']

        return embeddings