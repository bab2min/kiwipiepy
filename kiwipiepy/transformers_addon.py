import os
from typing import Union, List, Optional, Dict, Tuple

import numpy as np

from transformers import AutoTokenizer

from transformers.tokenization_utils_base import (
    PreTrainedTokenizerBase, 
    TextInput, 
    TextInputPair, 
    PreTokenizedInput, 
    PreTokenizedInputPair, 
    EncodedInput, 
    EncodedInputPair,
    PaddingStrategy,
    TruncationStrategy,
    TensorType,
    BatchEncoding,    
)

from kiwipiepy.sw_tokenizer import SwTokenizer, SwTokenizerConfig

class KiwiTokenizer(PreTrainedTokenizerBase):

    vocab_files_names = {"tokenizer_file": "tokenizer.json"}

    def __init__(self, tokenizer_file=None, **kwargs):
        super().__init__(**kwargs)
        if tokenizer_file is None:
            raise ValueError(f"Cannot instantiate tokenizer from {tokenizer_file!r}")
        
        self._tokenizer = SwTokenizer(tokenizer_file)
        self._post_processor = self._tokenizer.config.additional.get('post_processor') if self._tokenizer.config.additional else None
    
    @property
    def unk_token(self) -> str:
        return self._tokenizer.unk_token

    @property
    def cls_token(self) -> str:
        return self._tokenizer.cls_token

    @property
    def sep_token(self) -> str:
        return self._tokenizer.sep_token
    
    @property
    def pad_token(self) -> str:
        return self._tokenizer.pad_token

    @property
    def mask_token(self) -> str:
        return self._tokenizer.mask_token
    
    @property
    def bos_token(self) -> str:
        return self._tokenizer.bos_token
    
    @property
    def eos_token(self) -> str:
        return self._tokenizer.eos_token

    @property
    def unk_token_id(self) -> str:
        return self._tokenizer.unk_token_id

    @property
    def cls_token_id(self) -> str:
        return self._tokenizer.cls_token_id

    @property
    def sep_token_id(self) -> str:
        return self._tokenizer.sep_token_id
    
    @property
    def pad_token_id(self) -> str:
        return self._tokenizer.pad_token_id

    @property
    def mask_token_id(self) -> str:
        return self._tokenizer.mask_token_id
    
    @property
    def bos_token_id(self) -> str:
        return self._tokenizer.bos_token_id
    
    @property
    def eos_token_id(self) -> str:
        return self._tokenizer.eos_token_id

    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
            List[PreTokenizedInputPair],
            List[EncodedInput],
            List[EncodedInputPair],
        ],
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
    ) -> BatchEncoding:
        
        if not isinstance(batch_text_or_text_pairs, list):
            raise TypeError(f"batch_text_or_text_pairs has to be a list (got {type(batch_text_or_text_pairs)})")

        input_ids = []
        attention_mask = []
        token_type_ids = []
        for i in self._tokenizer.encode(batch_text_or_text_pairs):
            if add_special_tokens:
                i = np.pad(i, (1, 1))
                i[0] = self._tokenizer.cls_token_id
                i[-1] = self._tokenizer.sep_token_id
            input_ids.append(i)
            if return_attention_mask: attention_mask.append(np.ones_like(i))
            if return_token_type_ids: token_type_ids.append(np.zeros_like(i))
        
        data = dict(input_ids=input_ids)
        if return_attention_mask: data['attention_mask'] = attention_mask
        if return_token_type_ids: data['token_type_ids'] = token_type_ids

        for i in input_ids:
            self._eventual_warn_about_too_long_sequence(input_ids, max_length, verbose)
        return BatchEncoding(data, tensor_type=return_tensors)

    def get_added_vocab(self) -> Dict[str, int]:
        return {}
    
    def get_vocab(self):
        return self._tokenizer.vocab
    
    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self._tokenizer.vocab.get(tokens, self._tokenizer.unk_token_id)

        ids = []
        for token in tokens:
            ids.append(self._tokenizer.vocab.get(token, self._tokenizer.unk_token_id))
        return ids

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        if self._post_processor == 'bert':
            return 2 if pair else 1
        
        return 0

    def _save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        file_names: Tuple[str],
        legacy_format: Optional[bool] = None,
        filename_prefix: Optional[str] = None,
    ) -> Tuple[str]:
        save_directory = str(save_directory)

        if self.slow_tokenizer_class is None and legacy_format is True:
            raise ValueError(
                "Your tokenizer does not have a legacy version defined and therefore cannot register this version. You "
                "might consider leaving the legacy_format at `None` or setting it to `False`."
            )

        tokenizer_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + "kiwi_tokenizer.json"
        )
        self._tokenizer.save(tokenizer_file)
        file_names = file_names + (tokenizer_file,)

        return file_names

AutoTokenizer.register('KiwiTokenizer', None, KiwiTokenizer)
