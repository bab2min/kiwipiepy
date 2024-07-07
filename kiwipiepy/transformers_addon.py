'''
.. versionadded:: 0.15.1

`transformers_addon` 모듈은 Kiwi의 SwTokenizer를 
[huggingface transformers의 tokenizer](https://huggingface.co/docs/transformers/main_classes/tokenizer)와 
호환이 가능하도록 래핑한 `KiwiTokenizer` 클래스를 제공합니다.

이 기능을 사용하기 위해서는 `transformers>=4.12` 이 필요합니다.

```python
from transformers import AutoTokenizer

import kiwipiepy.transformers_addon
# kiwipiepy.transformers_addon를 import하면
# 자동으로 KiwiTokenizer가 AutoTokenizer에 register됩니다.
# KiwiTokenizer는 huggingface transformers의 토크나이저와 대부분의 기능이 호환되므로
# 기존 transformers 기반의 코드를 그대로 재사용할 수 있습니다.

# SwTokenizer가 some_path/tokenizer.json에 저장되어있다고 가정
tokenizer = AutoTokenizer.from_pretrained('some_path')
tokenizer.encode("한국어를 고려한 토크나이저!")
```

`KiwiTokenizer`의 주요 기능들은 transformers와 대부분 호환됩니다만, 다음 기능들은 현재 호환되지 않습니다.

* `add_tokens`, `add_special_tokens` 등 새 토큰을 추가하는 기능
* `encode_plus`의  `stride`, `is_split_into_words`, `return_overflowing_tokens`, `return_special_tokens_mask`, `return_length` 인자
'''


import os
import itertools
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

def _group_by_two(iterator):
    try:
        while True:
            a = next(iterator)
            b = next(iterator)
            yield a, b
    except StopIteration:
        pass

class KiwiTokenizer(PreTrainedTokenizerBase):

    vocab_files_names = {"tokenizer_file": "tokenizer.json"}

    def __init__(self, tokenizer_file=None, **kwargs):
        if tokenizer_file is None:
            raise ValueError(f"Cannot instantiate tokenizer from {tokenizer_file!r}")
        
        self._tokenizer = SwTokenizer(tokenizer_file)
        
        super().__init__(**kwargs)

        self._post_processor = self._tokenizer.config.additional.get('post_processor') if isinstance(self._tokenizer.config.additional, dict) else None
        if self._post_processor not in (None, 'bert'):
            raise ValueError(f"Unknown post_processor `{self._post_processor!r}`")

        self._bos_token = self._tokenizer.bos_token
        self._eos_token = self._tokenizer.eos_token
        self._unk_token = self._tokenizer.unk_token
        self._sep_token = self._tokenizer.sep_token
        self._pad_token = self._tokenizer.pad_token
        self._cls_token = self._tokenizer.cls_token
        self._mask_token = self._tokenizer.mask_token
    
    @property
    def unk_token(self) -> str:
        return self._tokenizer.unk_token
    
    @unk_token.setter
    def unk_token(self, s):
        if s != self._tokenizer.unk_token:
            raise AttributeError("can't set attribute 'unk_token'")

    @property
    def cls_token(self) -> str:
        return self._tokenizer.cls_token

    @cls_token.setter
    def cls_token(self, s):
        if s != self._tokenizer.cls_token:
            raise AttributeError("can't set attribute 'cls_token'")

    @property
    def sep_token(self) -> str:
        return self._tokenizer.sep_token
    
    @sep_token.setter
    def sep_token(self, s):
        if s != self._tokenizer.sep_token:
            raise AttributeError("can't set attribute 'sep_token'")

    @property
    def pad_token(self) -> str:
        return self._tokenizer.pad_token

    @pad_token.setter
    def pad_token(self, s):
        if s != self._tokenizer.pad_token:
            raise AttributeError("can't set attribute 'pad_token'")

    @property
    def mask_token(self) -> str:
        return self._tokenizer.mask_token
    
    @mask_token.setter
    def mask_token(self, s):
        if s != self._tokenizer.mask_token:
            raise AttributeError("can't set attribute 'mask_token'")

    @property
    def bos_token(self) -> str:
        return self._tokenizer.bos_token
    
    @bos_token.setter
    def bos_token(self, s):
        if s != self._tokenizer.bos_token:
            raise AttributeError("can't set attribute 'bos_token'")

    @property
    def eos_token(self) -> str:
        return self._tokenizer.eos_token

    @eos_token.setter
    def eos_token(self, s):
        if s != self._tokenizer.eos_token:
            raise AttributeError("can't set attribute 'eos_token'")

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
        **kwargs,
    ) -> BatchEncoding:
        
        if return_token_type_ids is None:
            return_token_type_ids = "token_type_ids" in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        if not isinstance(batch_text_or_text_pairs, (list, tuple)):
            raise TypeError(f"batch_text_or_text_pairs has to be a list (got {type(batch_text_or_text_pairs)})")

        input_ids, attention_mask, token_type_ids, offset_mapping = self._make_encoded(
            batch_text_or_text_pairs, add_special_tokens, 
            return_token_type_ids, return_attention_mask, return_offsets_mapping,
            return_as_list=(return_tensors is None),
            padding_strategy=padding_strategy, 
            truncation_strategy=truncation_strategy, 
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
        )
        data = dict(input_ids=input_ids)
        if return_attention_mask: data['attention_mask'] = attention_mask
        if return_token_type_ids: data['token_type_ids'] = token_type_ids
        if return_offsets_mapping: data['offset_mapping'] = offset_mapping

        for i in input_ids:
            self._eventual_warn_about_too_long_sequence(i, max_length, verbose)
        return BatchEncoding(data, tensor_type=return_tensors)

    def _encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput, EncodedInput],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
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
        **kwargs,
    ) -> BatchEncoding:
        
        if return_token_type_ids is None:
            return_token_type_ids = "token_type_ids" in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        text = text if text_pair is None else (text, text_pair)
        input_ids, attention_mask, token_type_ids, offset_mapping = self._make_encoded(
            [text], add_special_tokens, 
            return_token_type_ids, return_attention_mask, return_offsets_mapping,
            return_as_list=(return_tensors is None),
            padding_strategy=padding_strategy, 
            truncation_strategy=truncation_strategy, 
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
        )

        if return_tensors is None and not return_overflowing_tokens:
            input_ids = input_ids[0]
            if return_attention_mask: attention_mask = attention_mask[0]
            if return_token_type_ids: token_type_ids = token_type_ids[0]
            if return_offsets_mapping: offset_mapping = offset_mapping[0]
            self._eventual_warn_about_too_long_sequence(input_ids, max_length, verbose)
        else:
            self._eventual_warn_about_too_long_sequence(input_ids[0], max_length, verbose)

        data = dict(input_ids=input_ids)
        if return_attention_mask: data['attention_mask'] = attention_mask
        if return_token_type_ids: data['token_type_ids'] = token_type_ids
        if return_offsets_mapping: data['offset_mapping'] = offset_mapping
            
        return BatchEncoding(data, tensor_type=return_tensors)

    def _make_encoded(
        self, 
        batch_text_or_text_pairs, 
        add_special_tokens, 
        return_token_type_ids, 
        return_attention_mask,
        return_offsets_mapping,
        return_as_list = False,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ):
        input_ids = []
        attention_mask = []
        token_type_ids = []
        offset_mapping = []
        if isinstance(batch_text_or_text_pairs[0], str): # single
            special_token_size = self.num_special_tokens_to_add(False) if add_special_tokens else 0

            for i in self._tokenizer.encode(batch_text_or_text_pairs, return_offsets=return_offsets_mapping):
                if return_offsets_mapping:
                    i, i_offset = i
                    i_offset = i_offset.astype(np.int64)
                i = i.astype(np.int64)
                if (truncation_strategy in (TruncationStrategy.LONGEST_FIRST, TruncationStrategy.ONLY_FIRST) 
                    and len(i) > max_length - special_token_size):
                    i = i[:max_length - special_token_size]
                    if return_offsets_mapping:
                        i_offset = i_offset[:max_length - special_token_size]

                if add_special_tokens and self._post_processor == 'bert':
                    i = np.pad(i.astype(np.int64), (1, 1))
                    i[0] = self._tokenizer.cls_token_id
                    i[-1] = self._tokenizer.sep_token_id
                    if return_offsets_mapping:
                        i_offset = np.pad(i_offset, ((1, 1), (0, 0)))
                input_ids.append(i)
                if return_attention_mask: attention_mask.append(np.ones_like(i))
                if return_token_type_ids: token_type_ids.append(np.zeros_like(i))
                if return_offsets_mapping: offset_mapping.append(i_offset)
        
        else: # pair
            special_token_size = self.num_special_tokens_to_add(True) if add_special_tokens else 0

            for i, j in _group_by_two(self._tokenizer.encode(itertools.chain.from_iterable(batch_text_or_text_pairs), return_offsets=return_offsets_mapping)):
                if return_offsets_mapping:
                    i, i_offset = i
                    j, j_offset = j
                    i_offset = i_offset.astype(np.int64)
                    j_offset = j_offset.astype(np.int64)
                if (truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE
                    and len(i) + len(j) > max_length - special_token_size):
                    if truncation_strategy == TruncationStrategy.LONGEST_FIRST:
                        t = len(i) + len(j) - (max_length - special_token_size)
                        d = abs(len(i) - len(j))
                        trunc_size = min(d, t)
                        if len(i) > len(j):
                            i = i[:-trunc_size]
                            if return_offsets_mapping:
                                i_offset = i_offset[:-trunc_size]
                        else:
                            j = j[:-trunc_size]
                            if return_offsets_mapping:
                                j_offset = j_offset[:-trunc_size]
                        
                        if t > d:
                            i = i[:-((t - d + 1) // 2)]
                            j = j[:-((t - d) // 2)]
                            if return_offsets_mapping:
                                i_offset = i_offset[:-((t - d + 1) // 2)]
                                j_offset = j_offset[:-((t - d) // 2)]
                    elif truncation_strategy == TruncationStrategy.ONLY_FIRST:
                        i = i[:max(max_length - special_token_size - len(j), 0)]
                        if return_offsets_mapping:
                            i_offset = i_offset[:max(max_length - special_token_size - len(j), 0)]
                    elif truncation_strategy == TruncationStrategy.ONLY_SECOND:
                        j = j[:max(max_length - special_token_size - len(i), 0)]
                        if return_offsets_mapping:
                            j_offset = j_offset[:max(max_length - special_token_size - len(i), 0)]

                if add_special_tokens and self._post_processor == 'bert':
                    c = np.concatenate([[self._tokenizer.cls_token_id], i, [self._tokenizer.sep_token_id], j, [self._tokenizer.sep_token_id]])
                    t = (np.arange(len(c)) >= len(i) + 2).astype(np.int64)
                    if return_offsets_mapping:
                        c_offset = np.concatenate([np.pad(i_offset, ((1, 1), (0, 0))), np.pad(j_offset, ((0, 1), (0, 0)))], axis=0)
                else:
                    c = np.concatenate([i, j])
                    t = (np.arange(len(c)) >= len(i)).astype(np.int64)
                    if return_offsets_mapping:
                        c_offset = np.concatenate([i_offset, j_offset], axis=0)
                input_ids.append(c)
                if return_attention_mask: attention_mask.append(np.ones_like(c))
                if return_token_type_ids: token_type_ids.append(t)
                if return_offsets_mapping: offset_mapping.append(c_offset)

        if padding_strategy == PaddingStrategy.LONGEST:
            final_length = max(map(len, input_ids))
            if pad_to_multiple_of: final_length = ((final_length + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
        elif padding_strategy == PaddingStrategy.MAX_LENGTH:
            final_length = max(max(map(len, input_ids)), max_length or 0)
            if pad_to_multiple_of: final_length = ((final_length + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
        else:
            final_length = None
        
        if final_length:
            for i in range(len(input_ids)):
                input_ids[i] = np.pad(input_ids[i], (0, final_length - len(input_ids[i])), constant_values=self._tokenizer.pad_token_id)
                
                try: attention_mask[i] = np.pad(attention_mask[i], (0, final_length - len(attention_mask[i])))
                except IndexError: pass
                
                try: token_type_ids[i] = np.pad(token_type_ids[i], (0, final_length - len(token_type_ids[i])))
                except IndexError: pass
                
                try: offset_mapping[i] = np.pad(offset_mapping[i], ((0, final_length - len(offset_mapping[i])), (0, 0)))
                except IndexError: pass

        if return_as_list:
            input_ids = list(map(np.ndarray.tolist, input_ids))
            attention_mask = list(map(np.ndarray.tolist, attention_mask))
            token_type_ids = list(map(np.ndarray.tolist, token_type_ids))
            offset_mapping = list(map(lambda x:list(map(tuple, x)), (map(np.ndarray.tolist, offset_mapping))))

        return input_ids, attention_mask, token_type_ids, offset_mapping

    def _decode(
        self, 
        token_ids, 
        skip_special_tokens = False, 
        clean_up_tokenization_spaces = True,
    ):
        if isinstance(token_ids, int): token_ids = [token_ids]
        if skip_special_tokens:
            token_ids = [i for i in token_ids if i not in self.all_special_ids]
        return self._tokenizer.decode(token_ids)

    def get_added_vocab(self) -> Dict[str, int]:
        return {}
    
    def get_vocab(self):
        return self._tokenizer.vocab
    
    @property
    def vocab(self):
        return self._tokenizer.vocab
    
    @property
    def vocab_size(self) -> int:
        return len(self._tokenizer.vocab)

    def __len__(self):
        return len(self._tokenizer)

    @property
    def is_fast(self) -> bool:
        return False

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
            return 3 if pair else 2
        return 0

    def _add_tokens(self, new_tokens, special_tokens = False) -> int:
        if all(t in self.vocab for t in new_tokens):
            return 0
        raise NotImplementedError("`KiwiTokenizer.add_tokens` is not support yet.")

    def tokenize(
        self, 
        text: str, 
        pair: Optional[str] = None, 
        add_special_tokens: bool = False,
    ) -> List[str]:
        ids = self.encode(text, pair, add_special_tokens=add_special_tokens)
        return list(map(self._tokenizer.id2vocab.__getitem__, ids))

    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]], skip_special_tokens: bool = False
    ) -> Union[str, List[str]]:
        if isinstance(ids, int):
            if skip_special_tokens and ids in self.all_special_ids: return ''
            return self._tokenizer.id2vocab[ids]

        if skip_special_tokens:
            ids = [i for i in ids if i not in self.all_special_ids]
        return list(map(self._tokenizer.id2vocab.__getitem__, ids))

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
            save_directory, (filename_prefix + "-" if filename_prefix else "") + "tokenizer.json"
        )
        self._tokenizer.save(tokenizer_file)
        file_names = file_names + (tokenizer_file,)

        return file_names

    @property
    def added_tokens_decoder(self):
        return {}

AutoTokenizer.register('KiwiTokenizer', None, KiwiTokenizer)
