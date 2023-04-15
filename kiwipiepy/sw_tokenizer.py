import re
import itertools
from typing import Callable, List, Optional, Tuple, Union, Iterable, Dict
from dataclasses import dataclass
from functools import lru_cache
import warnings

import tqdm

from _kiwipiepy import _SwTokenizer

@dataclass
class SwTokenizerConfig:
    lowercase:bool = False
    split_chinese:bool = True
    whole_word_unk:bool = False
    integrate_allomorph: bool = True
    split_punct: bool = True
    simple_tag:bool = True
    split_verb:bool = True
    split_eomi:bool = True
    use_glue_token:bool = True
    use_newline_token:bool = False
    strict:bool = False
    fallback_hangul:bool = True
    fallback_byte:bool = False

    unk_token:str = "[UNK]"
    cls_token:str = None
    sep_token:str = None
    pad_token:str = None
    mask_token:str = None
    bos_token:str = None
    eos_token:str = None

class TrainerCallback:
    def begin_tokenization(self, num_processed_lines:int):
        pass

    def proc_tokenization(self, num_processed_lines:int):
        pass

    def end_tokenization(self, num_processed_lines:int):
        pass

    def begin_reduction(self, iteration:int, cur_vocab_size:int, unigram_loss:float):
        pass

    def proc_reduction(self, iteration:int, cur_vocab_size:int, unigram_loss:float):
        pass

    def end_reduction(self, iteration:int, cur_vocab_size:int, unigram_loss:float):
        pass

class ProgressShower(TrainerCallback):
    def __init__(self, file, total=None, iterations=None):
        super().__init__()
        self._file = file
        self._bar = None
        self._total = total
        self._iterations = iterations
    
    def __del__(self):
        if self._bar:
            self._bar.close()

    def begin_tokenization(self, num_processed_lines: int):
        self._bar = tqdm.tqdm(itertools.repeat(None), desc="Tokenizing", file=self._file, total=self._total)
        self._last_proc_lines = 0
    
    def proc_tokenization(self, num_processed_lines: int):
        self._bar.update(num_processed_lines - self._last_proc_lines)
        self._last_proc_lines = num_processed_lines
    
    def end_tokenization(self, num_processed_lines: int):
        self._bar.close()
        self._bar = None
    
    def begin_reduction(self, iteration: int, cur_vocab_size: int, unigram_loss: float):
        self._bar = tqdm.tqdm(itertools.repeat(None), desc="Reducing", file=self._file, total=self._iterations)
        self._bar.write(f"Iteration: {iteration} VocabSize: {cur_vocab_size} Loss: {unigram_loss:.4f}")
        self._bar.set_postfix(dict(vocab_size=cur_vocab_size, loss=unigram_loss))
        self._last_iteration = iteration
    
    def proc_reduction(self, iteration: int, cur_vocab_size: int, unigram_loss: float):
        self._bar.write(f"Iteration: {iteration} VocabSize: {cur_vocab_size} Loss: {unigram_loss:.4f}")
        self._bar.update(iteration - self._last_iteration)
        self._bar.set_postfix(dict(vocab_size=cur_vocab_size, loss=unigram_loss))
        self._last_iteration = iteration

    def end_reduction(self, iteration: int, cur_vocab_size: int, unigram_loss: float):
        self._bar.close()
        self._bar = None
        print(f"Finished. Iteration: {iteration} VocabSize: {cur_vocab_size} Loss: {unigram_loss:.4f}", file=self._file)

class SwTokenizer(_SwTokenizer):
    def __init__(self,
        path: str,
        kiwi: Optional['Kiwi'] = None,
    ) -> None:
        import kiwipiepy
        if kiwi is None:
            kiwi = kiwipiepy.Kiwi()
        if not isinstance(kiwi, kiwipiepy.Kiwi):
            raise ValueError("`kiwi` must be an instance of `Kiwi`.")

        super().__init__(kiwi, path)
    
    def encode(self, 
        text: Union[str, Iterable[str]],
        return_offsets: Optional[bool] = False,
    ):
        return super().encode(text, return_offsets=return_offsets)
    
    def encode_from_morphs(self, 
        morphs: Iterable[Tuple[str, str, bool]],
    ):
        return super().encode_from_morphs(morphs)

    def decode(self,
        ids: Iterable[int],
    ):
        return super().decode(ids)
    
    @property
    @lru_cache
    def vocab(self) -> Dict[str, int]:
        return super()._vocab

    @property
    @lru_cache
    def config(self) -> SwTokenizerConfig:
        return SwTokenizerConfig(**super()._config)

    def __repr__(self) -> str:
        return super().__repr__()

    @staticmethod
    def train(
        save_path: str,
        texts: Iterable[str],
        config: SwTokenizerConfig,
        vocab_size: int,
        chr_coverage: float = 0.9995,
        prefix_min_cnt: int = 5,
        prefix_max_length: int = 15,
        strict_reduction: bool = False,
        remove_repetitive: bool = True,
        iterations: int = 100,
        reduction_ratio: float = 0.1,
        kiwi: Optional['Kiwi'] = None,
        show_progress: bool = True,
        total_texts: Optional[int] = None,
        callback : Optional[Union[TrainerCallback, List[TrainerCallback]]] = None,
    ) -> 'SwTokenizer':
        import kiwipiepy
        if kiwi is None:
            kiwi = kiwipiepy.Kiwi()
        if not isinstance(kiwi, kiwipiepy.Kiwi):
            raise ValueError("`kiwi` must be an instance of `Kiwi`.")
        
        if not isinstance(config, SwTokenizerConfig):
            raise ValueError("`config` must be an instance of `SwTokenizerConfig`.")

        if callback is None:
            callback = []
        elif isinstance(callback, TrainerCallback):
            callback = [callback]
        else:
            callback = list(callback)
        
        if not all(isinstance(c, TrainerCallback) for c in callback):
            raise ValueError("`callback` must be an instance of `TrainerCallback`.")

        if show_progress:
            callback.insert(0, ProgressShower(None if show_progress is True else show_progress, total_texts, iterations))

        _SwTokenizer._train(
            save_path, texts, config, 
            vocab_size=vocab_size, 
            chr_coverage=chr_coverage, 
            strict_reduction=strict_reduction, 
            remove_repetitive=remove_repetitive, 
            iterations=iterations,
            reduction_ratio=reduction_ratio, 
            prefix_min_cnt=prefix_min_cnt,
            prefix_max_length=prefix_max_length,
            kiwi=kiwi, 
            callback=callback,
        )
        return SwTokenizer(save_path, kiwi)

