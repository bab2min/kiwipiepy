from typing import List, Tuple, Optional

import numpy as np
from _kiwipiepy import _KNLangModel

class KNLangModel(_KNLangModel):
    
    def __init__(self):
        raise NotImplementedError("This class is not supposed to be instantiated directly.")

    @classmethod
    def load(cls, path:str, num_workers:int = 0) -> 'KNLangModel':
        return _KNLangModel.load(cls, path, num_workers)

    @classmethod
    def from_arrays(cls, 
                    token_arrays:List[List[int]],
                    ngram_size:int,
                    min_cf:int,
                    bos_token_id:int,
                    eos_token_id:int,
                    unk_token_id:int,
                    num_workers:int = 0,
                    last_min_cf:Optional[int] = None,
                    ) -> 'KNLangModel':
        if last_min_cf is None:
            last_min_cf = min_cf

        return _KNLangModel.from_arrays(cls, 
                                        [np.array(a) for a in token_arrays], 
                                        ngram_size, 
                                        min_cf, 
                                        last_min_cf, 
                                        bos_token_id, 
                                        eos_token_id, 
                                        unk_token_id,
                                        num_workers)

    @property
    def ngram_size(self) -> int:
        return super()._ngram_size

    @property
    def vocab_size(self) -> int:
        return super()._vocab_size
    
    @property
    def num_workers(self) -> int:
        return super()._num_workers

    def __repr__(self) -> str:
        return f'<KNLangModel object at 0x{id(self):x}, .ngram_size={self.ngram_size}, .vocab_size={self.vocab_size}, .num_workers={self.num_workers}>'

    def save(self, path:str) -> None:
        return super().save(path)

    def next_tokens(self, 
                   prev_tokens:List[int],
                   top_n:int = 1,
                   deferred:bool = False,
                   ) -> Tuple[List[List[int]], List[List[float]]]:
        return super().next_tokens(np.array(prev_tokens), top_n, deferred)
