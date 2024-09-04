from typing import List, Tuple, Optional, Union

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
                    min_cf:Union[int, List[int]],
                    bos_token_id:int,
                    eos_token_id:int,
                    unk_token_id:int,
                    num_workers:int = 0,
                    token_clusters:Optional[List[List[int]]] = None,
                    ) -> 'KNLangModel':
        if isinstance(min_cf, int):
            min_cf = [min_cf] * ngram_size

        return _KNLangModel.from_arrays(cls, 
                                        [np.array(a) for a in token_arrays], 
                                        ngram_size, 
                                        min_cf, 
                                        bos_token_id, 
                                        eos_token_id, 
                                        unk_token_id,
                                        token_clusters or [],
                                        num_workers)

    @property
    def ngram_size(self) -> int:
        return super()._ngram_size

    @property
    def vocab_size(self) -> int:
        return super()._vocab_size
    
    @property
    def num_nodes(self) -> int:
        return super()._num_nodes

    @property
    def num_workers(self) -> int:
        return super()._num_workers

    def __repr__(self) -> str:
        return (f'<KNLangModel object at 0x{id(self):x},'
                f' .ngram_size={self.ngram_size},'
                f' .vocab_size={self.vocab_size},'
                f' .num_nodes={self.num_nodes},'
                f' .num_workers={self.num_workers}>')

    def save(self, path:str) -> None:
        return super().save(path)

    def next_tokens(self, 
                   token_ids:List[int],
                   top_n:int = 1,
                   deferred:bool = False,
                   ) -> Tuple[List[List[int]], List[List[float]]]:
        return super().next_tokens(np.array(token_ids), top_n, deferred)

    def evaluate(self,
                 token_ids:List[int],
                 deferred:bool = False,
                 ) -> List[float]:
        return super().evaluate(np.array(token_ids), deferred)
