import json
from typing import Dict, List, Optional, Sequence, Tuple, Union

from _kiwipiepy import _ByteGrammar, _GrammarMatcher, _LatticeTokens, _LatticeFollower

__all__ = ['ByteGrammar', 'GrammarMatcher', 'LatticeTokens', 'LatticeFollower']

GrammarMatcher = _GrammarMatcher
LatticeTokens = _LatticeTokens
LatticeFollower = _LatticeFollower


def _byte_decoder() -> Dict[str, int]:
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


class ByteGrammar(_ByteGrammar):
    def __init__(self,
        token_bytes: Sequence[Optional[bytes]],
        tag_tokens: Dict[str, Union[int, Sequence[int]]],
        sense_tokens: Sequence[int] = (),
        eos_tokens: Sequence[int] = (),
    ):
        super().__init__(list(token_bytes), dict(tag_tokens), list(sense_tokens), list(eos_tokens))

    @classmethod
    def from_tokenizer_json(cls,
        path: str,
        tag_tokens: Dict[str, Union[int, Sequence[int]]],
        sense_tokens: Sequence[int] = (),
        eos_tokens: Sequence[int] = (),
    ) -> 'ByteGrammar':
        with open(path, encoding='utf-8') as f:
            tok = json.load(f)
        model = tok['model']
        if model.get('type') != 'BPE':
            raise ValueError('Only BPE tokenizers are supported.')
        vocab = model['vocab']
        added = {t['id'] for t in tok.get('added_tokens', ())}
        dec = _byte_decoder()
        size = max(max(vocab.values()), max(added, default=-1)) + 1
        token_bytes = [None] * size
        for s, i in vocab.items():
            if i not in added:
                token_bytes[i] = bytes(dec[c] for c in s)
        return cls(token_bytes, tag_tokens, sense_tokens, eos_tokens)

    def matcher(self, graph: Tuple[int, List[tuple]], senses: Optional[Dict[Tuple[str, str], Sequence[int]]] = None,
                sense_required: bool = False) -> GrammarMatcher:
        num_states, edges = graph[:2]
        return super()._matcher(num_states, edges, senses, sense_required)

    def matcher_packed(self, num_states: int, strings: Sequence[str], packed: bytes,
                       senses: Optional[Dict[Tuple[str, str], Sequence[int]]] = None, sense_required: bool = False) -> GrammarMatcher:
        return super()._matcher_packed(num_states, list(strings), packed, senses, sense_required)

    def accept_tokens(self, matchers: Sequence[Optional[GrammarMatcher]], tokens: Sequence[int]) -> 'numpy.ndarray':
        return super()._accept_tokens(matchers, tokens).view(bool)

    def allowed_pairs(self, matchers: Sequence[Optional[GrammarMatcher]], rows: Optional[Sequence[int]] = None) -> Tuple['numpy.ndarray', 'numpy.ndarray']:
        return super()._allowed_pairs(matchers, rows)
