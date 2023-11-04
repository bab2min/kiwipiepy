import re
import string
from typing import List, Optional, Tuple, Union

format_pattern = re.compile(r'(\{\{)|(\}\})|(\{[^}]*\})')

def _to_kiwi_tokens(token):
    from kiwipiepy import Token
    if isinstance(token, Token):
        return [token]
    
    if isinstance(token, tuple):
        if len(token) == 2 and isinstance(token[0], str) and isinstance(token[1], str):
            return [token]
        if len(token) == 3 and isinstance(token[0], str) and isinstance(token[1], str) and isinstance(token[2], bool):
            return [token]

    if isinstance(token, list):
        ret = []
        for t in token:
            if isinstance(t, Token):
                ret.append(t)
            elif isinstance(t, tuple):
                if len(t) == 2 and isinstance(t[0], str) and isinstance(t[1], str):
                    ret.append(t)
                elif len(t) == 3 and isinstance(t[0], str) and isinstance(t[1], str) and isinstance(t[2], bool):
                    ret.append(t)
                else:
                    return None
            else:
                return None
        return ret

    return None

class Template:
    def __init__(self,
        kiwi: 'Kiwi',
        format_str: str,
    ):
        from kiwipiepy._wrap import _convert_consonant
        self._kiwi = kiwi
        self._format_str = format_str
        self._formatter = string.Formatter()

        chunks = []
        offset = 0
        pretokenized_lists = []
        self._parsed_format = []
        implicit_field_index = 0
        has_explicit_field_index = False
        for literal, field, format, conversion in self._formatter.parse(format_str):
            literal = _convert_consonant(literal)
            chunks.append(literal)
            offset += len(literal)
            if field is not None:
                chunks.append('{}')
                pretokenized_lists.append((offset, offset + 2, 'SSC'))
                offset += 2
                if field.isdigit():
                    has_explicit_field_index = True
                    if implicit_field_index:
                        raise ValueError('cannot switch from manual field specification to automatic field numbering')
                if field == '':
                    if has_explicit_field_index:
                        raise ValueError('cannot switch from automatic field numbering to manual field specification')
                    field = str(implicit_field_index)
                    implicit_field_index += 1
            self._parsed_format.append(([], field, format, conversion))
        
        tokens = kiwi.tokenize(''.join(chunks), pretokenized=pretokenized_lists)
        placeholder_iter = iter(pretokenized_lists)
        next_placeholder = next(placeholder_iter, None)
        parsed_iter = iter(self._parsed_format)
        target_tokens = next(parsed_iter)[0]
        for token in tokens:
            if next_placeholder and token.span == next_placeholder[:2]:
                target_tokens = next(parsed_iter)[0]
                next_placeholder = next(placeholder_iter, None)
            else:
                target_tokens.append(token)
    
    def format(self,
        *args,
        **kwargs,
    ):
        all_tokens = []
        for tokens, field, format, conversion in self._parsed_format:
            all_tokens += tokens
            if field is None:
                continue

            value, _ = self._formatter.get_field(field, args, kwargs)
            tokens = _to_kiwi_tokens(value)
            if tokens and not conversion:
                if format: 
                    raise ValueError('cannot specify format specifier for Kiwi Token')
                all_tokens += tokens
            else:
                value = self._formatter.convert_field(value, conversion)
                value = self._formatter.format_field(value, format)
                all_tokens.append((value, 'SW'))

        return self._kiwi.join(all_tokens)
