import os

_tag_set = {'NNG', 'NNP', 'NNB', 'NR', 'NP', 'VV', 'VA', 'VX', 'VCP', 'VCN', 'MM', 'MAG', 'MAJ', 'IC', 'JKS',
             'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC', 'EP', 'EF', 'EC', 'ETN', 'ETM', 'XPN', 'XSN',
             'XSV', 'XSA', 'XR', 'SF', 'SP', 'SS', 'SE', 'SO', 'SW', 'SL', 'SH', 'SN', 'UN', 'W_URL', 'W_EMAIL',
             'W_HASHTAG', 'W_MENTION'}

class Stopwords:

    def _load_stopwords(self, filename):
        stopwords = set()
        for stopword in open(filename, 'r', encoding='utf-8'):
            stopword = stopword.strip()
            try:
                form, tag = stopword.split('/')
            except:
                raise ValueError(f"Line in format 'form/tag' expected, but {repr(stopword)} found.")
            stopwords.add((form, tag))
        return stopwords

    def __init__(self, filename=None):
        if filename is None:
            path = os.path.abspath(__file__)
            dir_path = os.path.dirname(path)
            filename = dir_path + '/corpus/stopwords.txt'
        self.stopwords = self._load_stopwords(filename)

    def __contains__(self, word):
        return word in self.stopwords

    def _tag_exists(self, tag):
        if tag in _tag_set:
            return True
        raise ValueError(f'{repr(tag)} is an invalid tag.')

    def _token_exists(self, token):
        if token in self.stopwords:
            return True
        raise ValueError(f"{repr(token)} doesn't exist in stopwords")

    def _is_not_stopword(self, token):
        return (token.form, token.tag) not in self.stopwords

    def add(self, tokens):
        if type(tokens) is str:
            self.stopwords.add((tokens, 'NNP'))
        elif type(tokens) is tuple and self._tag_exists(tokens[1]):
            self.stopwords.add(tokens)
        else:
            for token in tokens:
                if type(token) is str:
                    token = (token, 'NNP')
                    self.stopwords.add(token)
                    continue
                if self._tag_exists(token[1]):
                    self.stopwords.add(token)
        self.stopwords = self.stopwords

    def remove(self, tokens):
        if type(tokens) is str and self._token_exists((tokens, 'NNP')):
            self.stopwords.remove((tokens, 'NNP'))
        elif type(tokens) is tuple and self._token_exists(tokens):
            self.stopwords.remove(tokens)
        else:
            for token in tokens:
                if type(token) is str:
                    token = (token, 'NNP')
                if self._token_exists(token):
                    self.stopwords.remove(token)
        self.stopwords = self.stopwords

    def filter(self, tokens):
        return list(filter(self._is_not_stopword, tokens))
