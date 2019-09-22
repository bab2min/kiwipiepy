#pragma once
#include <Python.h>

#define DOC_SIGNATURE_EN(name, signature, en) PyDoc_STRVAR(name, signature "\n--\n\n" en)
#define DOC_VARIABLE_EN(name, en) PyDoc_STRVAR(name, en)
#ifdef DOC_KO
#define DOC_SIGNATURE_EN_KO(name, signature, en, ko) PyDoc_STRVAR(name, signature "\n--\n\n" ko)
#define DOC_VARIABLE_EN_KO(name, en, ko) PyDoc_STRVAR(name, ko)
#else
#define DOC_SIGNATURE_EN_KO(name, signature, en, ko) PyDoc_STRVAR(name, signature "\n--\n\n" en)
#define DOC_VARIABLE_EN_KO(name, en, ko) PyDoc_STRVAR(name, en)
#endif

DOC_SIGNATURE_EN_KO(Kiwi_add_user_word__doc__, 
	"add_user_word(self, word, tag='NNP', score=0)",
	u8R""(add custom word into model)"",
	u8R""()"");

DOC_SIGNATURE_EN_KO(Kiwi_load_user_dictionary__doc__,
	"load_user_dictionary(self, dict_path)",
	u8R""(load custom dictionary file into model)"",
	u8R""()"");

DOC_SIGNATURE_EN_KO(Kiwi_extract_words__doc__,
	"extract_words(self, reader, min_cnt=10, max_word_len=10, min_score=0.25)",
	u8R""(extract words from corpus)"",
	u8R""()"");

DOC_SIGNATURE_EN_KO(Kiwi_extract_filter_words__doc__,
	"extract_filter_words(self, reader, min_cnt=10, max_word_len=10, min_score=0.25, pos_score=-3)",
	u8R""(extract words from corpus and filter the results)"",
	u8R""()"");

DOC_SIGNATURE_EN_KO(Kiwi_extract_add_words__doc__,
	"extract_add_words(self, reader, min_cnt=10, max_word_len=10, min_score=0.25, pos_score=-3)",
	u8R""(extract words from corpus and add them into model)"",
	u8R""()"");

DOC_SIGNATURE_EN_KO(Kiwi_perform__doc__,
	"perform(self, reader, receiver, top_n=1, min_cnt=10, max_word_len=10, min_score=0.25, pos_score=-3)",
	u8R""(extractAddWords + prepare + analyze)"",
	u8R""()"");

DOC_SIGNATURE_EN_KO(Kiwi_set_cutoff_threshold__doc__,
	"set_cutoff_threshold(self, threshold)",
	u8R""()"",
	u8R""()"");

DOC_SIGNATURE_EN_KO(Kiwi_prepare__doc__,
	"prepare(self)",
	u8R""(prepare the model to analyze text)"",
	u8R""()"");

DOC_SIGNATURE_EN_KO(Kiwi_get_option__doc__,
	"get_option(self, option)",
	u8R""(get option value)"",
	u8R""()"");

DOC_SIGNATURE_EN_KO(Kiwi_set_option__doc__,
	"set_option(self, option, value)",
	u8R""(set option value)"",
	u8R""()"");

DOC_SIGNATURE_EN_KO(Kiwi_analyze__doc__,
	"analyze(self, text, top_n=1)\nanalyze(self, reader, receiver, top_n=1)",
	u8R""(analyze text and return top_n results)"",
	u8R""()"");
