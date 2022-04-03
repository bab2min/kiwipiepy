#include <stdexcept>

#ifdef _DEBUG
#undef _DEBUG
#include "PyUtils.h"
#include "PyDoc.h"
#define _DEBUG
#else 
#include "PyUtils.h"
#include "PyDoc.h"
#endif

#include <kiwi/Kiwi.h>

using namespace std;
using namespace kiwi;

static PyObject* gModule;

struct KiwiObject : py::CObject<KiwiObject>
{
	static constexpr const char* _name = "kiwipiepy._Kiwi";
	static constexpr const char* _name_in_module = "_Kiwi";
	static constexpr int _flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;

	KiwiBuilder builder;
	Kiwi kiwi;

	static int init(KiwiObject *self, PyObject *args, PyObject *kwargs)
	{
		return py::handleExc([&]()
		{
			const char* modelPath = nullptr;
			size_t numThreads = 0, options = 3;
			int integrateAllomorph = -1, loadDefaultDict = -1;
			static const char* kwlist[] = { "num_workers", "model_path", "integrate_allomorph", "load_default_dict", nullptr };
			if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|nzpp", (char**)kwlist,
				&numThreads, &modelPath, &integrateAllomorph, &loadDefaultDict
			)) return -1;

			BuildOption boptions = BuildOption::integrateAllomorph | BuildOption::loadDefaultDict;

			if (integrateAllomorph >= 0)
			{
				boptions = (boptions & ~BuildOption::integrateAllomorph) 
					| (integrateAllomorph ? BuildOption::integrateAllomorph : BuildOption::none);
			}

			if (loadDefaultDict >= 0)
			{
				boptions = (boptions & ~BuildOption::loadDefaultDict)
					| (loadDefaultDict ? BuildOption::loadDefaultDict : BuildOption::none);
			}

			string spath;
			if (modelPath)
			{
				spath = modelPath;
			}
			else
			{
				py::UniqueObj modelModule{ PyImport_ImportModule("kiwipiepy_model") };
				if (!modelModule) throw py::ExcPropagation{};
				py::UniqueObj pathFunc{ PyObject_GetAttrString(modelModule, "get_model_path")};
				if (!pathFunc) throw py::ExcPropagation{};
				py::UniqueObj pathRet{ PyObject_CallObject(pathFunc, nullptr) };
				if (!pathRet) throw py::ExcPropagation{};
				spath = py::toCpp<string>(pathRet);
			}

			self->builder = KiwiBuilder{ spath, numThreads, (BuildOption)boptions };
			return 0;
		});
	}

	void doPrepare()
	{
		if (kiwi.ready()) return;
		kiwi = builder.build();
		py::UniqueObj handler{ PyObject_GetAttrString((PyObject*)this, "_on_build") };
		if (handler)
		{
			py::UniqueObj res{ PyObject_CallFunctionObjArgs(handler, nullptr) };
			if (!res) throw py::ExcPropagation{};
		}
		else
		{
			PyErr_Clear();
		}
	}

	PyObject* addUserWord(PyObject* args, PyObject* kwargs);
	PyObject* addPreAnalyzedWord(PyObject* args, PyObject* kwargs);
	PyObject* addRule(PyObject* args, PyObject* kwargs);
	PyObject* analyze(PyObject* args, PyObject* kwargs);
	PyObject* extractAddWords(PyObject* args, PyObject* kwargs);
	PyObject* extractWords(PyObject* args, PyObject* kwargs);
	PyObject* loadUserDictionary(PyObject* args, PyObject* kwargs);
	PyObject* perform(PyObject* args, PyObject* kwargs);
	PyObject* getMorpheme(PyObject* args, PyObject* kwargs);

	float getCutOffThreshold() const
	{
		return kiwi.getCutOffThreshold();
	}

	void setCutOffThreshold(float v)
	{
		kiwi.setCutOffThreshold(v);
	}

	size_t getMaxUnkFormSize() const
	{
		return kiwi.getMaxUnkFormSize();
	}

	void setMaxUnkFormSize(size_t v)
	{
		kiwi.setMaxUnkFormSize(v);
	}

	float getUnkScoreBias() const
	{
		return kiwi.getUnkScoreBias();
	}

	void setUnkScoreBias(float v)
	{
		kiwi.setUnkScoreBias(v);
	}

	float getUnkScoreScale() const
	{
		return kiwi.getUnkScoreScale();
	}

	void setUnkScoreScale(float v)
	{
		kiwi.setUnkScoreScale(v);
	}

	bool getIntegrateAllomorph() const
	{
		return kiwi.getIntegrateAllomorph();
	}

	void setIntegrateAllomorph(bool v)
	{
		kiwi.setIntegrateAllomorph(v);
	}

	size_t getSpaceTolerance() const
	{
		return kiwi.getSpaceTolerance();
	}

	void setSpaceTolerance(size_t v)
	{
		kiwi.setSpaceTolerance(v);
	}

	float getSpacePenalty() const
	{
		return kiwi.getSpacePenalty();
	}

	void setSpacePenalty(float v)
	{
		kiwi.setSpacePenalty(v);
	}

	size_t getNumWorkers() const
	{
		return kiwi.getNumThreads();
	}
};

py::TypeWrapper<KiwiObject> _KiwiSetter{ [](PyTypeObject& obj)
{
	static PyMethodDef methods[] =
	{
		{ "add_user_word", PY_METHOD_MEMFN(&KiwiObject::addUserWord), METH_VARARGS | METH_KEYWORDS, ""},
		{ "add_pre_analyzed_word", PY_METHOD_MEMFN(&KiwiObject::addPreAnalyzedWord), METH_VARARGS | METH_KEYWORDS, ""},
		{ "add_rule", PY_METHOD_MEMFN(&KiwiObject::addRule), METH_VARARGS | METH_KEYWORDS, ""},
		{ "load_user_dictionary", PY_METHOD_MEMFN(&KiwiObject::loadUserDictionary), METH_VARARGS | METH_KEYWORDS, "" },
		{ "extract_words", PY_METHOD_MEMFN(&KiwiObject::extractWords), METH_VARARGS | METH_KEYWORDS, "" },
		{ "extract_add_words", PY_METHOD_MEMFN(&KiwiObject::extractAddWords), METH_VARARGS | METH_KEYWORDS, "" },
		{ "perform", PY_METHOD_MEMFN(&KiwiObject::perform), METH_VARARGS | METH_KEYWORDS, "" },
		{ "analyze", PY_METHOD_MEMFN(&KiwiObject::analyze), METH_VARARGS | METH_KEYWORDS, "" },
		{ "morpheme", PY_METHOD_MEMFN(&KiwiObject::getMorpheme), METH_VARARGS | METH_KEYWORDS, "" },
		{ nullptr }
	};
	static PyGetSetDef getsets[] =
	{
		{ (char*)"_cutoff_threshold", PY_GETTER_MEMFN(&KiwiObject::getCutOffThreshold), PY_SETTER_MEMFN(&KiwiObject::setCutOffThreshold), "", nullptr },
		{ (char*)"_integrate_allomorph", PY_GETTER_MEMFN(&KiwiObject::getIntegrateAllomorph), PY_SETTER_MEMFN(&KiwiObject::setIntegrateAllomorph), "", nullptr },
		{ (char*)"_unk_score_bias", PY_GETTER_MEMFN(&KiwiObject::getUnkScoreBias), PY_SETTER_MEMFN(&KiwiObject::setUnkScoreBias), "", nullptr },
		{ (char*)"_unk_score_scale", PY_GETTER_MEMFN(&KiwiObject::getUnkScoreScale), PY_SETTER_MEMFN(&KiwiObject::setUnkScoreScale), "", nullptr },
		{ (char*)"_max_unk_form_size", PY_GETTER_MEMFN(&KiwiObject::getMaxUnkFormSize), PY_SETTER_MEMFN(&KiwiObject::setMaxUnkFormSize), "", nullptr },
		{ (char*)"_space_tolerance", PY_GETTER_MEMFN(&KiwiObject::getSpaceTolerance), PY_SETTER_MEMFN(&KiwiObject::setSpaceTolerance), "", nullptr },
		{ (char*)"_space_penalty", PY_GETTER_MEMFN(&KiwiObject::getSpacePenalty), PY_SETTER_MEMFN(&KiwiObject::setSpacePenalty), "", nullptr },
		{ (char*)"_num_workers", PY_GETTER_MEMFN(&KiwiObject::getNumWorkers), nullptr, "", nullptr },
		{ nullptr },
	};
	obj.tp_methods = methods;
	obj.tp_getset = getsets;
}};

struct TokenObject : py::CObject<TokenObject>
{
	static constexpr const char* _name = "kiwipiepy.Token";
	static constexpr const char* _name_in_module = "Token";
	static constexpr const char* _doc = Token__doc__;

	u16string _form;
	const char* _tag = nullptr;
	uint32_t _pos = 0, _len = 0, _wordPosition = 0, _sentPosition = 0, _lineNumber = 0;
	size_t _morphId = 0;
	const Morpheme* _morph = nullptr;
	const Morpheme* _baseMorph = nullptr;

	static int init(TokenObject* self, PyObject* args, PyObject* kwargs)
	{
		return 0;
	}

	uint32_t end()
	{
		return _pos + _len;
	}

	u16string taggedForm()
	{
		u16string ret = _form;
		ret.push_back(u'/');
		ret.insert(ret.end(), _tag, _tag + strlen(_tag));
	 	return ret;
	}

	u16string baseForm()
	{
	 	return kiwi::joinHangul(_baseMorph->getForm());
	}

	size_t baseId()
	{
		return (_baseMorph - _morph) + _morphId;
	}

	static Py_ssize_t len(TokenObject* self)
	{
		return 4;
	}

	static PyObject* getitem(TokenObject* self, Py_ssize_t idx)
	{
		return py::handleExc([&]()
		{
			if (idx < 0) idx += len(self);
			switch (idx)
			{
			case 0: return py::buildPyValue(self->_form);
			case 1: return py::buildPyValue(self->_tag);
			case 2: return py::buildPyValue(self->_pos);
			case 3: return py::buildPyValue(self->_len);
			}
			throw py::IndexError{ "index out of range" };
		});
	}

	static PyObject* repr(TokenObject* self)
	{
		return py::handleExc([&]()
		{
			return py::buildPyValue("Token("
				"form=" + py::reprFromCpp(self->_form) + ", "
				"tag=" + py::reprFromCpp(self->_tag) + ", "
				"start=" + to_string(self->_pos) + ", "
				"len=" + to_string(self->_len) +
			")");
		});
	}
};

py::TypeWrapper<TokenObject> _TokenSetter{ [](PyTypeObject& obj)
{
	static PyGetSetDef getsets[] =
	{
		{ (char*)"form", PY_GETTER_MEMPTR(&TokenObject::_form), nullptr, Token_form__doc__, nullptr },
		{ (char*)"tag", PY_GETTER_MEMPTR(&TokenObject::_tag), nullptr, Token_tag__doc__, nullptr },
		{ (char*)"start", PY_GETTER_MEMPTR(&TokenObject::_pos), nullptr, Token_start__doc__, nullptr },
		{ (char*)"len", PY_GETTER_MEMPTR(&TokenObject::_len), nullptr, Token_len__doc__, nullptr },
		{ (char*)"end", PY_GETTER_MEMFN(&TokenObject::end), nullptr, Token_end__doc__, nullptr },
		{ (char*)"id", PY_GETTER_MEMPTR(&TokenObject::_morphId), nullptr, Token_id__doc__, nullptr },
		{ (char*)"word_position", PY_GETTER_MEMPTR(&TokenObject::_wordPosition), nullptr, Token_word_position__doc__, nullptr },
		{ (char*)"sent_position", PY_GETTER_MEMPTR(&TokenObject::_sentPosition), nullptr, Token_sent_position__doc__, nullptr },
		{ (char*)"line_number", PY_GETTER_MEMPTR(&TokenObject::_lineNumber), nullptr, Token_line_number__doc__, nullptr },
		{ (char*)"base_form", PY_GETTER_MEMFN(&TokenObject::baseForm), nullptr, Token_base_form__doc__, nullptr },
		{ (char*)"base_id", PY_GETTER_MEMFN(&TokenObject::baseId), nullptr, Token_base_id__doc__, nullptr },
		{ (char*)"tagged_form", PY_GETTER_MEMFN(&TokenObject::taggedForm), nullptr, Token_tagged_form__doc__, nullptr },
		{ nullptr },
	};

	static PySequenceMethods seq = {
		(lenfunc)TokenObject::len,
		nullptr,
		nullptr,
		(ssizeargfunc)TokenObject::getitem,
	};

	obj.tp_getset = getsets;
	obj.tp_as_sequence = &seq;
} };

PyObject* resToPyList(vector<TokenResult>&& res, const Kiwi& kiwi)
{
	py::UniqueObj retList{ PyList_New(res.size()) };
	size_t idx = 0;
	for (auto& p : res)
	{
		py::UniqueObj rList{ PyList_New(p.first.size()) };
		size_t jdx = 0;
		size_t u32offset = 0;
		for (auto& q : p.first)
		{
			size_t u32chrs = 0;
			for (auto u : q.str)
			{
				if ((u & 0xFC00) == 0xD800) u32chrs++;
			}

			py::UniqueObj item{ PyObject_CallFunctionObjArgs((PyObject*)py::Type<TokenObject>, nullptr) };
			auto* tItem = (TokenObject*)item.get();
			tItem->_form = move(q.str);
			tItem->_tag = tagToString(q.tag);
			tItem->_pos = q.position - u32offset;
			tItem->_len = q.length - u32chrs;
			tItem->_wordPosition = q.wordPosition;
			tItem->_sentPosition = q.sentPosition;
			tItem->_lineNumber = q.lineNumber;
			tItem->_morph = q.morph;
			tItem->_morphId = kiwi.morphToId(q.morph);
			tItem->_baseMorph = kiwi.idToMorph(q.morph->lmMorphemeId);

			PyList_SetItem(rList, jdx++, item.release());
			u32offset += u32chrs;
		}
		PyList_SetItem(retList, idx++, py::buildPyTuple(move(rList), p.second));
	}
	return retList.release();
}

struct KiwiResIter : public py::ResultIter<KiwiResIter, vector<TokenResult>>
{
	static constexpr const char* _name = "kiwipiepy._ResIter";
	static constexpr const char* _name_in_module = "_ResIter";

	py::UniqueCObj<KiwiObject> kiwi;
	size_t topN = 1;
	Match matchOptions = Match::all;

	PyObject* buildPy(vector<TokenResult>&& v)
	{
		return py::handleExc([&]() -> PyObject*
		{
			if (v.size() > topN) v.erase(v.begin() + topN, v.end());
			return resToPyList(move(v), kiwi->kiwi);
		});
	}

	future<vector<TokenResult>> feedNext(py::SharedObj&& next)
	{
		if (!PyUnicode_Check(next)) throw py::ValueError{ "`analyze` requires an instance of `str` or an iterable of `str`." };
		return kiwi->kiwi.asyncAnalyze(PyUnicode_AsUTF8(next), topN, matchOptions);
	}
};

py::TypeWrapper<KiwiResIter> _ResIterSetter{ [](PyTypeObject&)
{
} };

inline POSTag parseTag(const char* tag)
{
	auto u16 = utf8To16(tag);
	transform(u16.begin(), u16.end(), u16.begin(), static_cast<int(*)(int)>(toupper));
	auto pos = toPOSTag(u16);
	if (pos >= POSTag::max) throw py::ValueError{ "Unknown tag value " + py::reprFromCpp(tag) };
	return pos;
}

inline POSTag parseTag(const u16string& tag)
{
	auto u16 = tag;
	transform(u16.begin(), u16.end(), u16.begin(), static_cast<int(*)(int)>(toupper));
	auto pos = toPOSTag(u16);
	if (pos >= POSTag::max) throw py::ValueError{ "Unknown tag value " + py::reprFromCpp(tag) };
	return pos;
}

PyObject* KiwiObject::addUserWord(PyObject* args, PyObject *kwargs)
{	
	return py::handleExc([&]() -> PyObject*
	{
		const char* word;
		const char* tag = "NNP";
		float score = 0;
		const char* origWord = nullptr;
		static const char* kwlist[] = { "word", "tag", "score", "orig_word", nullptr};
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|sfz", (char**)kwlist, &word, &tag, &score, &origWord)) return nullptr;

		auto pos = parseTag(tag);
		bool added = false;
		if (origWord)
		{
			added = builder.addWord(utf8To16(word), pos, score, utf8To16(origWord));
		}
		else
		{
			added = builder.addWord(utf8To16(word), pos, score);
		}
		if (added) kiwi = Kiwi{};
		return py::buildPyValue(added);
	});
}

PyObject* KiwiObject::addPreAnalyzedWord(PyObject* args, PyObject* kwargs)
{
	return py::handleExc([&]() -> PyObject*
	{
		const char* form;
		PyObject* oAnalyzed = nullptr;
		float score = 0;
		static const char* kwlist[] = { "form", "analyzed", "score", nullptr };
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sO|f", (char**)kwlist, &form, &oAnalyzed, &score)) return nullptr;

		vector<pair<u16string, POSTag>> analyzed;
		vector<pair<size_t, size_t>> positions;
		py::foreach<PyObject*>(oAnalyzed, [&](PyObject* item)
		{
			if (PyUnicode_Check(item))
			{
				auto str = py::toCpp<u16string>(item);
				auto p = str.rfind('/');
				if (p == str.npos)
				{
					throw py::ValueError{ "`analyzed` must be in format `{form}/{tag}`, but given : " + py::repr(item)};
				}
				analyzed.emplace_back(str.substr(0, p), parseTag(str.substr(p + 1)));
			}
			else if (PySequence_Check(item))
			{
				if (Py_SIZE(item) == 2)
				{
					auto p = py::toCpp<pair<u16string, const char*>>(item);
					analyzed.emplace_back(p.first, parseTag(p.second));
				}
				else
				{
					auto t = py::toCpp<tuple<u16string, const char*, size_t, size_t>>(item);
					analyzed.emplace_back(get<0>(t), parseTag(get<1>(t)));
					positions.emplace_back(get<2>(t), get<3>(t));
				}
			}
			else
			{
				throw py::ConversionFail{ "`analyzed` must be an iterable of `Tuple[str, str]` or `Tuple[str, str, int, int]`." }; 
			}
		}, "`analyzed` must be an iterable of `Tuple[str, str]` or `Tuple[str, str, int, int]`.");
		if (!positions.empty() && positions.size() != analyzed.size())
		{
			throw py::ValueError{ "All items of `analyzed` must be in the type `Tuple[str, str]` or `Tuple[str, str, int, int]`."};
		}
		try 
		{
			auto added = builder.addPreAnalyzedWord(utf8To16(form), analyzed, positions, score);
			if (added) kiwi = Kiwi{};
			return py::buildPyValue(added);
		}
		catch (const UnknownMorphemeException& e)
		{
			throw py::ValueError{ e.what() };
		}
	});
}

PyObject* KiwiObject::addRule(PyObject* args, PyObject* kwargs)
{
	return py::handleExc([&]() -> PyObject*
	{
		const char* tag = nullptr;
		PyObject* replacer = nullptr;
		float score = 0;
		static const char* kwlist[] = { "tag", "replacer", "score", nullptr };
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sO|f", (char**)kwlist, &tag, &replacer, &score)) return nullptr;

		if (!PyCallable_Check(replacer)) throw py::ValueError{ "`replacer` must be an callable." };

		auto pos = parseTag(tag);
		auto added = builder.addRule(pos, [&](const u16string& input)
		{
			py::UniqueObj ret{ PyObject_CallFunctionObjArgs(replacer, py::UniqueObj{ py::buildPyValue(input) }.get(), nullptr) };
			if (!ret) throw py::ExcPropagation{};
			return py::toCpp<u16string>(ret);
		}, score);
		if (!added.empty()) kiwi = Kiwi{};
		return py::buildPyValue(added);
	});
}

PyObject* KiwiObject::loadUserDictionary(PyObject* args, PyObject *kwargs)
{
	return py::handleExc([&]() -> PyObject*
	{
		const char* path;
		static const char* kwlist[] = { "dict_path", nullptr };
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s", (char**)kwlist, &path)) return nullptr;

		auto ret = builder.loadDictionary(path);
		if (ret) kiwi = Kiwi{};
		return py::buildPyValue(ret);
	});
}

U16MultipleReader obj2reader(PyObject* obj)
{
	return [obj]()
	{
		py::SharedObj iter{ PyObject_GetIter(obj) };

		if (!iter) throw py::ExcPropagation{};
		return [iter]() -> u16string
		{
			py::UniqueObj item{ PyIter_Next(iter) };
			if (!item)
			{
				if (PyErr_Occurred()) throw py::ExcPropagation{};
				return {};
			}
			auto ret = py::toCpp<u16string>(item);
			if (ret.empty()) ret.push_back(' ');
			return ret;
		};
	};
}

PyObject* KiwiObject::extractWords(PyObject* args, PyObject *kwargs)
{
	return py::handleExc([&]() -> PyObject*
	{
		PyObject* sentences;
		size_t minCnt = 10, maxWordLen = 10;
		float minScore = 0.25f, posScore = -3;
		size_t lmFilter = 1;
		static const char* kwlist[] = { "texts", "min_cnt", "max_word_len", "min_score", "pos_score", "lm_filter", nullptr };
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|nnffp", (char**)kwlist, &sentences, &minCnt, &maxWordLen, &minScore, &posScore, &lmFilter)) return nullptr;

		auto res = builder.extractWords(obj2reader(sentences), minCnt, maxWordLen, minScore, posScore, lmFilter);

		py::UniqueObj retList{ PyList_New(res.size()) };
		size_t idx = 0;
		for (auto& r : res)
		{
			auto v = py::buildPyTuple(utf16To8(r.form).c_str(), r.score, r.freq, r.posScore[POSTag::nnp]);
			if (!v) throw py::ExcPropagation{};
			PyList_SetItem(retList, idx++, v);
		}
		return retList.release();
	});
}

PyObject* KiwiObject::extractAddWords(PyObject* args, PyObject *kwargs)
{
	return py::handleExc([&]() -> PyObject*
	{
		PyObject* sentences;
		size_t minCnt = 10, maxWordLen = 10;
		float minScore = 0.25f, posScore = -3;
		size_t lmFilter = 1;
		static const char* kwlist[] = { "texts", "min_cnt", "max_word_len", "min_score", "pos_score", "lm_filter", nullptr };
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|nnffp", (char**)kwlist, &sentences, &minCnt, &maxWordLen, &minScore, &posScore, &lmFilter)) return nullptr;

		auto res = builder.extractAddWords(obj2reader(sentences), minCnt, maxWordLen, minScore, posScore, lmFilter);
		kiwi = Kiwi{};

		py::UniqueObj retList{ PyList_New(res.size()) };
		size_t idx = 0;
		for (auto& r : res)
		{
			auto v = py::buildPyTuple(utf16To8(r.form).c_str(), r.score, r.freq, r.posScore[POSTag::nnp]);
			if (!v) throw py::ExcPropagation{};
			PyList_SetItem(retList, idx++, v);
		}
		return retList.release();
	});
}

PyObject* KiwiObject::analyze(PyObject* args, PyObject *kwargs)
{
	return py::handleExc([&]() -> PyObject*
	{
		size_t topN = 1, matchOptions = (size_t)Match::all, echo = 0;
		PyObject* text;
		static const char* kwlist[] = { "text", "top_n", "match_options", "echo", nullptr};
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|nnp", (char**)kwlist, &text, &topN, &matchOptions, &echo)) return nullptr;

		doPrepare();
		if (PyUnicode_Check(text))
		{
			auto res = kiwi.analyze(PyUnicode_AsUTF8(text), max(topN, (size_t)10), (Match)matchOptions);
			if (res.size() > topN) res.erase(res.begin() + topN, res.end());
			return resToPyList(move(res), kiwi);
		}
		else
		{
			py::UniqueObj iter{ PyObject_GetIter(text) };
			if (!iter) throw py::ValueError{ "`analyze` requires a `str` or an iterable of `str` parameters." };
			py::UniqueCObj<KiwiResIter> ret{ (KiwiResIter*)PyObject_CallObject((PyObject*)py::Type<KiwiResIter>, nullptr) };
			if (!ret) throw py::ExcPropagation{};
			ret->kiwi = py::UniqueCObj<KiwiObject>{ this };
			Py_INCREF(this);
			ret->inputIter = move(iter);
			ret->topN = topN;
			ret->matchOptions = (Match)matchOptions;
			ret->echo = !!echo;
			for (int i = 0; i < kiwi.getNumThreads() * 16; ++i)
			{
				if (!ret->feed()) break;
			}
			return (PyObject*)ret.release();
		}
	});
}

PyObject* KiwiObject::perform(PyObject* args, PyObject *kwargs)
{
	return py::handleExc([&]() -> PyObject*
	{
		size_t topN = 1, matchOptions = (size_t)Match::all;
		PyObject* sentences;
		size_t minCnt = 10, maxWordLen = 10;
		float minScore = 0.25f, posScore = -3;
		size_t lmFilter = 1;
		static const char* kwlist[] = { "texts", "top_n", "match_options", "min_cnt", "max_word_len", "min_score", "pos_score", "lm_filter", nullptr };
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|nnnnffp", (char**)kwlist,
			&sentences, &topN, &matchOptions, &minCnt, &maxWordLen, &minScore, &posScore, &lmFilter)) return nullptr;

		auto tBuilder = builder;
		auto reader = obj2reader(sentences);
		tBuilder.extractAddWords(reader, minCnt, maxWordLen, minScore, posScore, lmFilter);
		auto tKiwi = tBuilder.build();
		py::UniqueObj ret{ PyList_New(0) };
		tKiwi.analyze(topN, reader(), [&](vector<TokenResult>&& res)
		{
			PyList_Append(ret, py::UniqueObj{ resToPyList(move(res), kiwi) });
		}, (Match)matchOptions);
		return ret.release();
	});
}

PyObject* KiwiObject::getMorpheme(PyObject* args, PyObject* kwargs)
{
	size_t id = 0;
	static const char* kwlist[] = { "id", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "n", (char**)kwlist, &id)) return nullptr;
	return py::handleExc([&]()
	{
		py::UniqueObj ret{ PyObject_CallFunctionObjArgs((PyObject*)py::Type<TokenObject>, nullptr) };
		auto* obj = (TokenObject*)ret.get();
		auto* morph = kiwi.idToMorph(id);
		if (!morph) throw py::ValueError{ "out of range" };
		auto& form = morph->getForm();
		obj->_form = u16string{ form.begin(), form.end() };
		obj->_tag = tagToString(morph->tag);
		obj->_morph = morph;
		obj->_morphId = id;
		return ret.release();
	});
}

PyObject* moduleInit()
{
	static PyModuleDef mod =
	{
		PyModuleDef_HEAD_INIT,
		"_kiwipiepy",
		"Kiwi API for Python",
		-1,
		nullptr
	};

	gModule = PyModule_Create(&mod);
	py::TypeManager::addToModule(gModule);
	return gModule;
}

PyMODINIT_FUNC PyInit__kiwipiepy()
{
	return moduleInit();
}
