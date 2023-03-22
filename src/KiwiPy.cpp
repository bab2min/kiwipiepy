#include <stdexcept>
#define USE_NUMPY
#define MAIN_MODULE

#include "PyUtils.h"
#include "PyDoc.h"

#include <kiwi/Kiwi.h>
#include <kiwi/HSDataset.h>

using namespace std;
using namespace kiwi;

static PyObject* gModule;

struct TypoTransformerObject : py::CObject<TypoTransformerObject>
{
	static constexpr const char* _name = "kiwipiepy._TypoTransformer";
	static constexpr const char* _name_in_module = "_TypoTransformer";
	static constexpr int _flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;

	TypoTransformer tt;
	PreparedTypoTransformer ptt;

	static int init(TypoTransformerObject* self, PyObject* args, PyObject* kwargs)
	{
		return py::handleExc([&]()
		{
			PyObject* defs;
			static const char* kwlist[] = { "defs", nullptr };
			if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", (char**)kwlist,
				&defs
			)) return -1;
			py::foreach<PyObject*>(defs, [&](PyObject* item)
			{
				auto orig = py::toCpp<std::vector<std::string>>(PyTuple_GET_ITEM(item, 0));
				auto error = py::toCpp<std::vector<std::string>>(PyTuple_GET_ITEM(item, 1));
				auto cost = py::toCpp<float>(PyTuple_GET_ITEM(item, 2));
				PyObject* cond = PyTuple_GET_ITEM(item, 3);
				CondVowel condVowel = CondVowel::none;
				if (cond == Py_None)
				{
				}
				else
				{
					auto conds = py::toCpp<std::string>(cond);
					if (conds == "any") condVowel = CondVowel::any;
					else if (conds == "vowel") condVowel = CondVowel::vowel;
					else if (conds == "applosive") condVowel = CondVowel::applosive;
				}
				
				for (auto& o : orig)
				{
					for (auto& e : error)
					{
						self->tt.addTypo(utf8To16(o), utf8To16(e), cost, condVowel);
					}
				}
			}, "`defs` must be an iterable of Tuple[List, List, float, str].");
			return 0;
		});
	}

	PyObject* generate(PyObject* args, PyObject* kwargs)
	{
		return py::handleExc([&]() -> PyObject*
		{
			const char* orig;
			float costThreshold = 2.5f;
			static const char* kwlist[] = { "orig", "cost_threshold", nullptr};
			if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|f", (char**)kwlist,
				&orig, &costThreshold
			)) return nullptr;

			if (!ptt.ready()) ptt = tt.prepare();

			py::UniqueObj ret{ PyList_New(0) };
			for (auto r : ptt.generate(utf8To16(orig), costThreshold))
			{
				PyList_Append(ret.get(), py::buildPyTuple(r.str, r.cost).get());
			}
			return ret.release();
		});
	}
};

py::TypeWrapper<TypoTransformerObject> _TypoTransformerSetter{ [](PyTypeObject& obj)
{
	static PyMethodDef methods[] =
	{
		{ "generate", PY_METHOD_MEMFN(&TypoTransformerObject::generate), METH_VARARGS | METH_KEYWORDS, ""},
		{ nullptr }
	};
	obj.tp_methods = methods;
} };

struct HSDatasetIterObject;

struct HSDatasetObject : py::CObject<HSDatasetObject>
{
	static constexpr const char* _name = "kiwipiepy._HSDataset";
	static constexpr const char* _name_in_module = "_HSDataset";
	static constexpr int _flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;

	HSDataset hsd;

	static int init(HSDatasetObject* self, PyObject* args, PyObject* kwargs)
	{
		return py::handleExc([&]()
		{
			return 0;
		});
	}

	static HSDatasetIterObject* iter(HSDatasetObject* self)
	{
		py::UniqueCObj<HSDatasetIterObject> ret{ (HSDatasetIterObject*)PyObject_CallFunctionObjArgs((PyObject*)py::Type<HSDatasetIterObject>, self, nullptr) };
		return ret.release();
	}

	size_t getVocabSize() const
	{
		return hsd.vocabSize();
	}

	size_t getNgramNodeSize() const
	{
		return hsd.ngramNodeSize();
	}

	size_t getBatchSize() const
	{
		return hsd.getBatchSize();
	}

	size_t getWindowSize() const
	{
		return hsd.getWindowSize();
	}

	size_t numSents() const
	{
		return hsd.numSents();
	}

	static Py_ssize_t len(HSDatasetObject* self)
	{
		return self->hsd.numEstimBatches();
	}

	PyObject* estimVocabFrequency()
	{
		return py::handleExc([&]() -> PyObject*
		{
			return py::buildPyValue(hsd.estimVocabFrequency()).release();
		});
	}

	PyObject* getVocabInfo(PyObject* args, PyObject* kwargs)
	{
		return py::handleExc([&]() -> PyObject*
		{
			size_t index;
			static const char* kwlist[] = { "index", nullptr };
			if (!PyArg_ParseTupleAndKeywords(args, kwargs, "n", (char**)kwlist,
				&index
			)) return nullptr;

			if (index >= hsd.vocabSize()) throw py::ValueError{ to_string(index) };
			return py::buildPyTuple(hsd.vocabForm(index), tagToString(hsd.vocabInfo(index).tag)).release();
		});
	}

	PyObject* getSent(PyObject* args, PyObject* kwargs)
	{
		return py::handleExc([&]() -> PyObject*
		{
			size_t index, augment = 0;
			static const char* kwlist[] = { "index", "augment", nullptr};
			if (!PyArg_ParseTupleAndKeywords(args, kwargs, "n|p", (char**)kwlist,
				&index, &augment
			)) return nullptr;

			if (index >= hsd.numSents()) throw py::ValueError{ to_string(index) };
			if (augment)
			{
				auto sent = hsd.getAugmentedSent(index);
				return py::buildPyValueTransform(sent.begin(), sent.end(), [](size_t v) { return (uint32_t)v; }).release();
			}
			else
			{
				auto sent = hsd.getSent(index);
				return py::buildPyValueTransform(sent.begin(), sent.end(), [](size_t v) { return (uint32_t)v; }).release();
			}
		});
	}
};

py::TypeWrapper<HSDatasetObject> _HSDatasetSetter{ [](PyTypeObject& obj)
{
	static PyMethodDef methods[] =
	{
		{ "get_vocab_info", PY_METHOD_MEMFN(&HSDatasetObject::getVocabInfo), METH_VARARGS | METH_KEYWORDS, ""},
		{ "get_sent", PY_METHOD_MEMFN(&HSDatasetObject::getSent), METH_VARARGS | METH_KEYWORDS, ""},
		{ "estim_vocab_frequency", PY_METHOD_MEMFN(&HSDatasetObject::estimVocabFrequency), METH_NOARGS, ""},
		{ nullptr }
	};
	static PyGetSetDef getsets[] =
	{
		{ (char*)"vocab_size", PY_GETTER_MEMFN(&HSDatasetObject::getVocabSize), nullptr, "", nullptr },
		{ (char*)"ngram_node_size", PY_GETTER_MEMFN(&HSDatasetObject::getNgramNodeSize), nullptr, "", nullptr },
		{ (char*)"batch_size", PY_GETTER_MEMFN(&HSDatasetObject::getBatchSize), nullptr, "", nullptr },
		{ (char*)"window_size", PY_GETTER_MEMFN(&HSDatasetObject::getWindowSize), nullptr, "", nullptr },
		{ (char*)"num_sents", PY_GETTER_MEMFN(&HSDatasetObject::numSents), nullptr, "", nullptr },
		{ nullptr },
	};
	static PySequenceMethods seq = {
		(lenfunc)HSDatasetObject::len,
		nullptr,
		nullptr,
		nullptr,
	};

	obj.tp_methods = methods;
	obj.tp_getset = getsets;
	obj.tp_as_sequence = &seq;
} };

struct HSDatasetIterObject : py::CObject<HSDatasetIterObject>
{
	static constexpr const char* _name = "kiwipiepy._HSDatasetIter";
	static constexpr const char* _name_in_module = "_HSDatasetIter";
	static constexpr int _flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;

	py::UniqueCObj<HSDatasetObject> obj;

	static int init(HSDatasetIterObject* self, PyObject* args, PyObject* kwargs)
	{
		return py::handleExc([&]()
		{
			PyObject* dataset;
			static const char* kwlist[] = { "dataset", nullptr};
			if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", (char**)kwlist,
				&dataset
			)) return -1;
			Py_INCREF(dataset);
			self->obj = py::UniqueCObj<HSDatasetObject>{ (HSDatasetObject*)dataset };
			self->obj->hsd.reset();
			return 0;
		});
	}

	static HSDatasetIterObject* iter(HSDatasetIterObject* self)
	{
		Py_INCREF(self);
		return self;
	}

	static PyObject* iternext(HSDatasetIterObject* self)
	{
		const size_t batchSize = self->obj->hsd.getBatchSize();
		const size_t windowSize = self->obj->hsd.getWindowSize();
		npy_intp sizes[2] = { (npy_intp)batchSize * 4, (npy_intp)windowSize };
		py::UniqueObj inData{ PyArray_EMPTY(2, sizes, NPY_INT64, 0) };
		py::UniqueObj outData{ PyArray_EMPTY(1, sizes, NPY_INT64, 0) };
		py::UniqueObj lmLProbsData{ PyArray_EMPTY(1, sizes, NPY_FLOAT32, 0) };
		py::UniqueObj outNgramNodeData{ PyArray_EMPTY(1, sizes, NPY_INT64, 0) };
		float restLm = 0;
		uint32_t restLmCnt = 0;

		const size_t sz = self->obj->hsd.next(
			(int64_t*)PyArray_DATA((PyArrayObject*)inData.get()),
			(int64_t*)PyArray_DATA((PyArrayObject*)outData.get()),
			(float*)PyArray_DATA((PyArrayObject*)lmLProbsData.get()),
			(int64_t*)PyArray_DATA((PyArrayObject*)outNgramNodeData.get()),
			restLm,
			restLmCnt
		);
		if (!sz) return nullptr;

		//if (sz < batchSize)
		{
			py::UniqueObj slice{ PySlice_New(nullptr, py::buildPyValue(sz).get(), nullptr)};
			inData = py::UniqueObj{ PyObject_GetItem(inData.get(), slice.get())};
			outData = py::UniqueObj{ PyObject_GetItem(outData.get(), slice.get())};
			lmLProbsData = py::UniqueObj{ PyObject_GetItem(lmLProbsData.get(), slice.get())};
			outNgramNodeData = py::UniqueObj{ PyObject_GetItem(outNgramNodeData.get(), slice.get())};
		}
		return py::buildPyTuple(inData, outData, lmLProbsData, outNgramNodeData, restLm, restLmCnt).release();
	}
};

py::TypeWrapper<HSDatasetIterObject> _HSDatasetIterSetter{ [](PyTypeObject& obj)
{
} };

struct KiwiObject : py::CObject<KiwiObject>
{
	static constexpr const char* _name = "kiwipiepy._Kiwi";
	static constexpr const char* _name_in_module = "_Kiwi";
	static constexpr int _flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;

	KiwiBuilder builder;
	Kiwi kiwi;
	TypoTransformerObject* typos = nullptr;
	float typoCostThreshold = 2.5f;

	static int init(KiwiObject *self, PyObject *args, PyObject *kwargs)
	{
		return py::handleExc([&]()
		{
			const char* modelPath = nullptr;
			size_t numThreads = 0, options = 3;
			int integrateAllomorph = -1, loadDefaultDict = -1, loadTypoDict = 0;
			size_t sbg = 0;
			PyObject* typos = nullptr;
			float typoCostThreshold = 2.5f;
			static const char* kwlist[] = { "num_workers", "model_path", "integrate_allomorph", 
				"load_default_dict", "load_typo_dict",
				"sbg", "typos", "typo_cost_threshold", nullptr};
			if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|nzppppOf", (char**)kwlist,
				&numThreads, &modelPath, &integrateAllomorph, &loadDefaultDict, &loadTypoDict, &sbg, &typos, &typoCostThreshold
			)) return -1;

			if (typos == nullptr || typos == Py_None)
			{
				self->typos = nullptr;
			}
			else if (PyObject_IsInstance(typos, (PyObject*)py::Type<TypoTransformerObject>))
			{
				self->typos = (TypoTransformerObject*)typos;
			}
			else
			{
				throw py::ValueError{ "invalid `typos` value: " + py::repr(typos)};
			}
			self->typoCostThreshold = typoCostThreshold;

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

			if (loadTypoDict) boptions |= BuildOption::loadTypoDict;

			string spath;
			if (modelPath)
			{
				spath = modelPath;
			}
			else
			{
				py::UniqueObj modelModule{ PyImport_ImportModule("kiwipiepy_model") };
				if (!modelModule) throw py::ExcPropagation{};
				py::UniqueObj pathFunc{ PyObject_GetAttrString(modelModule.get(), "get_model_path")};
				if (!pathFunc) throw py::ExcPropagation{};
				py::UniqueObj pathRet{ PyObject_CallObject(pathFunc.get(), nullptr)};
				if (!pathRet) throw py::ExcPropagation{};
				spath = py::toCpp<string>(pathRet.get());
			}

			self->builder = KiwiBuilder{ spath, numThreads, (BuildOption)boptions, !!sbg };
			return 0;
		});
	}

	void doPrepare()
	{
		if (kiwi.ready()) return;
		kiwi = builder.build(typos ? typos->tt : withoutTypo, typoCostThreshold);
		py::UniqueObj handler{ PyObject_GetAttrString((PyObject*)this, "_on_build") };
		if (handler)
		{
			py::UniqueObj res{ PyObject_CallFunctionObjArgs(handler.get(), nullptr)};
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
	PyObject* join(PyObject* args, PyObject* kwargs);
	PyObject* makeHSDataset(PyObject* args, PyObject* kwargs);

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

	float getTypoCostWeight() const
	{
		return kiwi.getTypoCostWeight();
	}

	void setTypoCostWeight(float v)
	{
		kiwi.setTypoCostWeight(v);
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
		{ "join", PY_METHOD_MEMFN(&KiwiObject::join), METH_VARARGS | METH_KEYWORDS, "" },
		{ "make_hsdataset", PY_METHOD_MEMFN(&KiwiObject::makeHSDataset), METH_VARARGS | METH_KEYWORDS, "" },
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
		{ (char*)"_typo_cost_weight", PY_GETTER_MEMFN(&KiwiObject::getTypoCostWeight), PY_SETTER_MEMFN(&KiwiObject::setTypoCostWeight), "", nullptr },
		{ (char*)"_typo_cost_threshold", PY_GETTER_MEMPTR(&KiwiObject::typoCostThreshold), PY_SETTER_MEMPTR(&KiwiObject::typoCostThreshold), "", nullptr },
		{ (char*)"_num_workers", PY_GETTER_MEMFN(&KiwiObject::getNumWorkers), nullptr, "", nullptr },
		{ nullptr },
	};
	obj.tp_methods = methods;
	obj.tp_getset = getsets;
}};

static bool allowInitTokenObject = false;
struct AllowInitToken
{
	AllowInitToken()
	{
		allowInitTokenObject = true;
	}

	~AllowInitToken()
	{
		allowInitTokenObject = false;
	}
};

struct TokenObject : py::CObject<TokenObject>
{
	static constexpr const char* _name = "kiwipiepy.Token";
	static constexpr const char* _name_in_module = "Token";
	static constexpr const char* _doc = Token__doc__;

	u16string _form, _raw_form;
	const char* _tag = nullptr;
	uint32_t _pos = 0, _len = 0, _wordPosition = 0, _sentPosition = 0, _subSentPosition = 0, _lineNumber = 0;
	float _score = 0, _typoCost = 0;
	size_t _morphId = 0;
	const Morpheme* _morph = nullptr;
	const Morpheme* _baseMorph = nullptr;
	bool _regularity = false;

	static int init(TokenObject* self, PyObject* args, PyObject* kwargs)
	{
		return py::handleExc([&]()
		{
			if (allowInitTokenObject) return 0;
			throw py::RuntimeError{ "Cannot create a new instance of `kiwipiepy.Token`." };
		});
	}

	uint32_t end()
	{
		return _pos + _len;
	}

	u16string taggedForm() const
	{
		u16string ret = _form;
		ret.push_back(u'/');
		ret.insert(ret.end(), _tag, _tag + strlen(_tag));
	 	return ret;
	}

	u16string baseForm() const
	{
	 	return kiwi::joinHangul(_baseMorph->getForm());
	}

	size_t baseId() const
	{
		return (_baseMorph - _morph) + _morphId;
	}

	static Py_ssize_t len(TokenObject* self)
	{
		return 4;
	}

	PyObject* regularity()
	{
		if (_tag[0] == 'V') return py::buildPyValue(_regularity).release();
		return py::buildPyValue(nullptr).release();
	}

	u16string lemma() const
	{
		if (_tag[0] == 'V') return _form + u'다';
		else return _form;
	}

	static PyObject* getitem(TokenObject* self, Py_ssize_t idx)
	{
		return py::handleExc([&]()
		{
			if (idx < 0) idx += len(self);
			switch (idx)
			{
			case 0: return py::buildPyValue(self->_form).release();
			case 1: return py::buildPyValue(self->_tag).release();
			case 2: return py::buildPyValue(self->_pos).release();
			case 3: return py::buildPyValue(self->_len).release();
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
			")").release();
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
		{ (char*)"sub_sent_position", PY_GETTER_MEMPTR(&TokenObject::_subSentPosition), nullptr, Token_sub_sent_position__doc__, nullptr },
		{ (char*)"line_number", PY_GETTER_MEMPTR(&TokenObject::_lineNumber), nullptr, Token_line_number__doc__, nullptr },
		{ (char*)"base_form", PY_GETTER_MEMFN(&TokenObject::baseForm), nullptr, Token_base_form__doc__, nullptr },
		{ (char*)"base_id", PY_GETTER_MEMFN(&TokenObject::baseId), nullptr, Token_base_id__doc__, nullptr },
		{ (char*)"tagged_form", PY_GETTER_MEMFN(&TokenObject::taggedForm), nullptr, Token_tagged_form__doc__, nullptr },
		{ (char*)"score", PY_GETTER_MEMPTR(&TokenObject::_score), nullptr, Token_score__doc__, nullptr },
		{ (char*)"typo_cost", PY_GETTER_MEMPTR(&TokenObject::_typoCost), nullptr, Token_typo_cost__doc__, nullptr },
		{ (char*)"raw_form", PY_GETTER_MEMPTR(&TokenObject::_raw_form), nullptr, Token_raw_form__doc__, nullptr },
		{ (char*)"regularity", PY_GETTER_MEMFN(&TokenObject::regularity), nullptr, Token_regularity__doc__, nullptr },
		{ (char*)"lemma", PY_GETTER_MEMFN(&TokenObject::lemma), nullptr, Token_lemma__doc__, nullptr },
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

py::UniqueObj resToPyList(vector<TokenResult>&& res, const Kiwi& kiwi)
{
	AllowInitToken allowInitToken;
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
			tItem->_regularity = !isIrregular(q.tag);
			POSTag tag = clearIrregular(q.tag);
			if (tag == POSTag::vv || tag == POSTag::va || tag == POSTag::vx || tag == POSTag::xsa)
			{
				size_t coda = (tItem->_form.back() - 0xAC00) % 28;
				if (coda == 7 || coda == 17 || coda == 19 || tItem->_form == u"이르")
				{
					if (tItem->_regularity)
					{
						switch (tag)
						{
						case POSTag::vv:
							tItem->_tag = "VV-R";
							break;
						case POSTag::va:
							tItem->_tag = "VA-R";
							break;
						case POSTag::vx:
							tItem->_tag = "VX-R";
							break;
						case POSTag::xsa:
							tItem->_tag = "XSA-R";
							break;
						default:
							break;
						}
					}
					else
					{
						tItem->_tag = tagToString(q.tag);
					}
				}
				else
				{
					tItem->_tag = tagToString(tag);
				}
			}
			else
			{
				tItem->_tag = tagToString(tag);
			}
			tItem->_pos = q.position - u32offset;
			tItem->_len = q.length - u32chrs;
			tItem->_wordPosition = q.wordPosition;
			tItem->_sentPosition = q.sentPosition;
			tItem->_subSentPosition = q.subSentPosition;
			tItem->_lineNumber = q.lineNumber;
			tItem->_score = q.score;
			tItem->_typoCost = q.typoCost;
			tItem->_morph = q.morph;
			tItem->_morphId = kiwi.morphToId(q.morph);
			tItem->_baseMorph = kiwi.idToMorph(q.morph->lmMorphemeId);
			tItem->_raw_form = q.typoCost ? kiwi.getTypoForm(q.typoFormId) : tItem->_form;

			PyList_SET_ITEM(rList.get(), jdx++, item.release());
			u32offset += u32chrs;
		}
		PyList_SET_ITEM(retList.get(), idx++, py::buildPyTuple(move(rList), p.second).release());
	}
	return retList;
}

inline POSTag parseTag(const char* tag)
{
	auto u16 = utf8To16(tag);
	transform(u16.begin(), u16.end(), u16.begin(), static_cast<int(*)(int)>(toupper));
	auto pos = toPOSTag(u16);
	if (clearIrregular(pos) >= POSTag::max) throw py::ValueError{ "Unknown tag value " + py::reprFromCpp(tag) };
	return pos;
}

inline POSTag parseTag(const u16string& tag)
{
	auto u16 = tag;
	transform(u16.begin(), u16.end(), u16.begin(), static_cast<int(*)(int)>(toupper));
	auto pos = toPOSTag(u16);
	if (clearIrregular(pos) >= POSTag::max) throw py::ValueError{ "Unknown tag value " + py::reprFromCpp(tag) };
	return pos;
}

struct MorphemeSetObject : py::CObject<MorphemeSetObject>
{
	static constexpr const char* _name = "kiwipiepy._MorphemeSet";
	static constexpr const char* _name_in_module = "_MorphemeSet";
	static constexpr int _flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;

	py::UniqueCObj<KiwiObject> kiwi;
	std::unordered_set<const kiwi::Morpheme*> morphSet;

	static int init(MorphemeSetObject* self, PyObject* args, PyObject* kwargs)
	{
		return py::handleExc([&]()
		{
			PyObject* kiwi;
			static const char* kwlist[] = { "kiwi", nullptr };
			if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", (char**)kwlist,
				&kiwi
			)) return -1;
			Py_INCREF(kiwi);
			self->kiwi = py::UniqueCObj<KiwiObject>{ (KiwiObject*)kiwi };
			return 0;
		});
	}

	PyObject* update(PyObject* args, PyObject* kwargs)
	{
		return py::handleExc([&]() -> PyObject*
		{
			PyObject* morphs;
			static const char* kwlist[] = { "morphs", nullptr };
			if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", (char**)kwlist,
				&morphs
			)) return nullptr;
			morphSet.clear();

			py::foreach<PyObject*>(morphs, [&](PyObject* item)
			{
				if (PyTuple_Check(item) && PyTuple_GET_SIZE(item) == 2)
				{
					auto form = py::toCpp<string>(PyTuple_GET_ITEM(item, 0));
					auto stag = py::toCpp<string>(PyTuple_GET_ITEM(item, 1));
					POSTag tag = POSTag::unknown;
					if (!stag.empty())
					{
						tag = parseTag(stag.c_str());
					}
					auto m = kiwi->kiwi.findMorpheme(utf8To16(form), tag);
					morphSet.insert(m.begin(), m.end());
				}
				else
				{
					throw py::ForeachFailed{};
				}
			}, "`morphs` must be an iterable of `str`.");

			Py_INCREF(Py_None);
			return Py_None;
		});
	}
};

py::TypeWrapper<MorphemeSetObject> _MorphemeSetSetter{ [](PyTypeObject& obj)
{
	static PyMethodDef methods[] =
	{
		{ "_update", PY_METHOD_MEMFN(&MorphemeSetObject::update), METH_VARARGS | METH_KEYWORDS, ""},
		{ nullptr }
	};
	obj.tp_methods = methods;
} };

struct KiwiResIter : public py::ResultIter<KiwiResIter, vector<TokenResult>>
{
	static constexpr const char* _name = "kiwipiepy._ResIter";
	static constexpr const char* _name_in_module = "_ResIter";

	py::UniqueCObj<KiwiObject> kiwi;
	py::UniqueCObj<MorphemeSetObject> blocklist;
	size_t topN = 1;
	Match matchOptions = Match::all;

	~KiwiResIter()
	{
		waitQueue();
	}

	py::UniqueObj buildPy(vector<TokenResult>&& v)
	{
		return py::handleExc([&]()
		{
			if (v.size() > topN) v.erase(v.begin() + topN, v.end());
			return resToPyList(move(v), kiwi->kiwi);
		});
	}

	future<vector<TokenResult>> feedNext(py::SharedObj&& next)
	{
		if (!PyUnicode_Check(next)) throw py::ValueError{ "`analyze` requires an instance of `str` or an iterable of `str`." };
		return kiwi->kiwi.asyncAnalyze(PyUnicode_AsUTF8(next), topN, matchOptions, blocklist ? &blocklist->morphSet : nullptr);
	}
};

py::TypeWrapper<KiwiResIter> _ResIterSetter{ [](PyTypeObject&)
{
} };

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
		return py::buildPyValue(added).release();
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
			return py::buildPyValue(added).release();
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
			return py::toCpp<u16string>(ret.get());
		}, score);
		if (!added.empty()) kiwi = Kiwi{};
		return py::buildPyValue(added).release();
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
		return py::buildPyValue(ret).release();
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
			auto ret = py::toCpp<u16string>(item.get());
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
			PyList_SET_ITEM(retList.get(), idx++, v.release());
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
			PyList_SET_ITEM(retList.get(), idx++, v.release());
		}
		return retList.release();
	});
}

PyObject* KiwiObject::analyze(PyObject* args, PyObject *kwargs)
{
	return py::handleExc([&]() -> PyObject*
	{
		size_t topN = 1, matchOptions = (size_t)Match::all, echo = 0;
		PyObject* text, *blockList = Py_None;
		static const char* kwlist[] = { "text", "top_n", "match_options", "echo", "blocklist", nullptr};
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|nnpO", (char**)kwlist, &text, &topN, &matchOptions, &echo, &blockList)) return nullptr;

		doPrepare();
		if (PyUnicode_Check(text))
		{
			const unordered_set<const Morpheme*>* morphs = nullptr;
			if (blockList != Py_None) morphs = &((MorphemeSetObject*)blockList)->morphSet;
			auto res = kiwi.analyze(PyUnicode_AsUTF8(text), max(topN, (size_t)10), (Match)matchOptions, morphs);
			if (res.size() > topN) res.erase(res.begin() + topN, res.end());
			return resToPyList(move(res), kiwi).release();
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
			if (blockList != Py_None)
			{
				ret->blocklist = py::UniqueCObj<MorphemeSetObject>{ (MorphemeSetObject*)blockList };
				Py_INCREF(blockList);
			}
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
			PyList_Append(ret.get(), resToPyList(move(res), kiwi).get());
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

PyObject* KiwiObject::join(PyObject* args, PyObject* kwargs)
{
	PyObject* morphs;
	size_t lm_search = 1;
	static const char* kwlist[] = { "morphs", "lm_search", nullptr};
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|p", (char**)kwlist, &morphs, &lm_search)) return nullptr;
	return py::handleExc([&]()
	{
		doPrepare();
		auto joiner = kiwi.newJoiner(!!lm_search);
		py::foreach<PyObject*>(morphs, [&](PyObject* item)
		{
			if (PyObject_IsInstance(item, _TokenSetter.getTypeObj()))
			{
				auto& token = *((TokenObject*)item);
				if (token._morph->kform && !token._morph->kform->empty())
				{
					joiner.add(token._morphId);
				}
				else
				{
					joiner.add(token._form, token._morph->tag, false);
				}
			}
			else if (PyTuple_Check(item) && PyTuple_Size(item) == 2)
			{
				const char* form = py::toCpp<const char*>(PyTuple_GET_ITEM(item, 0));
				const char* tag = py::toCpp<const char*>(PyTuple_GET_ITEM(item, 1));
				const char* p = strchr(tag, '-');
				joiner.add(utf8To16(form), parseTag(tag), p ? false : true);
			}
			else
			{
				throw py::ConversionFail{ "`morphs` must be an iterable of `Tuple[str, str]`." };
			}
		}, "`morphs` must be an iterable of `Tuple[str, str]`.");
		return py::buildPyValue(joiner.getU8()).release();
	});
}

PyObject* KiwiObject::makeHSDataset(PyObject* args, PyObject* kwargs)
{
	return py::handleExc([&]() -> PyObject*
	{
		PyObject* inputPathes, * tokenFilter = nullptr;
		size_t batchSize, windowSize, numWorkers, seed = 42;
		double dropout = 0, splitRatio = 0;
		static const char* kwlist[] = { "input_pathes", "batch_size", "window_size", "num_workers", "dropout", "token_filter", "split_ratio", "seed", nullptr};
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Onnn|dOdn", (char**)kwlist, &inputPathes, &batchSize, &windowSize, &numWorkers, &dropout, &tokenFilter, &splitRatio, &seed)) return nullptr;

		KiwiBuilder::TokenFilter tf;
		if (tokenFilter && tokenFilter != Py_None)
		{
			tf = [&](const u16string& form, POSTag tag)
			{
				py::UniqueObj ret{ PyObject_CallObject(tokenFilter, py::buildPyTuple(form, tagToString(tag)).get()) };
				if (!ret) throw py::ExcPropagation{};
				auto truth = PyObject_IsTrue(ret.get());
				if (truth < 0) throw py::ExcPropagation{};
				return !!truth;
			};
		}

		HSDataset anotherDataset;
		auto dataset = builder.makeHSDataset(py::toCpp<vector<string>>(inputPathes), batchSize, windowSize, numWorkers, dropout, tf, splitRatio, &anotherDataset);
		dataset.seed(seed);
		if (splitRatio == 0)
		{
			py::UniqueObj ret{ PyObject_CallObject((PyObject*)py::Type<HSDatasetObject>, nullptr) };
			((HSDatasetObject*)ret.get())->hsd = move(dataset);
			return ret.release();
		}
		else
		{
			py::UniqueObj ret1{ PyObject_CallObject((PyObject*)py::Type<HSDatasetObject>, nullptr) };
			((HSDatasetObject*)ret1.get())->hsd = move(dataset);
			py::UniqueObj ret2{ PyObject_CallObject((PyObject*)py::Type<HSDatasetObject>, nullptr) };
			((HSDatasetObject*)ret2.get())->hsd = move(anotherDataset);
			auto ret = py::buildPyTuple(ret1, ret2);
			return ret.release();
		}
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
	import_array();
	return moduleInit();
}
