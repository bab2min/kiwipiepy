#define _SILENCE_CXX17_RESULT_OF_DEPRECATION_WARNING

#include <stdexcept>
#include <fstream>
#include <algorithm>

#define USE_NUMPY
#define MAIN_MODULE

#include "PyUtils.h"

#include <kiwi/Kiwi.h>
#include <kiwi/Dataset.h>
#include <kiwi/SwTokenizer.h>
#include <kiwi/SubstringExtractor.h>

using namespace std;
using namespace kiwi;

vector<pair<u16string, size_t>> pyExtractSubstrings(const u16string& str, size_t minCnt, size_t minLength, size_t maxLength, bool longestOnly, const u16string& stopChr)
{
	if (stopChr.size() > 1)
	{
		throw py::ValueError{ "stopChr must be a single character." };
	}

	return extractSubstrings(str.data(), str.data() + str.size(), minCnt, minLength, maxLength, longestOnly, stopChr.empty() ? 0 : stopChr[0]);
}

static py::Module gModule{ "_kiwipiepy", "Kiwi API for Python", [](PyModuleDef& def)
{
	static PyMethodDef methods[] =
	{
		{ "_extract_substrings", PY_METHOD(&pyExtractSubstrings), METH_VARARGS | METH_KEYWORDS, "" },
		{ nullptr }
	};
	def.m_methods = methods;

} };

struct TypoTransformerObject : py::CObject<TypoTransformerObject>
{
	static constexpr const char* _name = "kiwipiepy._TypoTransformer";
	static constexpr const char* _name_in_module = "_TypoTransformer";
	static constexpr int _flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;

	TypoTransformer tt;
	PreparedTypoTransformer ptt;
	bool prepared = false;

	using _InitArgs = std::tuple<PyObject*, float, float>;

	TypoTransformerObject() = default;

	TypoTransformerObject(PyObject* defs, float continualTypoCost, float lengtheningTypoCost)
	{
		if (continualTypoCost)
		{
			tt.setContinualTypoCost(continualTypoCost);
		}

		if (lengtheningTypoCost)
		{
			tt.setLengtheningTypoCost(lengtheningTypoCost);
		}

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
					tt.addTypo(utf8To16(o), utf8To16(e), cost, condVowel);
				}
			}
		}, "`defs` must be an iterable of Tuple[List, List, float, str].");
	}

	py::UniqueObj copy(PyObject* type)
	{
		auto obj = py::makeNewObject<TypoTransformerObject>((PyTypeObject*)type);
		obj->tt = tt;
		return obj;
	}

	void update(PyObject* obj)
	{
		if (!PyObject_IsInstance(obj, (PyObject*)py::Type<TypoTransformerObject>))
		{
			throw py::ValueError{ "`obj` must be an instance of `TypoTransformer`." };
		}
		tt.update(((TypoTransformerObject*)obj)->tt);
	}

	void scaleCost(float scale)
	{
		tt.scaleCost(scale);
	}

	float getContinualTypoCost() const
	{
		return tt.getContinualTypoCost();
	}

	float getLengtheningTypoCost() const
	{
		return tt.getLengtheningTypoCost();
	}

	py::UniqueObj getDefs() const
	{
		py::UniqueObj ret{ PyList_New(0) };
		vector<pair<tuple<KString, KString, CondVowel>, float>> defs{ tt.getTypos().begin(), tt.getTypos().end() };
		sort(defs.begin(), defs.end());
		for (auto& p : defs)
		{
			const auto orig = joinHangul(get<0>(p.first));
			const auto error = joinHangul(get<1>(p.first));
			const auto cond = get<2>(p.first);
			const auto cost = p.second;
			py::UniqueObj pyCond = py::buildPyValue(nullptr);
			if (cond == CondVowel::any)
			{
				pyCond = py::buildPyValue("any");
			}
			else if (cond == CondVowel::vowel)
			{
				pyCond = py::buildPyValue("vowel");
			}
			else if (cond == CondVowel::applosive)
			{
				pyCond = py::buildPyValue("applosive");
			}
			PyList_Append(ret.get(), py::buildPyTuple(orig, error, cost, pyCond).get());
		}
		return ret;
	}

	PreparedTypoTransformer& getPtt()
	{
		if (!prepared) ptt = tt.prepare();
		prepared = true;
		return ptt;
	}

	py::UniqueObj generate(const char* orig, float costThreshold = 2.5)
	{
		py::UniqueObj ret{ PyList_New(0) };
		for (auto r : getPtt().generate(utf8To16(orig), costThreshold))
		{
			PyList_Append(ret.get(), py::buildPyTuple(r.str, r.cost).get());
		}
		return ret;
	}
};

py::TypeWrapper<TypoTransformerObject> _TypoTransformerSetter{ gModule, [](PyTypeObject& obj)
{
	static PyMethodDef methods[] =
	{
		{ "generate", PY_METHOD(&TypoTransformerObject::generate), METH_VARARGS | METH_KEYWORDS, ""},
		{ "copy", PY_METHOD(&TypoTransformerObject::copy), METH_VARARGS | METH_KEYWORDS, ""},
		{ "update", PY_METHOD(&TypoTransformerObject::update), METH_VARARGS | METH_KEYWORDS, ""},
		{ "scale_cost", PY_METHOD(&TypoTransformerObject::scaleCost), METH_VARARGS | METH_KEYWORDS, ""},
		{ nullptr }
	};
	obj.tp_methods = methods;

	static PyGetSetDef getsets[] =
	{
		{ "_continual_typo_cost", PY_GETTER(&TypoTransformerObject::getContinualTypoCost), nullptr, "", nullptr },
		{ "_lengthening_typo_cost", PY_GETTER(&TypoTransformerObject::getLengtheningTypoCost), nullptr, "", nullptr },
		{ "_defs", PY_GETTER(&TypoTransformerObject::getDefs), nullptr, "", nullptr },
		{ nullptr },
	};
	obj.tp_getset = getsets;
} };

struct HSDatasetIterObject;

struct HSDatasetObject : py::CObject<HSDatasetObject>
{
	static constexpr const char* _name = "kiwipiepy._HSDataset";
	static constexpr const char* _name_in_module = "_HSDataset";
	static constexpr int _flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;

	HSDataset hsd;

	py::UniqueCObj<HSDatasetIterObject> iter() const
	{
		py::UniqueCObj<HSDatasetIterObject> ret{ (HSDatasetIterObject*)PyObject_CallFunctionObjArgs((PyObject*)py::Type<HSDatasetIterObject>, this, nullptr) };
		return ret;
	}

	size_t getVocabSize() const
	{
		return hsd.vocabSize();
	}

	size_t getKnlmVocabSize() const
	{
		return hsd.getKnlmVocabSize();
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

	Py_ssize_t len() const
	{
		return hsd.numEstimBatches();
	}

	std::vector<size_t> estimVocabFrequency() const
	{
		return hsd.estimVocabFrequency();
	}

	py::UniqueObj getVocabInfo(size_t index) const
	{
		if (index >= hsd.vocabSize()) throw py::ValueError{ to_string(index) };
		return py::buildPyTuple(hsd.vocabForm(index), tagToString(hsd.vocabInfo(index).tag));
	}

	py::UniqueObj getSent(size_t index, bool augment = false)
	{
		if (index >= hsd.numSents()) throw py::ValueError{ to_string(index) };
		if (augment)
		{
			auto sent = hsd.getAugmentedSent(index);
			return py::buildPyValueTransform(sent.begin(), sent.end(), [](size_t v) { return (uint32_t)v; });
		}
		else
		{
			auto sent = hsd.getSent(index);
			return py::buildPyValueTransform(sent.begin(), sent.end(), [](size_t v) { return (uint32_t)v; });
		}
	}
};

py::TypeWrapper<HSDatasetObject> _HSDatasetSetter{ gModule, [](PyTypeObject& obj)
{
	static PyMethodDef methods[] =
	{
		{ "get_vocab_info", PY_METHOD(&HSDatasetObject::getVocabInfo), METH_VARARGS | METH_KEYWORDS, ""},
		{ "get_sent", PY_METHOD(&HSDatasetObject::getSent), METH_VARARGS | METH_KEYWORDS, ""},
		{ "estim_vocab_frequency", PY_METHOD(&HSDatasetObject::estimVocabFrequency), METH_VARARGS | METH_KEYWORDS, ""},
		{ nullptr }
	};
	static PyGetSetDef getsets[] =
	{
		{ (char*)"vocab_size", PY_GETTER(&HSDatasetObject::getVocabSize), nullptr, "", nullptr },
		{ (char*)"knlm_vocab_size", PY_GETTER(&HSDatasetObject::getKnlmVocabSize), nullptr, "", nullptr },
		{ (char*)"ngram_node_size", PY_GETTER(&HSDatasetObject::getNgramNodeSize), nullptr, "", nullptr },
		{ (char*)"batch_size", PY_GETTER(&HSDatasetObject::getBatchSize), nullptr, "", nullptr },
		{ (char*)"window_size", PY_GETTER(&HSDatasetObject::getWindowSize), nullptr, "", nullptr },
		{ (char*)"num_sents", PY_GETTER(&HSDatasetObject::numSents), nullptr, "", nullptr },
		{ nullptr },
	};
	static PySequenceMethods seq = {
		PY_LENFUNC(&HSDatasetObject::len),
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

	using _InitArgs = std::tuple<py::UniqueCObj<HSDatasetObject>>;

	HSDatasetIterObject() = default;

	HSDatasetIterObject(py::UniqueCObj<HSDatasetObject>&& dataset)
	{
		obj = std::move(dataset);
		obj->hsd.reset();
	}

	py::UniqueCObj<HSDatasetIterObject> iter() const
	{
		Py_INCREF(this);
		return py::UniqueCObj<HSDatasetIterObject>(const_cast<HSDatasetIterObject*>(this));
	}

	py::UniqueObj iternext()
	{
		const size_t batchSize = obj->hsd.getBatchSize();
		const size_t windowSize = obj->hsd.getWindowSize();
		npy_intp sizes[2] = { (npy_intp)batchSize * 4, (npy_intp)windowSize };
		py::UniqueObj inData{ PyArray_EMPTY(2, sizes, NPY_INT64, 0) };
		py::UniqueObj outData{ PyArray_EMPTY(1, sizes, NPY_INT64, 0) };
		py::UniqueObj lmLProbsData{ PyArray_EMPTY(1, sizes, NPY_FLOAT32, 0) };
		py::UniqueObj outNgramNodeData{ PyArray_EMPTY(1, sizes, NPY_INT64, 0) };
		float restLm = 0;
		uint32_t restLmCnt = 0;

		const size_t sz = obj->hsd.next(
			(int64_t*)PyArray_DATA((PyArrayObject*)inData.get()),
			(int64_t*)PyArray_DATA((PyArrayObject*)outData.get()),
			(float*)PyArray_DATA((PyArrayObject*)lmLProbsData.get()),
			(int64_t*)PyArray_DATA((PyArrayObject*)outNgramNodeData.get()),
			restLm,
			restLmCnt
		);
		if (!sz) throw py::ExcPropagation{};

		//if (sz < batchSize)
		{
			py::UniqueObj slice{ PySlice_New(nullptr, py::buildPyValue(sz).get(), nullptr)};
			inData = py::UniqueObj{ PyObject_GetItem(inData.get(), slice.get())};
			outData = py::UniqueObj{ PyObject_GetItem(outData.get(), slice.get())};
			lmLProbsData = py::UniqueObj{ PyObject_GetItem(lmLProbsData.get(), slice.get())};
			outNgramNodeData = py::UniqueObj{ PyObject_GetItem(outNgramNodeData.get(), slice.get())};
		}
		return py::buildPyTuple(inData, outData, lmLProbsData, outNgramNodeData, restLm, restLmCnt);
	}
};

py::TypeWrapper<HSDatasetIterObject> _HSDatasetIterSetter{ gModule, [](PyTypeObject& obj)
{
} };

struct KNLangModelObject;

struct KNLangModelNextTokensResultObject : py::CObject<KNLangModelNextTokensResultObject>
{
	static constexpr const char* _name = "kiwipiepy._KNLangModelNextTokensResult";
	static constexpr const char* _name_in_module = "_KNLangModelNextTokensResult";
	static constexpr int _flags = Py_TPFLAGS_DEFAULT;

	using _InitArgs = std::tuple<>;

	py::UniqueObj inArray, outIdx, outLl;
	py::UniqueCObj<KNLangModelObject> parent;

	mutable std::future<void> future;

	size_t len() const
	{
		return 2;
	}

	py::UniqueObj getitem(Py_ssize_t idx) const
	{
		if (future.valid())
		{
			future.get();
		}

		if (idx < 0) idx += len();
		switch(idx)
		{
		case 0:
			return py::buildPyValue(outIdx);
		case 1:
			return py::buildPyValue(outLl);
		default:
			throw py::IndexError{ "Index out of range." };
		}
	}
};


py::TypeWrapper<KNLangModelNextTokensResultObject> _KNLangModelNextTokensResultObjectSetter{ gModule, [](PyTypeObject& obj)
{
	static PySequenceMethods seq = {
		PY_LENFUNC(&KNLangModelNextTokensResultObject::len),
		nullptr,
		nullptr,
		PY_SSIZEARGFUNC(&KNLangModelNextTokensResultObject::getitem),
	};
	obj.tp_as_sequence = &seq;
} };

struct KNLangModelEvaluateResultObject : py::CObject<KNLangModelEvaluateResultObject>
{
	static constexpr const char* _name = "kiwipiepy._KNLangModelEvaluateResult";
	static constexpr const char* _name_in_module = "_KNLangModelEvaluateResult";
	static constexpr int _flags = Py_TPFLAGS_DEFAULT;

	using _InitArgs = std::tuple<>;

	py::UniqueObj inArray, outLl;
	py::UniqueCObj<KNLangModelObject> parent;

	mutable std::future<void> future;

	size_t len() const
	{
		return PyObject_Length(outLl.get());
	}

	py::UniqueObj getitem(py::UniqueObj arg) const
	{
		if (future.valid())
		{
			future.get();
		}
		return py::UniqueObj{ PyObject_GetItem(outLl.get(), arg.get()) };
	}

	py::UniqueObj getattr(py::UniqueObj arg) const
	{
		if (PyUnicode_Check(arg))
		{
			auto argStr = py::toCpp<string>(arg.get());
			if (argStr == "__dict__")
			{
				throw py::AttributeError{ "__dict__" };
			}
		}
		py::UniqueObj ret{ PyObject_GenericGetAttr((PyObject*)this, arg.get()) };
		if (ret) return ret;
		PyErr_Clear();

		if (future.valid())
		{
			future.get();
		}
		return py::UniqueObj{ PyObject_GetAttr(outLl.get(), arg.get()) };
	}

	py::UniqueObj iter() const
	{
		return py::UniqueObj{ PyObject_GetIter(outLl.get()) };
	}

	py::UniqueObj dir() const
	{
		return py::UniqueObj{ PyObject_Dir(outLl.get()) };
	}
	
};


py::TypeWrapper<KNLangModelEvaluateResultObject> _KNLangModelEvaluateResultObjectSetter{ gModule, [](PyTypeObject& obj)
{
	static PyMappingMethods map = {
		PY_LENFUNC(&KNLangModelEvaluateResultObject::len),
		PY_BINARYFUNC(&KNLangModelEvaluateResultObject::getitem),
		nullptr,
	};
	obj.tp_as_mapping = &map;
	obj.tp_getattro = PY_BINARYFUNC(&KNLangModelEvaluateResultObject::getattr);

	static PyMethodDef methods[] = 	{
		{ "__dir__", PY_METHOD(&KNLangModelEvaluateResultObject::dir), METH_VARARGS | METH_KEYWORDS, ""},
		{ nullptr }
	};
	obj.tp_methods = methods;
} };

struct KNLangModelObject : py::CObject<KNLangModelObject>
{
	static constexpr const char* _name = "kiwipiepy._KNLangModel";
	static constexpr const char* _name_in_module = "_KNLangModel";
	static constexpr int _flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;

	std::unique_ptr<lm::KnLangModelBase> langModel;
	std::unique_ptr<utils::ThreadPool> workers;
	ClusterData clusterData;

	using _InitArgs = std::tuple<>;

	void initClusterData()
	{
		if (langModel && langModel->getExtraBuf())
		{
			clusterData = ClusterData{ langModel->getExtraBuf(), langModel->getHeader().extra_buf_size};
		}
	}

	static py::UniqueCObj<KNLangModelObject> fromArrays(
		py::UniqueObj cls,
		py::UniqueObj arrays,
		size_t ngramSize,
		const std::vector<size_t>& minCf,
		size_t bosTokenId,
		size_t eosTokenId,
		size_t unkTokenId,
		const std::vector<std::vector<size_t>>& clusters,
		size_t numWorkers
	)
	{
		PrefixCounter pfCnt{ ngramSize, minCf[0], numWorkers, clusters };

		py::foreach<PyObject*>(arrays.get(), [&](PyObject* item)
		{
			if (!PyArray_Check(item)) throw py::ValueError{ "arrays must be a list of numpy arrays." };
			const size_t dims = PyArray_NDIM((PyArrayObject*)item);
			if (dims != 1) throw py::ValueError{ "arrays must be a list of 1D numpy arrays." };
			const size_t len = PyArray_DIM((PyArrayObject*)item, 0);
			const auto dtype = PyArray_TYPE((PyArrayObject*)item);
			if (dtype == NPY_UINT16 || dtype == NPY_INT16)
			{
				auto* ptr = (const uint16_t*)PyArray_DATA((PyArrayObject*)item);
				pfCnt.addArray(ptr, ptr + len);
			}
			else if (dtype == NPY_UINT32 || dtype == NPY_INT32)
			{
				auto* ptr = (const uint32_t*)PyArray_DATA((PyArrayObject*)item);
				pfCnt.addArray(ptr, ptr + len);
			}
			else if (dtype == NPY_UINT64 || dtype == NPY_INT64)
			{
				auto* ptr = (const uint64_t*)PyArray_DATA((PyArrayObject*)item);
				pfCnt.addArray(ptr, ptr + len);
			}
			else
			{
				throw py::ValueError{ "arrays must be a list of numpy arrays of uint16, uint32 or uint64." };
			}
		}, "arrays must be a list of numpy arrays.");

		auto lm = pfCnt.buildLM(minCf, bosTokenId, eosTokenId, unkTokenId, ArchType::balanced);

		auto* clsType = (PyTypeObject*)cls.get();
		py::UniqueCObj<KNLangModelObject> ret{ (KNLangModelObject*)clsType->tp_new(clsType, nullptr, nullptr) };
		ret->langModel = std::move(lm);
		ret->initClusterData();
		if (numWorkers >= 1)
		{
			ret->workers = std::make_unique<utils::ThreadPool>(numWorkers);
		}
		return ret;
	}

	size_t ngramSize() const
	{
		return langModel->getHeader().order;
	}

	size_t vocabSize() const
	{
		return langModel->getHeader().vocab_size;
	}

	size_t numNodes() const
	{
		return langModel->getHeader().num_nodes;
	}

	size_t numWorkers() const
	{
		return workers ? workers->size() : 0;
	}

	static py::UniqueObj load(py::UniqueObj cls, const char* path, size_t numWorkers)
	{
		auto lm = lm::KnLangModelBase::create(utils::MMap(path), ArchType::balanced);

		auto* clsType = (PyTypeObject*)cls.get();
		py::UniqueCObj<KNLangModelObject> ret{ (KNLangModelObject*)clsType->tp_new(clsType, nullptr, nullptr) };
		ret->langModel = std::move(lm);
		ret->initClusterData();
		if (numWorkers >= 1)
		{
			ret->workers = std::make_unique<utils::ThreadPool>(numWorkers);
		}
		return ret;
	}

	void save(const char* path) const
	{
		ofstream ofs;
		if (!openFile(ofs, path, ios_base::binary | ios_base::out))
		{
			throw py::OSError{ "Failed to open file: " + string{ path } };
		}
		ofs.write((const char*)langModel->getMemory().get(), langModel->getMemory().size());
	}

	py::UniqueObj nextTokens(py::UniqueObj obj, size_t topN, bool deferred) const
	{
		if (deferred && !workers)
		{
			throw py::ValueError{ "numWorkers must be greater than 0 when `deferred=True`." };
		}
		if (!PyArray_Check(obj.get())) throw py::ValueError{ "obj must be a numpy array." };
		const size_t dims = PyArray_NDIM((PyArrayObject*)obj.get());
		if (dims != 1) throw py::ValueError{ "obj must be a 1D numpy array." };
		const size_t len = PyArray_DIM((PyArrayObject*)obj.get(), 0);
		const auto dtype = PyArray_TYPE((PyArrayObject*)obj.get());
		const void* inData = PyArray_DATA((PyArrayObject*)obj.get());

		npy_intp sizes[2] = { (npy_intp)len, (npy_intp)topN };
		py::UniqueObj outIdx{ PyArray_EMPTY(2, sizes, NPY_UINT32, 0) };
		py::UniqueObj outLl{ PyArray_EMPTY(2, sizes, NPY_FLOAT32, 0) };
		auto* idxData = (uint32_t*)PyArray_DATA((PyArrayObject*)outIdx.get());
		auto* llData = (float*)PyArray_DATA((PyArrayObject*)outLl.get());

		if (deferred)
		{
			auto ret = py::makeNewObject<KNLangModelNextTokensResultObject>();
			ret->inArray = move(obj);
			ret->outIdx = move(outIdx);
			ret->outLl = move(outLl);
			Py_INCREF(this);
			ret->parent = py::UniqueCObj<KNLangModelObject>{ (KNLangModelObject*)this };
			if (dtype == NPY_UINT16 || dtype == NPY_INT16)
			{
				ret->future = workers->enqueue([=](size_t threadIdx)
				{
					auto* ptr = (const uint16_t*)inData;
					langModel->predictTopN(ptr, ptr + len, topN, idxData, llData);
				});
			}
			else if (dtype == NPY_UINT32 || dtype == NPY_INT32)
			{
				ret->future = workers->enqueue([=](size_t threadIdx)
				{
					auto* ptr = (const uint32_t*)inData;
					langModel->predictTopN(ptr, ptr + len, topN, idxData, llData);
				});
			}
			else if (dtype == NPY_UINT64 || dtype == NPY_INT64)
			{
				ret->future = workers->enqueue([=](size_t threadIdx)
				{
					auto* ptr = (const uint64_t*)inData;
					langModel->predictTopN(ptr, ptr + len, topN, idxData, llData);
				});
			}
			else
			{
				throw py::ValueError{ "obj must be a numpy array of uint16, uint32 or uint64." };
			}
			return ret;
		}
		else
		{
			if (dtype == NPY_UINT16 || dtype == NPY_INT16)
			{
				auto* ptr = (const uint16_t*)inData;
				langModel->predictTopN(ptr, ptr + len, topN, idxData, llData);
			}
			else if (dtype == NPY_UINT32 || dtype == NPY_INT32)
			{
				auto* ptr = (const uint32_t*)inData;
				langModel->predictTopN(ptr, ptr + len, topN, idxData, llData);
			}
			else if (dtype == NPY_UINT64 || dtype == NPY_INT64)
			{
				auto* ptr = (const uint64_t*)inData;
				langModel->predictTopN(ptr, ptr + len, topN, idxData, llData);
			}
			else
			{
				throw py::ValueError{ "obj must be a numpy array of uint16, uint32 or uint64." };
			}
			return py::buildPyTuple(move(outIdx), move(outLl));
		}
	}

	template<class Ty>
	void evaluateWithCluster(const Ty* data, size_t length, float* llData) const
	{
		ptrdiff_t node = 0;
		for (size_t i = 0; i < length; ++i)
		{
			auto id = clusterData.cluster(data[i]);
			auto score = clusterData.score(data[i]);
			llData[i] = langModel->progress(node, id) + score;
		}
	}

	py::UniqueObj evaluate(py::UniqueObj obj, bool deferred) const
	{
		if (deferred && !workers)
		{
			throw py::ValueError{ "numWorkers must be greater than 0 when `deferred=True`." };
		}
		if (!PyArray_Check(obj.get())) throw py::ValueError{ "obj must be a numpy array." };
		const size_t dims = PyArray_NDIM((PyArrayObject*)obj.get());
		if (dims != 1) throw py::ValueError{ "obj must be a 1D numpy array." };
		const size_t len = PyArray_DIM((PyArrayObject*)obj.get(), 0);
		const auto dtype = PyArray_TYPE((PyArrayObject*)obj.get());
		const void* inData = PyArray_DATA((PyArrayObject*)obj.get());

		npy_intp sizes[1] = { (npy_intp)len, };
		py::UniqueObj outLl{ PyArray_EMPTY(1, sizes, NPY_FLOAT32, 0) };
		auto* llData = (float*)PyArray_DATA((PyArrayObject*)outLl.get());
		if (deferred)
		{
			auto ret = py::makeNewObject<KNLangModelEvaluateResultObject>();
			ret->inArray = move(obj);
			ret->outLl = move(outLl);
			Py_INCREF(this);
			ret->parent = py::UniqueCObj<KNLangModelObject>{ (KNLangModelObject*)this };
			if (dtype == NPY_UINT16 || dtype == NPY_INT16)
			{
				ret->future = workers->enqueue([=](size_t threadIdx)
				{
					auto* ptr = (const uint16_t*)inData;
					evaluateWithCluster(ptr, len, llData);
				});
			}
			else if (dtype == NPY_UINT32 || dtype == NPY_INT32)
			{
				ret->future = workers->enqueue([=](size_t threadIdx)
				{
					auto* ptr = (const uint32_t*)inData;
					evaluateWithCluster(ptr, len, llData);
				});
			}
			else if (dtype == NPY_UINT64 || dtype == NPY_INT64)
			{
				ret->future = workers->enqueue([=](size_t threadIdx)
				{
					auto* ptr = (const uint64_t*)inData;
					evaluateWithCluster(ptr, len, llData);
				});
			}
			else
			{
				throw py::ValueError{ "obj must be a numpy array of uint16, uint32 or uint64." };
			}
			return ret;
		}
		else
		{
			if (dtype == NPY_UINT16 || dtype == NPY_INT16)
			{
				auto* ptr = (const uint16_t*)inData;
				evaluateWithCluster(ptr, len, llData);
			}
			else if (dtype == NPY_UINT32 || dtype == NPY_INT32)
			{
				auto* ptr = (const uint32_t*)inData;
				evaluateWithCluster(ptr, len, llData);
			}
			else if (dtype == NPY_UINT64 || dtype == NPY_INT64)
			{
				auto* ptr = (const uint64_t*)inData;
				evaluateWithCluster(ptr, len, llData);
			}
			else
			{
				throw py::ValueError{ "obj must be a numpy array of uint16, uint32 or uint64." };
			}
			return outLl;
		}
	}
};

py::TypeWrapper<KNLangModelObject> _KNLangModelObjectSetter{ gModule, [](PyTypeObject& obj)
{
	static PyMethodDef methods[] =
	{
		{ "from_arrays", PY_METHOD(&KNLangModelObject::fromArrays), METH_VARARGS | METH_KEYWORDS | METH_STATIC, ""},
		{ "load", PY_METHOD(&KNLangModelObject::load), METH_VARARGS | METH_KEYWORDS | METH_STATIC, ""},
		{ "save", PY_METHOD(&KNLangModelObject::save), METH_VARARGS | METH_KEYWORDS, ""},
		{ "next_tokens", PY_METHOD(&KNLangModelObject::nextTokens), METH_VARARGS | METH_KEYWORDS, ""},
		{ "evaluate", PY_METHOD(&KNLangModelObject::evaluate), METH_VARARGS | METH_KEYWORDS, ""},
		{ nullptr }
	};
	static PyGetSetDef getsets[] =
	{
		{ "_ngram_size", PY_GETTER(&KNLangModelObject::ngramSize), nullptr, "", nullptr },
		{ "_vocab_size", PY_GETTER(&KNLangModelObject::vocabSize), nullptr, "", nullptr },
		{ "_num_nodes", PY_GETTER(&KNLangModelObject::numNodes), nullptr, "", nullptr },
		{ "_num_workers", PY_GETTER(&KNLangModelObject::numWorkers), nullptr, "", nullptr },
		{ nullptr },
	};
	obj.tp_methods = methods;
	obj.tp_getset = getsets;
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

	using _InitArgs = std::tuple<
		size_t,
		std::optional<const char*>,
		bool,
		bool,
		bool,
		bool,
		bool,
		PyObject*,
		float
	>;

	KiwiObject() = default;

	KiwiObject(size_t numThreads, 
		std::optional<const char*> modelPath = {}, 
		bool integrateAllomorph = true, 
		bool loadDefaultDict = true, 
		bool loadTypoDict = true, 
		bool loadMultiDict = true,
		bool sbg = false, 
		PyObject* _typos = nullptr, 
		float _typoCostThreshold = 2.5f
	)
	{
		if (_typos == nullptr || _typos == Py_None)
		{
			_typos = nullptr;
		}
		else if (PyObject_IsInstance(_typos, (PyObject*)py::Type<TypoTransformerObject>))
		{
			typos = (TypoTransformerObject*)_typos;
		}
		else
		{
			throw py::ValueError{ "invalid `typos` value: " + py::repr(_typos)};
		}
		typoCostThreshold = _typoCostThreshold;

		BuildOption boptions = BuildOption::none;
		boptions |= integrateAllomorph ? BuildOption::integrateAllomorph : BuildOption::none;
		boptions |= loadDefaultDict ? BuildOption::loadDefaultDict : BuildOption::none;
		boptions |= loadTypoDict ? BuildOption::loadTypoDict : BuildOption::none;
		boptions |= loadMultiDict ? BuildOption::loadMultiDict : BuildOption::none;

		string spath;
		if (modelPath)
		{
			spath = *modelPath;
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

		builder = KiwiBuilder{ spath, numThreads, (BuildOption)boptions, !!sbg };
	}

	void doPrepare()
	{
		if (kiwi.ready()) return;
		kiwi = builder.build(typos ? typos->tt : getDefaultTypoSet(DefaultTypoSet::withoutTypo), typoCostThreshold);
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

	std::pair<uint32_t, bool> addUserWord(const char* word, const char* tag = "NNP", float score = 0, std::optional<const char*> origWord = {});
	bool addPreAnalyzedWord(const char* form, PyObject* oAnalyzed = nullptr, float score = 0);
	std::vector<std::pair<uint32_t, std::u16string>> addRule(const char* tag, PyObject* replacer, float score = 0);
	py::UniqueObj analyze(PyObject* text, size_t topN = 1, Match matchOptions = Match::all, bool echo = false, PyObject* blockList = Py_None, PyObject* pretokenized = Py_None);
	py::UniqueObj extractAddWords(PyObject* sentences, size_t minCnt = 10, size_t maxWordLen = 10, float minScore = 0.25f, float posScore = -3, bool lmFilter = true);
	py::UniqueObj extractWords(PyObject* sentences, size_t minCnt, size_t maxWordLen = 10, float minScore = 0.25f, float posScore = -3, bool lmFilter = true) const;
	size_t loadUserDictionary(const char* path);
	py::UniqueObj getMorpheme(size_t id);
	py::UniqueObj join(PyObject* morphs, bool lmSearch = true, bool returnPositions = false);
	py::UniqueObj makeHSDataset(PyObject* inputPathes, size_t batchSize, size_t windowSize, size_t numWorkers, 
		float dropout = 0, PyObject* tokenFilter = nullptr, float splitRatio = 0, bool separateDefaultMorpheme = false, size_t seed = 42) const;
	py::UniqueObj listAllScripts() const;

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

py::TypeWrapper<KiwiObject> _KiwiSetter{ gModule, [](PyTypeObject& obj)
{
	static PyMethodDef methods[] =
	{
		{ "add_user_word", PY_METHOD(&KiwiObject::addUserWord), METH_VARARGS | METH_KEYWORDS, ""},
		{ "add_pre_analyzed_word", PY_METHOD(&KiwiObject::addPreAnalyzedWord), METH_VARARGS | METH_KEYWORDS, ""},
		{ "add_rule", PY_METHOD(&KiwiObject::addRule), METH_VARARGS | METH_KEYWORDS, ""},
		{ "load_user_dictionary", PY_METHOD(&KiwiObject::loadUserDictionary), METH_VARARGS | METH_KEYWORDS, "" },
		{ "extract_words", PY_METHOD(&KiwiObject::extractWords), METH_VARARGS | METH_KEYWORDS, "" },
		{ "extract_add_words", PY_METHOD(&KiwiObject::extractAddWords), METH_VARARGS | METH_KEYWORDS, "" },
		{ "analyze", PY_METHOD(&KiwiObject::analyze), METH_VARARGS | METH_KEYWORDS, "" },
		{ "morpheme", PY_METHOD(&KiwiObject::getMorpheme), METH_VARARGS | METH_KEYWORDS, "" },
		{ "join", PY_METHOD(&KiwiObject::join), METH_VARARGS | METH_KEYWORDS, "" },
		{ "make_hsdataset", PY_METHOD(&KiwiObject::makeHSDataset), METH_VARARGS | METH_KEYWORDS, "" },
		{ "list_all_scripts", PY_METHOD(&KiwiObject::listAllScripts), METH_VARARGS | METH_KEYWORDS, "" },
		{ nullptr }
	};
	static PyGetSetDef getsets[] =
	{
		{ (char*)"_cutoff_threshold", PY_GETTER(&KiwiObject::getCutOffThreshold), PY_SETTER(&KiwiObject::setCutOffThreshold), "", nullptr },
		{ (char*)"_integrate_allomorph", PY_GETTER(&KiwiObject::getIntegrateAllomorph), PY_SETTER(&KiwiObject::setIntegrateAllomorph), "", nullptr },
		{ (char*)"_unk_score_bias", PY_GETTER(&KiwiObject::getUnkScoreBias), PY_SETTER(&KiwiObject::setUnkScoreBias), "", nullptr },
		{ (char*)"_unk_score_scale", PY_GETTER(&KiwiObject::getUnkScoreScale), PY_SETTER(&KiwiObject::setUnkScoreScale), "", nullptr },
		{ (char*)"_max_unk_form_size", PY_GETTER(&KiwiObject::getMaxUnkFormSize), PY_SETTER(&KiwiObject::setMaxUnkFormSize), "", nullptr },
		{ (char*)"_space_tolerance", PY_GETTER(&KiwiObject::getSpaceTolerance), PY_SETTER(&KiwiObject::setSpaceTolerance), "", nullptr },
		{ (char*)"_space_penalty", PY_GETTER(&KiwiObject::getSpacePenalty), PY_SETTER(&KiwiObject::setSpacePenalty), "", nullptr },
		{ (char*)"_typo_cost_weight", PY_GETTER(&KiwiObject::getTypoCostWeight), PY_SETTER(&KiwiObject::setTypoCostWeight), "", nullptr },
		{ (char*)"_typo_cost_threshold", PY_GETTER(&KiwiObject::typoCostThreshold), PY_SETTER(&KiwiObject::typoCostThreshold), "", nullptr },
		{ (char*)"_num_workers", PY_GETTER(&KiwiObject::getNumWorkers), nullptr, "", nullptr },
		{ nullptr },
	};
	obj.tp_methods = methods;
	obj.tp_getset = getsets;
}};

struct TokenObject : py::CObject<TokenObject>
{
	static constexpr const char* _name = "kiwipiepy.Token";
	static constexpr const char* _name_in_module = "Token";

	u16string _form, _raw_form;
	const char* _tag = nullptr;
	size_t resultHash = 0;
	uint32_t _pos = 0, _len = 0, _wordPosition = 0, _sentPosition = 0, _subSentPosition = 0, _lineNumber = 0;
	int32_t _pairedToken = -1, _sense = 0;
	float _score = 0, _typoCost = 0;
	size_t _morphId = 0;
	const Morpheme* _morph = nullptr;
	const Morpheme* _baseMorph = nullptr;
	py::UniqueObj _userValue;
	POSTag _rawTag = POSTag::unknown;
	ScriptType _script = ScriptType::unknown;
	bool _regularity = false;

	using _InitArgs = std::tuple<int>;
	
	TokenObject() = default;

	TokenObject(int dummy)
	{
		throw py::RuntimeError{ "Cannot create a new instance of `kiwipiepy.Token`." };
	}

	uint32_t end()
	{
		return _pos + _len;
	}

	tuple<uint32_t, uint32_t> span()
	{
		return make_tuple(_pos, _pos + _len);
	}

	u16string taggedForm() const
	{
		u16string ret = _form;
		ret.push_back(u'/');
		ret += utf8To16(_tag);
	 	return ret;
	}
	
	py::UniqueObj formTag() const
	{
		return py::buildPyTuple(_form, _tag);
	}

	u16string baseForm() const
	{
		return _baseMorph ? kiwi::joinHangul(_baseMorph->getForm()) : u16string{};
	}

	size_t baseId() const
	{
		return (_baseMorph - _morph) + _morphId;
	}

	Py_ssize_t len() const
	{
		return 4;
	}

	py::UniqueObj regularity()
	{
		if (_tag[0] == 'V') return py::buildPyValue(_regularity);
		return py::buildPyValue(nullptr);
	}

	py::UniqueObj script() const
	{
		if (_script == ScriptType::unknown)
		{
			return py::buildPyValue(nullptr);
		}
		else
		{
			return py::buildPyValue(getScriptName(_script));
		}
	}

	u16string lemma() const
	{
		if (_tag[0] == 'V') return _form + u'\uB2E4';
		else return _form;
	}

	py::UniqueObj getitem(Py_ssize_t idx) const
	{
		if (idx < 0) idx += 4;
		switch (idx)
		{
		case 0: return py::buildPyValue(_form);
		case 1: return py::buildPyValue(_tag);
		case 2: return py::buildPyValue(_pos);
		case 3: return py::buildPyValue(_len);
		}
		throw py::IndexError{ "index out of range" };
	}

	std::string repr() const
	{
		if (resultHash)
		{
			if (_sense)
			{
				return "Token("
					"form=" + py::reprFromCpp(_form) + ", "
					"tag=" + py::reprFromCpp(_tag) + ", "
					"start=" + to_string(_pos) + ", "
					"len=" + to_string(_len) + ", "
					"sense=" + to_string(_sense) + ")";
			}
			else
			{
				return "Token("
					"form=" + py::reprFromCpp(_form) + ", "
					"tag=" + py::reprFromCpp(_tag) + ", "
					"start=" + to_string(_pos) + ", "
					"len=" + to_string(_len) + ")";
			}
		}
		else
		{
			if (_sense)
			{
				return "Token("
					"form=" + py::reprFromCpp(_form) + ", "
					"tag=" + py::reprFromCpp(_tag) + ", "
					"sense=" + to_string(_sense) + ")";
			}
			else
			{
				return "Token("
					"form=" + py::reprFromCpp(_form) + ", "
					"tag=" + py::reprFromCpp(_tag) + ")";
			}
		}
	}
};

py::TypeWrapper<TokenObject> _TokenSetter{ gModule, [](PyTypeObject& obj)
{
	static PyGetSetDef getsets[] =
	{
		{ (char*)"form", PY_GETTER(&TokenObject::_form), nullptr, "", nullptr },
		{ (char*)"tag", PY_GETTER(&TokenObject::_tag), nullptr, "", nullptr},
		{ (char*)"start", PY_GETTER(&TokenObject::_pos), nullptr, "", nullptr},
		{ (char*)"len", PY_GETTER(&TokenObject::_len), nullptr, "", nullptr},
		{ (char*)"end", PY_GETTER(&TokenObject::end), nullptr, "", nullptr},
		{ (char*)"span", PY_GETTER(&TokenObject::span), nullptr, "", nullptr},
		{ (char*)"id", PY_GETTER(&TokenObject::_morphId), nullptr, "", nullptr},
		{ (char*)"word_position", PY_GETTER(&TokenObject::_wordPosition), nullptr, "", nullptr},
		{ (char*)"sent_position", PY_GETTER(&TokenObject::_sentPosition), nullptr, "", nullptr},
		{ (char*)"sub_sent_position", PY_GETTER(&TokenObject::_subSentPosition), nullptr, "", nullptr},
		{ (char*)"line_number", PY_GETTER(&TokenObject::_lineNumber), nullptr, "", nullptr},
		{ (char*)"base_form", PY_GETTER(&TokenObject::baseForm), nullptr, "", nullptr},
		{ (char*)"base_id", PY_GETTER(&TokenObject::baseId), nullptr, "", nullptr},
		{ (char*)"tagged_form", PY_GETTER(&TokenObject::taggedForm), nullptr, "", nullptr},
		{ (char*)"form_tag", PY_GETTER(&TokenObject::formTag), nullptr, "", nullptr},
		{ (char*)"score", PY_GETTER(&TokenObject::_score), nullptr, "", nullptr},
		{ (char*)"typo_cost", PY_GETTER(&TokenObject::_typoCost), nullptr, "", nullptr},
		{ (char*)"raw_form", PY_GETTER(&TokenObject::_raw_form), nullptr, "", nullptr},
		{ (char*)"regularity", PY_GETTER(&TokenObject::regularity), nullptr, "", nullptr},
		{ (char*)"lemma", PY_GETTER(&TokenObject::lemma), nullptr, "", nullptr},
		{ (char*)"paired_token", PY_GETTER(&TokenObject::_pairedToken), nullptr, "", nullptr},
		{ (char*)"user_value", PY_GETTER(&TokenObject::_userValue), nullptr, "", nullptr},
		{ (char*)"script", PY_GETTER(&TokenObject::script), nullptr, "", nullptr},
		{ (char*)"sense", PY_GETTER(&TokenObject::_sense), nullptr, "", nullptr},
		{ nullptr },
	};

	static PySequenceMethods seq = {
		PY_LENFUNC(&TokenObject::len),
		nullptr,
		nullptr,
		PY_SSIZEARGFUNC(&TokenObject::getitem),
	};

	obj.tp_getset = getsets;
	obj.tp_as_sequence = &seq;
} };

inline size_t hashTokenInfo(const vector<TokenInfo>& tokens)
{
	size_t ret = 1;
	for (auto& t : tokens)
	{
		ret = ((ret << 3) | (ret >> (sizeof(size_t) * 8 - 3))) ^ hash<u16string>{}(t.str);
	}
	return ret;
}

inline const char* getTagStr(const POSTag tag, const u16string& form)
{
	const bool regularity = !isIrregular(tag);
	const auto stag = clearIrregular(tag);
	if (stag == POSTag::vv || stag == POSTag::va || stag == POSTag::vx || stag == POSTag::xsa)
	{
		size_t coda = (form.back() - 0xAC00) % 28;
		if (coda == 7 || coda == 17 || coda == 19 || form == u"이르")
		{
			if (regularity)
			{
				switch (stag)
				{
				case POSTag::vv:
					return "VV-R";
				case POSTag::va:
					return "VA-R";
				case POSTag::vx:
					return "VX-R";
				case POSTag::xsa:
					return "XSA-R";
				}
			}
		}
	}
	return tagToString(tag);
}

py::UniqueObj resToPyList(vector<TokenResult>&& res, const KiwiObject* kiwiObj, vector<py::UniqueObj>&& userValues = {})
{
	auto& kiwi = kiwiObj->kiwi;
	// set the following objects semi-immortal. (they are neither freed nor managed)
	// it prevents crashes at Python3.12
	static PyObject* userValuesAttr = py::buildPyValue("_user_values").release();
	static PyObject* tagAttr = py::buildPyValue("tag").release();
	py::UniqueObj userValuesObj{ PyObject_GetAttr((PyObject*)kiwiObj, userValuesAttr) };
	PyErr_Clear();
	py::UniqueObj retList{ PyList_New(res.size()) };
	size_t idx = 0;
	for (auto& p : res)
	{
		py::UniqueObj rList{ PyList_New(p.first.size()) };
		size_t jdx = 0;
		size_t u32offset = 0;
		size_t resultHash = hashTokenInfo(p.first);
		for (auto& q : p.first)
		{
			size_t u32chrs = 0;
			for (auto u : q.str)
			{
				if ((u & 0xFC00) == 0xD800) u32chrs++;
			}

			auto tItem = py::makeNewObject<TokenObject>();
			tItem->_form = move(q.str);
			tItem->_regularity = !isIrregular(q.tag);
			tItem->_rawTag = q.tag;
			tItem->resultHash = resultHash;
			tItem->_tag = getTagStr(q.tag, tItem->_form);
			tItem->_pos = q.position - u32offset;
			tItem->_len = q.length - u32chrs;
			tItem->_wordPosition = q.wordPosition;
			tItem->_sentPosition = q.sentPosition;
			tItem->_subSentPosition = q.subSentPosition;
			tItem->_lineNumber = q.lineNumber;
			tItem->_score = q.score;
			tItem->_typoCost = q.typoCost;
			tItem->_morph = q.morph;
			tItem->_morphId = q.morph ? kiwi.morphToId(q.morph) : -1;
			tItem->_baseMorph = q.morph ? (q.morph->origMorphemeId ?  kiwi.idToMorph(q.morph->origMorphemeId) : q.morph) : nullptr;
			tItem->_raw_form = q.typoCost ? kiwi.getTypoForm(q.typoFormId) : tItem->_form;
			tItem->_pairedToken = q.pairedToken;
			if (q.tag == POSTag::sl || q.tag == POSTag::sh || q.tag == POSTag::sw || q.tag == POSTag::w_emoji)
			{
				tItem->_script = q.script;
			}
			else
			{
				tItem->_sense = q.senseId;
			}

			if (!q.typoCost && q.typoFormId && userValues[q.typoFormId - 1])
			{
				tItem->_userValue = move(userValues[q.typoFormId - 1]);
			}
			else
			{
				tItem->_userValue = py::UniqueObj{ userValuesObj ? PyDict_GetItem(userValuesObj.get(), py::buildPyValue(tItem->_morphId).get()) : nullptr };
				if (!tItem->_userValue) tItem->_userValue = py::UniqueObj{ Py_None };
				Py_INCREF(tItem->_userValue.get());
			}

			if (PyDict_Check(tItem->_userValue.get()))
			{
				// tag override
				auto v = PyDict_GetItem(tItem->_userValue.get(), tagAttr);
				if (v)
				{
					tItem->_tag = PyUnicode_AsUTF8(v);
				}
			}

			PyList_SET_ITEM(rList.get(), jdx++, (PyObject*)tItem.release());
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

	using _InitArgs = std::tuple<py::UniqueCObj<KiwiObject>>;

	MorphemeSetObject() = default;

	MorphemeSetObject(py::UniqueCObj<KiwiObject>&& _kiwi)
	{
		kiwi = std::move(_kiwi);
		kiwi->doPrepare();
	}

	void update(PyObject* morphs)
	{
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
	}
};

py::TypeWrapper<MorphemeSetObject> _MorphemeSetSetter{ gModule, [](PyTypeObject& obj)
{
	static PyMethodDef methods[] =
	{
		{ "_update", PY_METHOD(&MorphemeSetObject::update), METH_VARARGS | METH_KEYWORDS, ""},
		{ nullptr }
	};
	obj.tp_methods = methods;
} };

inline SwTokenizerConfig convertToConfig(PyObject* obj)
{
	SwTokenizerConfig cfg;
	cfg.doLowercase = py::getAttr<bool>(obj, "lowercase");
	cfg.splitChinese = py::getAttr<bool>(obj, "split_chinese");
	cfg.wholeWordUnk = py::getAttr<bool>(obj, "whole_word_unk");
	cfg.integrateAllomoprh = py::getAttr<bool>(obj, "integrate_allomorph");
	cfg.splitPunct = py::getAttr<bool>(obj, "split_punct");
	cfg.simpleTag = py::getAttr<bool>(obj, "simple_tag");
	cfg.splitVerb = py::getAttr<bool>(obj, "split_verb");
	cfg.splitEomi = py::getAttr<bool>(obj, "split_eomi");
	cfg.useGlueToken = py::getAttr<bool>(obj, "use_glue_token");
	cfg.newlineToken = py::getAttr<bool>(obj, "use_newline_token");
	cfg.strict = py::getAttr<bool>(obj, "strict");
	cfg.fallbackHangul = py::getAttr<bool>(obj, "fallback_hangul");
	cfg.fallbackByte = py::getAttr<bool>(obj, "fallback_byte");

	py::UniqueObj jsonMod{ PyImport_ImportModule("json") };
	if (!jsonMod) throw py::ExcPropagation{};
	py::UniqueObj additional{
		PyObject_CallFunctionObjArgs(py::getAttr<py::UniqueObj>(jsonMod.get(), "dumps").get(), py::getAttr<py::UniqueObj>(obj, "additional").get(), nullptr)
	};
	if (!additional) throw py::ExcPropagation{};
	cfg.additionalJson = py::toCpp<string>(additional.get());

	static const char* sptoken_names[] = {
		"unk_token", "cls_token", "sep_token", "pad_token", "mask_token", "bos_token", "eos_token",
	};
	for (size_t i = 0; i <= cfg.eos; ++i)
	{
		auto s = py::getAttr<optional<string>>(obj, sptoken_names[i]);
		if (s) cfg.specialTokens[i] = *s;
	}
	return cfg;
}

struct SwTokenizerObject : py::CObject<SwTokenizerObject>
{
	static constexpr const char* _name = "kiwipiepy._SwTokenizer";
	static constexpr const char* _name_in_module = "_SwTokenizer";
	static constexpr int _flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;

	py::UniqueCObj<KiwiObject> kiwi;
	kiwi::SwTokenizer tokenizer;

	using _InitArgs = std::tuple<py::UniqueCObj<KiwiObject>, const char*>;

	SwTokenizerObject() = default;

	SwTokenizerObject(py::UniqueCObj<KiwiObject>&& _kiwi, const char* path)
	{
		kiwi = std::move(_kiwi);
		kiwi->doPrepare();
		std::ifstream ifs;
		tokenizer = kiwi::SwTokenizer::load(kiwi->kiwi, openFile(ifs, path));
	}

	void save(const char* path) const
	{
		std::ofstream ofs;
		tokenizer.save(openFile(ofs, path));
	}

	py::UniqueObj encode(PyObject* text, bool returnOffsets = false) const;

	py::UniqueObj encodeFromMorphs(PyObject* morphs, bool returnOffsets = false) const;

	py::UniqueObj tokenizeAndEncode(PyObject* text, bool returnOffsets = false) const;

	std::string decode(PyObject* ids, bool ignoreErrors = true) const;

	py::UniqueObj config()
	{
		py::UniqueObj ret{ PyDict_New() };
		auto& cfg = tokenizer.getConfig();
		PyDict_SetItemString(ret.get(), "lowercase", py::buildPyValue(cfg.doLowercase).get());
		PyDict_SetItemString(ret.get(), "split_chinese", py::buildPyValue(cfg.splitChinese).get());
		PyDict_SetItemString(ret.get(), "whole_word_unk", py::buildPyValue(cfg.wholeWordUnk).get());
		PyDict_SetItemString(ret.get(), "integrate_allomorph", py::buildPyValue(cfg.integrateAllomoprh).get());
		PyDict_SetItemString(ret.get(), "split_punct", py::buildPyValue(cfg.splitPunct).get());
		PyDict_SetItemString(ret.get(), "simple_tag", py::buildPyValue(cfg.simpleTag).get());
		PyDict_SetItemString(ret.get(), "split_verb", py::buildPyValue(cfg.splitVerb).get());
		PyDict_SetItemString(ret.get(), "split_eomi", py::buildPyValue(cfg.splitEomi).get());
		PyDict_SetItemString(ret.get(), "use_glue_token", py::buildPyValue(cfg.useGlueToken).get());
		PyDict_SetItemString(ret.get(), "strict", py::buildPyValue(cfg.strict).get());
		PyDict_SetItemString(ret.get(), "fallback_hangul", py::buildPyValue(cfg.fallbackHangul).get());
		PyDict_SetItemString(ret.get(), "fallback_byte", py::buildPyValue(cfg.fallbackByte).get());

		py::UniqueObj jsonMod{ PyImport_ImportModule("json") };
		if (!jsonMod) throw py::ExcPropagation{};
		py::UniqueObj additional;
		if (cfg.additionalJson.empty())
		{
			additional = py::buildPyValue(nullptr);
		}
		else
		{
			additional = py::UniqueObj{
				PyObject_CallFunctionObjArgs(py::getAttr<py::UniqueObj>(jsonMod.get(), "loads").get(), py::buildPyValue(cfg.additionalJson).get(), nullptr)
			};
		}
		if (!additional) throw py::ExcPropagation{};
		PyDict_SetItemString(ret.get(), "additional", additional.get());
			
		static const char* sptoken_names[] = {
			"unk_token", "cls_token", "sep_token", "pad_token", "mask_token", "bos_token", "eos_token",
		};
		for (size_t i = 0; i <= cfg.eos; ++i)
		{
			if (cfg.specialTokens[i].empty()) continue;
			PyDict_SetItemString(ret.get(), sptoken_names[i], py::buildPyValue(cfg.specialTokens[i]).get());
		}
		return ret;
	}

	py::UniqueObj vocab()
	{
		py::UniqueObj ret{ PyDict_New() };
			
		for (size_t i = 0; i < tokenizer.size(); ++i)
		{
			auto& v = tokenizer.getVocab(i);
			string form = utf16To8(u16string{ v.form, v.form + v.length });
			if (v.flags == SwTokenFlag::subword)
			{
				form = "##" + form;
			}
			else if (v.pos != POSTag::unknown)
			{
				form += "/";
				form += tagToReprStr(v.pos);
			}
			else if (v.flags == SwTokenFlag::glue)
			{
				form = "##";
			}
			else if (v.flags == SwTokenFlag::byte)
			{
				form = "<0x";
				form += "0123456789ABCDEF"[v.byte >> 4];
				form += "0123456789ABCDEF"[v.byte & 0xF];
				form += ">";
			}
			PyDict_SetItemString(ret.get(), form.c_str(), py::buildPyValue(i).get());
		}
			
		return ret;
	}

	Py_ssize_t len() const
	{
		return tokenizer.size();
	}

	static void train(PyObject* savePath,
		PyObject* texts,
		PyObject* config,
		PyObject* vocabSize,
		size_t iterations, size_t prefixMinCnt, size_t prefixMaxLength,
		bool strictReduction, bool removeRepetitive, bool preventMixedDigitTokens,
		float chrCoverage, float reductionRatio,
		py::UniqueCObj<KiwiObject> kiwi,
		PyObject* callback
	)
	{
		auto cfg = convertToConfig(config);
		auto savePathes = py::toCpp<vector<string>>(savePath);
		auto vocabSizes = py::toCpp<vector<size_t>>(vocabSize);
		if (savePathes.size() != vocabSizes.size())
		{
			throw py::ValueError{ "`save_path` should have the same number of elements to `vocab_size`." };
		}

		UnigramSwTrainerConfig trainCfg;
		trainCfg.chrCoverage = chrCoverage;
		trainCfg.reduceStrict = strictReduction;
		trainCfg.removeRepetitive = removeRepetitive;
		trainCfg.preventMixedDigitTokens = !!preventMixedDigitTokens;
		
		kiwi->doPrepare();
		UnigramSwTrainer trainer{ kiwi->kiwi, cfg, trainCfg };
		py::UniqueObj methodNames[] {
			py::buildPyValue("begin_tokenization"),
			py::buildPyValue("proc_tokenization"),
			py::buildPyValue("end_tokenization"),
			py::buildPyValue("begin_reduction"),
			py::buildPyValue("proc_reduction"),
			py::buildPyValue("end_reduction"),
		};

		py::UniqueObj textsIter{ PyObject_GetIter(texts) };
		if (!textsIter) throw py::ValueError{ "`texts` must be an iterable of `str`." };

		vector<PyObject*> callbackItems;
		py::foreach<PyObject*>(callback, [&](PyObject* item)
		{
			callbackItems.emplace_back(item);
		}, "");

		for (auto ci : callbackItems)
		{
			py::UniqueObj r{ PyObject_CallMethodObjArgs(ci, methodNames[0].get(),
				py::buildPyValue(0).get(),
				nullptr
			) };
			if (!r) throw py::ExcPropagation{};
		}

		size_t sentCnt = 0;
		trainer.addSentences([&]() -> u16string
		{
			while (1)
			{
				py::UniqueObj item{ PyIter_Next(textsIter.get()) };
				if (!item)
				{
					if (PyErr_Occurred()) throw py::ExcPropagation{};
					else return {};
				}
				auto ret = py::toCpp<u16string>(item.get());
				if (++sentCnt % 16 == 0)
				{
					for (auto ci : callbackItems)
					{
						py::UniqueObj r{ PyObject_CallMethodObjArgs(ci, methodNames[1].get(),
							py::buildPyValue(sentCnt).get(),
							nullptr
						) };
						if (!r) throw py::ExcPropagation{};
					}
				}
				if (!ret.empty()) return ret;
			}
			return {};
		});

		for (auto ci : callbackItems)
		{
			py::UniqueObj r{ PyObject_CallMethodObjArgs(ci, methodNames[2].get(),
				py::buildPyValue(sentCnt).get(),
				nullptr
			) };
			if (!r) throw py::ExcPropagation{};
		}

		for (size_t tn = 0; tn < savePathes.size(); ++tn)
		{
			trainer.getTrainConfig().vocabSize = vocabSizes[tn];
			float loss = trainer.buildSubwordVocabs(prefixMinCnt, prefixMaxLength), lastLoss = 0;
			size_t lastVocabSize = 0;

			for (auto ci : callbackItems)
			{
				py::UniqueObj r{ PyObject_CallMethodObjArgs(ci, methodNames[3].get(),
					py::buildPyValue(tn).get(),
					py::buildPyValue(0).get(),
					py::buildPyValue(trainer.getCurrentVocabSize()).get(),
					py::buildPyValue(loss).get(),
					nullptr
				) };
				if (!r) throw py::ExcPropagation{};
			}

			size_t iter = 0;
			for (; iter < iterations; ++iter)
			{
				trainer.updateTokenization();
				trainer.updateProb();
				trainer.reduceVocab(reductionRatio);
				trainer.updateTokenization();
				loss = trainer.updateProb();
				size_t curVocabSize = trainer.getCurrentVocabSize();

				for (auto ci : callbackItems)
				{
					py::UniqueObj r{ PyObject_CallMethodObjArgs(ci, methodNames[4].get(),
						py::buildPyValue(tn).get(),
						py::buildPyValue(iter + 1).get(),
						py::buildPyValue(curVocabSize).get(),
						py::buildPyValue(loss).get(),
						nullptr
					) };
					if (!r) throw py::ExcPropagation{};
				}

				if (curVocabSize <= vocabSizes[tn] || (curVocabSize == lastVocabSize && loss == lastLoss))
				{
					++iter;
					break;
				}
				lastVocabSize = curVocabSize;
				lastLoss = loss;
			}

			for (auto ci : callbackItems)
			{
				py::UniqueObj r{ PyObject_CallMethodObjArgs(ci, methodNames[5].get(),
					py::buildPyValue(tn).get(),
					py::buildPyValue(iter).get(),
					py::buildPyValue(trainer.getCurrentVocabSize()).get(),
					py::buildPyValue(loss).get(),
					nullptr
				) };
				if (!r) throw py::ExcPropagation{};
			}

			auto tokenizer = trainer.build();
			{
				ofstream ofs;
				tokenizer.save(openFile(ofs, savePathes[tn]));
			}
		}
	}
};

py::TypeWrapper<SwTokenizerObject> _SwTokenizerSetter{ gModule, [](PyTypeObject& obj)
{
	static PyMethodDef methods[] =
	{
		{ "encode", PY_METHOD(&SwTokenizerObject::encode), METH_VARARGS | METH_KEYWORDS, ""},
		{ "encode_from_morphs", PY_METHOD(&SwTokenizerObject::encodeFromMorphs), METH_VARARGS | METH_KEYWORDS, ""},
		{ "tokenize_encode", PY_METHOD(&SwTokenizerObject::tokenizeAndEncode), METH_VARARGS | METH_KEYWORDS, ""},
		{ "decode", PY_METHOD(&SwTokenizerObject::decode), METH_VARARGS | METH_KEYWORDS, ""},
		{ "_train", PY_METHOD(&SwTokenizerObject::train), METH_VARARGS | METH_KEYWORDS | METH_STATIC, ""},
		{ "save", PY_METHOD(&SwTokenizerObject::save), METH_VARARGS | METH_KEYWORDS, ""},
		{ nullptr }
	};

	static PyGetSetDef getsets[] =
	{
		{ (char*)"_config", PY_GETTER(&SwTokenizerObject::config), nullptr, "", nullptr},
		{ (char*)"_vocab", PY_GETTER(&SwTokenizerObject::vocab), nullptr, "", nullptr},
		{ (char*)"_kiwi", PY_GETTER(&SwTokenizerObject::kiwi), nullptr, "", nullptr},
		{ nullptr },
	};

	static PySequenceMethods seq = {
		PY_LENFUNC(&SwTokenizerObject::len),
		nullptr,
		nullptr,
		nullptr,
	};

	obj.tp_methods = methods;
	obj.tp_getset = getsets;
	obj.tp_as_sequence = &seq;
} };

inline pair<vector<PretokenizedSpan>, vector<py::UniqueObj>> makePretokenizedSpans(PyObject* obj)
{
	vector<PretokenizedSpan> ret;
	vector<py::UniqueObj> userValues;
	if (obj == Py_None) return make_pair(move(ret), move(userValues));

	vector<size_t> groupBoundaries;
	vector<tuple<PretokenizedSpan*, size_t, py::UniqueObj>> spanPtrs;

	py::foreach<PyObject*>(obj, [&](PyObject* group)
	{
		py::foreachVisit<variant<
			tuple<uint32_t, uint32_t>,
			tuple<uint32_t, uint32_t, PyObject*>,
			tuple<uint32_t, uint32_t, PyObject*, PyObject*>
		>>(group, [&](auto&& item)
		{
			using T = decay_t<decltype(item)>;
			ret.emplace_back(PretokenizedSpan{ get<0>(item), get<1>(item) });
			if constexpr (is_same_v<T, tuple<uint32_t, uint32_t, PyObject*>> || is_same_v<T, tuple<uint32_t, uint32_t, PyObject*, PyObject*>>)
			{
				if (PyUnicode_Check(get<2>(item))) // POSTag
				{
					ret.back().tokenization.emplace_back();
					auto tag = parseTag(py::toCpp<u16string>(get<2>(item)));
					if (tag == POSTag::max) throw py::ValueError{ "wrong tag value: " + py::repr(get<2>(item)) };
					auto& token = ret.back().tokenization.back();
					token.tag = tag;
					token.begin = 0;
					token.end = get<1>(item) - get<0>(item);
				}
				else
				{
					tuple<u16string, u16string, size_t, size_t> singleItem;
					if (py::toCpp<tuple<u16string, u16string, size_t, size_t>>(get<2>(item), singleItem))
					{
						auto tag = parseTag(get<1>(singleItem));
						if (tag == POSTag::max) throw py::ValueError{ "wrong tag value: " + utf16To8(get<1>(singleItem)) };
						ret.back().tokenization.emplace_back();
						auto& token = ret.back().tokenization.back();
						token.form = move(get<0>(singleItem));
						token.tag = tag;
						token.begin = get<2>(singleItem);
						token.end = get<3>(singleItem);
					}
					else
					{
						py::foreach<tuple<u16string, u16string, size_t, size_t>>(get<2>(item), [&](auto&& i)
						{
							auto tag = parseTag(get<1>(i));
							if (tag == POSTag::max) throw py::ValueError{ "wrong tag value: " + utf16To8(get<1>(i)) };
							ret.back().tokenization.emplace_back();
							auto& token = ret.back().tokenization.back();
							token.form = move(get<0>(i));
							token.tag = tag;
							token.begin = get<2>(i);
							token.end = get<3>(i);
						}, "");
					}
				}
			}

			if constexpr (is_same_v<T, tuple<uint32_t, uint32_t, PyObject*, PyObject*>>)
			{
				userValues.emplace_back(py::UniqueObj{ get<3>(item) });
				Py_INCREF(userValues.back().get());
			}
			else
			{
				userValues.emplace_back();
			}
		}, "`pretokenized` must be an iterable of `Tuple[int, int]`, `Tuple[int, int, str]`, `Tuple[int, int, List[Token]]`");
		groupBoundaries.emplace_back(ret.size());
	}, "`pretokenized` must be an iterable of `Tuple[int, int]`, `Tuple[int, int, str]`, `Tuple[int, int, List[Token]]`");

	if (groupBoundaries.size() > 1)
	{
		spanPtrs.reserve(ret.size());
		size_t g = 0;
		for (size_t i = 0; i < ret.size(); ++i)
		{
			while (i >= groupBoundaries[g]) ++g;
			spanPtrs.emplace_back(&ret[i], g, move(userValues[i]));
		}

		sort(spanPtrs.begin(), spanPtrs.end(), [&](auto&& a, auto&& b)
		{
			return get<0>(a)->begin < get<0>(b)->begin;
		});

		size_t target = 0;
		for (size_t cursor = 1; cursor < spanPtrs.size(); ++cursor)
		{
			if (get<0>(spanPtrs[target])->end > get<0>(spanPtrs[cursor])->begin)
			{
				if (get<1>(spanPtrs[target]) == get<1>(spanPtrs[cursor])) throw py::ValueError{ "Overlapped spans in `pretokenized` are not allowed: " + py::repr(obj) };
				
				if (get<1>(spanPtrs[target]) < get<1>(spanPtrs[cursor]))
				{
					spanPtrs[target] = move(spanPtrs[cursor]);
				}
			}
			else
			{
				++target;
				if (target != cursor) spanPtrs[target] = move(spanPtrs[cursor]);
			}
		}
		++target;
		vector<PretokenizedSpan> temp;
		vector<py::UniqueObj> tempUserValues;
		for (size_t i = 0; i < target; ++i)
		{
			temp.emplace_back(move(*get<0>(spanPtrs[i])));
			tempUserValues.emplace_back(move(get<2>(spanPtrs[i])));
		}
		ret.swap(temp);
		userValues.swap(tempUserValues);
	}

	return make_pair(move(ret), move(userValues));
}

inline void updatePretokenizedSpanToU16(vector<PretokenizedSpan>& spans, const py::StringWithOffset<u16string>& so)
{
	for (auto& s : spans)
	{
		for (auto& t : s.tokenization)
		{
			t.begin = so.offsets[s.begin + t.begin] - so.offsets[s.begin];
			t.end = so.offsets[s.begin + t.end] - so.offsets[s.begin];
		}
		s.begin = so.offsets[s.begin];
		s.end = so.offsets[s.end];

		if (s.tokenization.size() == 1 && s.tokenization[0].form.empty())
		{
			s.tokenization[0].form = so.str.substr(s.begin, s.end - s.begin);
		}
	}
}

template<class FutureTy, class CarriedTy>
struct FutureCarrier
{
	std::future<FutureTy> future;
	CarriedTy carried;

	FutureCarrier(std::future<FutureTy>&& _future, const CarriedTy& _carried)
		: future{std::move(_future)}, carried{_carried}
	{}

	FutureCarrier(std::future<FutureTy>&& _future, CarriedTy&& _carried)
		: future{ std::move(_future) }, carried{ std::move(_carried) }
	{}

	FutureCarrier(FutureCarrier&&) = default;
	FutureCarrier& operator=(FutureCarrier&&) = default;

	std::pair<FutureTy, CarriedTy> get()
	{
		return std::make_pair(future.get(), std::move(carried));
	}
};

template<class FutureTy, class CarriedTy>
auto makeFutureCarrier(std::future<FutureTy>&& future, CarriedTy&& carried)
{
	return FutureCarrier<FutureTy, std::remove_reference_t<CarriedTy>>{ std::move(future), std::forward<CarriedTy>(carried) };
}

struct KiwiResIter : public py::ResultIter<KiwiResIter, vector<TokenResult>, FutureCarrier<vector<TokenResult>, vector<py::UniqueObj>>>
{
	static constexpr const char* _name = "kiwipiepy._ResIter";
	static constexpr const char* _name_in_module = "_ResIter";

	py::UniqueCObj<KiwiObject> kiwi;
	py::UniqueCObj<MorphemeSetObject> blocklist;
	py::UniqueObj pretokenizedCallable;
	size_t topN = 1;
	Match matchOptions = Match::all;

	KiwiResIter() = default;
	KiwiResIter(KiwiResIter&&) = default;
	KiwiResIter& operator=(KiwiResIter&&) = default;

	~KiwiResIter()
	{
		waitQueue();
	}

	py::UniqueObj buildPy(pair<vector<TokenResult>, vector<py::UniqueObj>>&& v)
	{
		return py::handleExc([&]()
		{
			if (v.first.size() > topN) v.first.erase(v.first.begin() + topN, v.first.end());
			return resToPyList(move(v.first), kiwi.get(), move(v.second));
		});
	}

	FutureTy feedNext(py::SharedObj&& next)
	{
		if (!PyUnicode_Check(next)) throw py::ValueError{ "`analyze` requires an instance of `str` or an iterable of `str`." };
		
		pair<vector<PretokenizedSpan>, vector<py::UniqueObj>> pretokenized;
		if (pretokenizedCallable)
		{
			py::UniqueObj ptResult{ PyObject_CallFunctionObjArgs(pretokenizedCallable.get(), next.get(), nullptr) };
			pretokenized = makePretokenizedSpans(ptResult.get());
		}
		py::StringWithOffset<u16string> so;
		if (pretokenized.first.empty())
		{
			so.str = py::toCpp<u16string>(next);
		}
		else
		{
			so = py::toCpp<py::StringWithOffset<u16string>>(next);
			updatePretokenizedSpanToU16(pretokenized.first, so);
		}
		return makeFutureCarrier(
			kiwi->kiwi.asyncAnalyze(move(so.str), topN, matchOptions, blocklist ? &blocklist->morphSet : nullptr, move(pretokenized.first)), 
			move(pretokenized.second)
		);
	}
};

py::TypeWrapper<KiwiResIter> _ResIterSetter{ gModule, [](PyTypeObject&)
{
} };

using EncodeResult = pair<vector<uint32_t>, vector<pair<uint32_t, uint32_t>>>;

struct SwTokenizerResIter : public py::ResultIter<SwTokenizerResIter, EncodeResult>
{
	static constexpr const char* _name = "kiwipiepy._SwTokenizerResIter";
	static constexpr const char* _name_in_module = "_SwTokenizerResIter";

	py::UniqueCObj<SwTokenizerObject> tokenizer;
	bool returnOffsets = false;

	SwTokenizerResIter() = default;
	SwTokenizerResIter(SwTokenizerResIter&&) = default;
	SwTokenizerResIter& operator=(SwTokenizerResIter&&) = default;

	~SwTokenizerResIter()
	{
		waitQueue();
	}

	py::UniqueObj buildPy(EncodeResult&& v)
	{
		if (returnOffsets) return py::buildPyTuple(v.first, v.second);
		return py::buildPyValue(v.first);
	}

	future<EncodeResult> feedNext(py::SharedObj&& next)
	{
		if (!PyUnicode_Check(next)) throw py::ValueError{ "`encode` requires an instance of `str` or an iterable of `str`." };
		return tokenizer->tokenizer.asyncEncodeOffset(py::toCpp<string>(next), true);
	}
};

py::TypeWrapper<SwTokenizerResIter> _SwTokenizerResIterSetter{ gModule, [](PyTypeObject&)
{
} };

inline void chrOffsetsToTokenOffsets(const vector<TokenInfo>& tokens, vector<pair<uint32_t, uint32_t>>& offsets)
{
	pair<uint32_t, uint32_t> prev = { 0, 0 };
	for (auto& p : offsets)
	{
		size_t start = upper_bound(tokens.begin(), tokens.end(), p.first, [](uint32_t v, const TokenInfo& t)
		{
			return v < t.position;
		}) - tokens.begin() - 1;

		size_t end = lower_bound(tokens.begin(), tokens.end(), p.second, [](const TokenInfo& t, uint32_t v)
		{
			return t.position + t.length < v;
		}) - tokens.begin() + 1;

		if (start == end)
		{
			if (start > prev.second) start = prev.second;
			else end += 1;
		}

		p.first = start;
		p.second = end;
		prev = p;
	}
}

using TokenEncodeResult = tuple<vector<TokenResult>, vector<uint32_t>, vector<pair<uint32_t, uint32_t>>>;

struct SwTokenizerResTEIter : public py::ResultIter<SwTokenizerResTEIter, TokenEncodeResult>
{
	static constexpr const char* _name = "kiwipiepy._SwTokenizerResTEIter";
	static constexpr const char* _name_in_module = "_SwTokenizerResTEIter";

	py::UniqueCObj<SwTokenizerObject> tokenizer;
	bool returnOffsets = false;

	SwTokenizerResTEIter() = default;
	SwTokenizerResTEIter(SwTokenizerResTEIter&&) = default;
	SwTokenizerResTEIter& operator=(SwTokenizerResTEIter&&) = default;

	~SwTokenizerResTEIter()
	{
		waitQueue();
	}

	py::UniqueObj buildPy(TokenEncodeResult&& v)
	{
		if (returnOffsets) return py::buildPyTuple(resToPyList(move(get<0>(v)), tokenizer->kiwi.get()), get<1>(v), get<2>(v));
		return py::buildPyTuple(resToPyList(move(get<0>(v)), tokenizer->kiwi.get()), get<1>(v));
	}

	future<TokenEncodeResult> feedNext(py::SharedObj&& next)
	{
		if (!PyUnicode_Check(next)) throw py::ValueError{ "`tokenize_encode` requires an instance of `str` or an iterable of `str`." };
		return tokenizer->kiwi->kiwi.getThreadPool()->enqueue([&](size_t, const string& text)
		{
			vector<pair<uint32_t, uint32_t>> offsets;
			auto res = tokenizer->kiwi->kiwi.analyze(text, 1, Match::allWithNormalizing | Match::zCoda);
			auto tokenIds = tokenizer->tokenizer.encode(res[0].first.data(), res[0].first.size(), returnOffsets ? &offsets : nullptr);
			if (returnOffsets) chrOffsetsToTokenOffsets(res[0].first, offsets);
			return make_tuple(move(res), move(tokenIds), move(offsets));
		}, py::toCpp<string>(next));
	}
};

py::TypeWrapper<SwTokenizerResTEIter> _SwTokenizerResTEIterSetter{ gModule, [](PyTypeObject&)
{
} };

py::UniqueObj SwTokenizerObject::encode(PyObject* text, bool returnOffsets) const
{
	if (PyUnicode_Check(text))
	{
		vector<pair<uint32_t, uint32_t>> offsets;
		auto tokenIds = tokenizer.encode(py::toCpp<string>(text), returnOffsets ? &offsets : nullptr, true);
		if (returnOffsets)
		{
			return py::buildPyTuple(tokenIds, offsets);
		}
		else
		{
			return py::buildPyValue(tokenIds);
		}
	}

	py::UniqueObj iter{ PyObject_GetIter(text) };
	if (!iter) throw py::ValueError{ "`encode` requires a `str` or an iterable of `str` parameters." };
	py::UniqueCObj<SwTokenizerResIter> ret{ (SwTokenizerResIter*)PyObject_CallObject((PyObject*)py::Type<SwTokenizerResIter>, nullptr) };
	if (!ret) throw py::ExcPropagation{};
	ret->tokenizer = py::UniqueCObj<SwTokenizerObject>{ const_cast<SwTokenizerObject*>(this) };
	Py_INCREF(this);
	ret->inputIter = move(iter);
	ret->returnOffsets = !!returnOffsets;
		
	for (size_t i = 0; i < kiwi->kiwi.getNumThreads() * 16; ++i)
	{
		if (!ret->feed()) break;
	}
	return ret;
}

py::UniqueObj SwTokenizerObject::encodeFromMorphs(PyObject* morphs, bool returnOffsets) const
{
	py::UniqueObj iter{ PyObject_GetIter(morphs) };
	if (!iter) throw py::ValueError{ "`encodeFromMorphs` requires an iterable of `Tuple[str, str, bool]` parameters." };
	vector<tuple<u16string, POSTag, bool>> tokens;
	py::foreachVisit<variant<
		tuple<string, string, bool>,
		tuple<string, string>
	>>(iter.get(), [&](auto&& item)
	{
		using T = decay_t<decltype(item)>;
		if constexpr (is_same_v<T, tuple<string, string, bool>>)
		{
			auto form = utf8To16(get<0>(item));
			auto pos = parseTag(get<1>(item).c_str());
			auto spaceness = get<2>(item);
			tokens.emplace_back(form, pos, spaceness);
		}
		else if constexpr (is_same_v<T, tuple<string, string>>)
		{
			auto form = utf8To16(get<0>(item));
			auto pos = parseTag(get<1>(item).c_str());
			auto spaceness = false;
			tokens.emplace_back(form, pos, spaceness);
		}
	}, "`encodeFromMorphs` requires an iterable of `Tuple[str, str, bool]` parameters.");
	vector<pair<uint32_t, uint32_t>> offsets;
	auto tokenIds = tokenizer.encode(tokens, returnOffsets ? &offsets : nullptr);
	if (returnOffsets)
	{
		return py::buildPyTuple(tokenIds, offsets);
	}
	else
	{
		return py::buildPyValue(tokenIds);
	}
}

py::UniqueObj SwTokenizerObject::tokenizeAndEncode(PyObject* text, bool returnOffsets) const
{
	if (PyUnicode_Check(text))
	{
		vector<pair<uint32_t, uint32_t>> offsets;
		auto res = tokenizer.getKiwi()->analyze(py::toCpp<string>(text), 1, Match::allWithNormalizing | Match::zCoda);
		auto tokenIds = tokenizer.encode(res[0].first.data(), res[0].first.size(), returnOffsets ? &offsets : nullptr);
		if (returnOffsets)
		{
			chrOffsetsToTokenOffsets(res[0].first, offsets);
			return py::buildPyTuple(resToPyList(move(res), kiwi.get()), tokenIds, offsets);
		}
		else
		{
			return py::buildPyTuple(resToPyList(move(res), kiwi.get()), tokenIds);
		}
	}

	py::UniqueObj iter{ PyObject_GetIter(text) };
	if (!iter) throw py::ValueError{ "`tokenize_encode` requires a `str` or an iterable of `str` parameters." };
	py::UniqueCObj<SwTokenizerResTEIter> ret{ (SwTokenizerResTEIter*)PyObject_CallObject((PyObject*)py::Type<SwTokenizerResTEIter>, nullptr) };
	if (!ret) throw py::ExcPropagation{};
	ret->tokenizer = py::UniqueCObj<SwTokenizerObject>{ const_cast<SwTokenizerObject*>(this) };
	Py_INCREF(this);
	ret->inputIter = move(iter);
	ret->returnOffsets = !!returnOffsets;

	for (size_t i = 0; i < kiwi->kiwi.getNumThreads() * 16; ++i)
	{
		if (!ret->feed()) break;
	}
	return ret;
}

std::string SwTokenizerObject::decode(PyObject* ids, bool ignoreErrors) const
{
	return tokenizer.decode(py::toCpp<vector<uint32_t>>(ids), !!ignoreErrors);
}

std::pair<uint32_t, bool> KiwiObject::addUserWord(const char* word, const char* tag, float score, std::optional<const char*> origWord)
{	
	auto pos = parseTag(tag);
	std::pair<uint32_t, bool> added = std::make_pair(0, false);
	if (origWord)
	{
		added = builder.addWord(utf8To16(word), pos, score, utf8To16(*origWord));
	}
	else
	{
		added = builder.addWord(utf8To16(word), pos, score);
	}
	if (added.second) kiwi = Kiwi{};
	return added;
}

bool KiwiObject::addPreAnalyzedWord(const char* form, PyObject* oAnalyzed, float score)
{
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

	auto added = builder.addPreAnalyzedWord(utf8To16(form), analyzed, positions, score);
	if (added) kiwi = Kiwi{};
	return added;
}

std::vector<std::pair<uint32_t, std::u16string>> KiwiObject::addRule(const char* tag, PyObject* replacer, float score)
{
	if (!PyCallable_Check(replacer)) throw py::ValueError{ "`replacer` must be an callable." };

	auto pos = parseTag(tag);
	auto added = builder.addRule(pos, [&](const u16string& input)
	{
		py::UniqueObj ret{ PyObject_CallFunctionObjArgs(replacer, py::UniqueObj{ py::buildPyValue(input) }.get(), nullptr) };
		if (!ret) throw py::ExcPropagation{};
		return py::toCpp<u16string>(ret.get());
	}, score);
	if (!added.empty()) kiwi = Kiwi{};
	return added;
}

size_t KiwiObject::loadUserDictionary(const char* path)
{
	auto ret = builder.loadDictionary(path);
	if (ret) kiwi = Kiwi{};
	return ret;
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

py::UniqueObj KiwiObject::extractWords(PyObject* sentences, size_t minCnt, size_t maxWordLen, float minScore, float posScore, bool lmFilter) const
{
	auto res = builder.extractWords(obj2reader(sentences), minCnt, maxWordLen, minScore, posScore, lmFilter);

	py::UniqueObj retList{ PyList_New(res.size()) };
	size_t idx = 0;
	for (auto& r : res)
	{
		auto v = py::buildPyTuple(utf16To8(r.form).c_str(), r.score, r.freq, r.posScore[POSTag::nnp]);
		if (!v) throw py::ExcPropagation{};
		PyList_SET_ITEM(retList.get(), idx++, v.release());
	}
	return retList;
}

py::UniqueObj KiwiObject::extractAddWords(PyObject* sentences, size_t minCnt, size_t maxWordLen, float minScore, float posScore, bool lmFilter)
{
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
	return retList;
}

py::UniqueObj KiwiObject::analyze(PyObject* text, size_t topN, Match matchOptions, bool echo, PyObject* blockList, PyObject* pretokenized)
{
	doPrepare();
	if (PyUnicode_Check(text))
	{
		const unordered_set<const Morpheme*>* morphs = nullptr;
		pair<vector<PretokenizedSpan>, vector<py::UniqueObj>> pretokenizedSpans;
		if (blockList != Py_None) morphs = &((MorphemeSetObject*)blockList)->morphSet;
		if (PyCallable_Check(pretokenized))
		{
			py::UniqueObj ptResult{ PyObject_CallFunctionObjArgs(pretokenized, text, nullptr) };
			if (!ptResult) throw py::ExcPropagation{};
			pretokenizedSpans = makePretokenizedSpans(ptResult.get());
		}
		else if (pretokenized != Py_None)
		{
			pretokenizedSpans = makePretokenizedSpans(pretokenized);
		}

		py::StringWithOffset<u16string> so;
		if (pretokenizedSpans.first.empty())
		{
			so.str = py::toCpp<u16string>(text);
		}
		else
		{
			so = py::toCpp<py::StringWithOffset<u16string>>(text);
			updatePretokenizedSpanToU16(pretokenizedSpans.first, so);
		}

		auto res = kiwi.analyze(so.str, topN, matchOptions, morphs, pretokenizedSpans.first);
		if (res.size() > topN) res.erase(res.begin() + topN, res.end());
		return resToPyList(move(res), this, move(pretokenizedSpans.second));
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
		ret->matchOptions = matchOptions;
		ret->echo = !!echo;
		if (blockList != Py_None)
		{
			ret->blocklist = py::UniqueCObj<MorphemeSetObject>{ (MorphemeSetObject*)blockList };
			Py_INCREF(blockList);
		}

		if (PyCallable_Check(pretokenized))
		{
			ret->pretokenizedCallable = py::UniqueObj{ pretokenized };
			Py_INCREF(pretokenized);
		}
		else if (pretokenized != Py_None)
		{
			throw py::ValueError{ "`analyze` of multiple inputs requires a callable `pretokenized` argument." };
		}

		for (size_t i = 0; i < kiwi.getNumThreads() * 16; ++i)
		{
			if (!ret->feed()) break;
		}
		return ret;
	}
}

py::UniqueObj KiwiObject::getMorpheme(size_t id)
{
	auto ret = py::makeNewObject<TokenObject>();
	doPrepare();
	auto* morph = kiwi.idToMorph(id);
	if (!morph) throw py::ValueError{ "out of range" };
	auto joinedForm = joinHangul(morph->getForm());
	ret->_form = move(joinedForm);
	ret->_tag = getTagStr(morph->tag, ret->_form);
	ret->_baseMorph = ret->_morph = morph;
	ret->_morphId = id;
	ret->_regularity = !isIrregular(morph->tag);
	ret->_sense = morph->senseId;
	return ret;
}

py::UniqueObj KiwiObject::join(PyObject* morphs, bool lmSearch, bool returnPositions)
{
	doPrepare();
	auto joiner = kiwi.newJoiner(!!lmSearch);
	size_t prevHash = 0;
	size_t prevEnd = 0;
	py::foreach<PyObject*>(morphs, [&](PyObject* item)
	{
		if (PyObject_IsInstance(item, _TokenSetter.getTypeObj()))
		{
			auto& token = *((TokenObject*)item);
			cmb::Space space = cmb::Space::none;
			if (token.resultHash == prevHash)
			{
				space = token._pos <= prevEnd ? cmb::Space::no_space : cmb::Space::insert_space;
			}

			if (token._morph && token._morph->kform && !token._morph->kform->empty())
			{
				joiner.add(token._morphId, space);
			}
			else
			{
				joiner.add(token._form, token._rawTag, false, space);
			}
			prevHash = token.resultHash;
			prevEnd = token.end();
		}
		else if (PyTuple_Check(item) && PyTuple_Size(item) == 2)
		{
			const char* form = py::toCpp<const char*>(PyTuple_GET_ITEM(item, 0));
			const char* tag = py::toCpp<const char*>(PyTuple_GET_ITEM(item, 1));
			const char* p = strchr(tag, '-');
			joiner.add(utf8To16(form), parseTag(tag), p ? false : true);
			prevHash = 0;
			prevEnd = 0;
		}
		else if (PyTuple_Check(item) && PyTuple_Size(item) == 3)
		{
			const char* form = py::toCpp<const char*>(PyTuple_GET_ITEM(item, 0));
			const char* tag = py::toCpp<const char*>(PyTuple_GET_ITEM(item, 1));
			const char* p = strchr(tag, '-');
			cmb::Space space = PyObject_IsTrue(PyTuple_GET_ITEM(item, 2)) ? cmb::Space::insert_space : cmb::Space::no_space;
			joiner.add(utf8To16(form), parseTag(tag), p ? false : true, space);
			prevHash = 0;
			prevEnd = 0;
		}
		else
		{
			throw py::ConversionFail{ "`morphs` must be an iterable of `Tuple[str, str]`." };
		}
	}, "`morphs` must be an iterable of `Tuple[str, str]`.");
	
	if (returnPositions)
	{
		vector<pair<uint32_t, uint32_t>> positions;
		auto ret = joiner.getU16(&positions);
		// adjust positions for u16 surrogate pairs
		vector<size_t> surrogates(ret.size() + 1);
		size_t acc = 0;
		for (size_t i = 0; i < ret.size(); ++i)
		{
			surrogates[i] = acc;
			acc += ((ret[i] & 0xFC00) == 0xD800 ? 1 : 0);
		}
		surrogates.back() = acc;

		for (auto& p : positions)
		{
			p.first -= surrogates[p.first];
			p.second -= surrogates[p.second];
		}

		return py::buildPyTuple(ret, py::buildPyValue(positions, py::force_list));
	}
	else
	{
		return py::buildPyValue(joiner.getU16());
	}
}

py::UniqueObj KiwiObject::makeHSDataset(PyObject* inputPathes, size_t batchSize, size_t windowSize, size_t numWorkers, 
	float dropout, PyObject* tokenFilter, float splitRatio, bool separateDefaultMorpheme, size_t seed) const
{
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
	auto dataset = builder.makeHSDataset(py::toCpp<vector<string>>(inputPathes), batchSize, windowSize, numWorkers, dropout, tf, splitRatio, separateDefaultMorpheme, &anotherDataset);
	dataset.seed(seed);
	if (splitRatio == 0)
	{
		py::UniqueObj ret{ PyObject_CallObject((PyObject*)py::Type<HSDatasetObject>, nullptr) };
		((HSDatasetObject*)ret.get())->hsd = move(dataset);
		return ret;
	}
	else
	{
		py::UniqueObj ret1{ PyObject_CallObject((PyObject*)py::Type<HSDatasetObject>, nullptr) };
		((HSDatasetObject*)ret1.get())->hsd = move(dataset);
		py::UniqueObj ret2{ PyObject_CallObject((PyObject*)py::Type<HSDatasetObject>, nullptr) };
		((HSDatasetObject*)ret2.get())->hsd = move(anotherDataset);
		auto ret = py::buildPyTuple(ret1, ret2);
		return ret;
	}
}

py::UniqueObj KiwiObject::listAllScripts() const
{
	py::UniqueObj ret{ PyList_New(0) };
	for (int i = 1; i < (int)kiwi::ScriptType::max; ++i)
	{
		auto s = kiwi::getScriptName((kiwi::ScriptType)i);
		PyList_Append(ret.get(), py::buildPyValue(s).get());
	}
	return ret;
}


struct NgramExtractorObject : py::CObject<NgramExtractorObject>
{
	static constexpr const char* _name = "kiwipiepy._NgramExtractor";
	static constexpr const char* _name_in_module = "_NgramExtractor";
	static constexpr int _flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;

	using _InitArgs = std::tuple<PyObject*, bool>;

	NgramExtractor ne;

	NgramExtractorObject() = default;

	NgramExtractorObject(PyObject* kiwi, bool gatherLmScore)
	{
		if (!PyObject_IsInstance(kiwi, (PyObject*)py::Type<KiwiObject>))
		{
			throw py::ValueError{ "`kiwi` must be an instance of `Kiwi`." };
		}
		((KiwiObject*)kiwi)->doPrepare();
		ne = NgramExtractor{ ((KiwiObject*)kiwi)->kiwi, gatherLmScore };
	}

	size_t add(PyObject* texts)
	{
		if (PyUnicode_Check(texts))
		{
			return ne.addText(py::toCpp<u16string>(texts));
		}
		else
		{
			py::UniqueObj iter{ PyObject_GetIter(texts) };
			auto ret = ne.addTexts([&]()
			{
				py::UniqueObj text{ PyIter_Next(iter.get()) };
				if (!text) return u16string{};
				return py::toCpp<u16string>(text.get());
			});

			if (PyErr_Occurred())
			{
				throw py::ExcPropagation{};
			}

			return ret;
		}
	}

	py::UniqueObj extract(PyObject* retTy, size_t maxCandidates, size_t minCnt, size_t maxLength, float minScore, size_t numWorkers)
	{
		auto ret = ne.extract(maxCandidates, minCnt, maxLength, minScore, numWorkers);
		py::UniqueObj retList{ PyList_New(0) };
		for (auto& r : ret)
		{
			py::UniqueObj tokens{ PyList_New(0) };
			for (auto& t : r.tokens)
			{
				auto v = py::buildPyTuple(t.substr(1), t.substr(0, 1));
				PyList_Append(tokens.get(), v.get());
			}
			py::UniqueObj v{ PyObject_CallObject(retTy, 
				py::buildPyTuple(r.text, tokens, r.tokenScores, r.cnt, r.df, r.score, r.npmi, r.leftBranch, r.rightBranch, r.lmScore).get()) 
			};
			PyList_Append(retList.get(), v.get());
		}
		return retList;
	}
};

py::TypeWrapper<NgramExtractorObject> _NgramExtractorSetter{ gModule, [](PyTypeObject& obj)
{
	static PyMethodDef methods[] =
	{
		{ "add", PY_METHOD(&NgramExtractorObject::add), METH_VARARGS | METH_KEYWORDS, ""},
		{ "extract", PY_METHOD(&NgramExtractorObject::extract), METH_VARARGS | METH_KEYWORDS, ""},
		{ nullptr }
	};
	obj.tp_methods = methods;

	static PyGetSetDef getsets[] =
	{
		{ nullptr },
	};
	obj.tp_getset = getsets;
} };


PyMODINIT_FUNC PyInit__kiwipiepy()
{
	import_array();
	py::CustomExcHandler::add<kiwi::IOException, py::OSError>();
	py::CustomExcHandler::add<kiwi::FormatException, py::ValueError>();
	py::CustomExcHandler::add<kiwi::UnicodeException, py::ValueError>();
	py::CustomExcHandler::add<kiwi::UnknownMorphemeException, py::ValueError>();
	py::CustomExcHandler::add<kiwi::SwTokenizerException, py::ValueError>();
	py::CustomExcHandler::add<kiwi::Exception, py::Exception>();
	return gModule.init();
}
