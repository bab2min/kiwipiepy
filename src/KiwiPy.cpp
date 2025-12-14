#define _SILENCE_CXX17_RESULT_OF_DEPRECATION_WARNING

#include <stdexcept>
#include <fstream>
#include <algorithm>
#include <mutex>
#include <shared_mutex>

#define USE_NUMPY
#define MAIN_MODULE

#include "PyUtils.h"

#include <kiwi/Kiwi.h>
#include <kiwi/Dataset.h>
#include <kiwi/SwTokenizer.h>
#include <kiwi/SubstringExtractor.h>

using namespace std;
using namespace kiwi;

static py::Module gModule{ "_kiwipiepy", "Kiwi API for Python" };

vector<pair<u16string, size_t>> pyExtractSubstrings(const u16string& str, size_t minCnt, size_t minLength, size_t maxLength, bool longestOnly, const u16string& stopChr)
{
	if (stopChr.size() > 1)
	{
		throw py::ValueError{ "stopChr must be a single character." };
	}

	return extractSubstrings(str.data(), str.data() + str.size(), minCnt, minLength, maxLength, longestOnly, stopChr.empty() ? 0 : stopChr[0]);
}

struct TypoTransformerObject : py::CObject<TypoTransformerObject>
{
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
			auto orig = py::toCpp<std::vector<std::string>>(PyTuple_GetItem(item, 0));
			auto error = py::toCpp<std::vector<std::string>>(PyTuple_GetItem(item, 1));
			auto cost = py::toCpp<float>(PyTuple_GetItem(item, 2));
			PyObject* cond = PyTuple_GetItem(item, 3);
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
		vector<pair<tuple<KString, KString, CondVowel, Dialect>, float>> defs{ tt.getTypos().begin(), tt.getTypos().end() };
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
		if (!prepared)
		{
#ifdef Py_GIL_DISABLED
			Py_BEGIN_CRITICAL_SECTION(this);
#endif
			ptt = tt.prepare();
			prepared = true;
#ifdef Py_GIL_DISABLED
			Py_END_CRITICAL_SECTION();
#endif
		}
		return ptt;
	}

	py::UniqueObj generate(const string& orig, float costThreshold = 2.5)
	{
		py::UniqueObj ret{ PyList_New(0) };
		for (auto r : getPtt().generate(utf8To16(orig), costThreshold))
		{
			PyList_Append(ret.get(), py::buildPyTuple(r.str, r.cost).get());
		}
		return ret;
	}
};

struct HSDatasetIterObject;

struct HSDatasetObject : py::CObject<HSDatasetObject>
{
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

	const kiwi::Vector<uint8_t>& getWindowTokenValidness() const
	{
		return hsd.getWindowTokenValidness();
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

	std::vector<std::pair<std::vector<uint32_t>, size_t>> extractPrefixes(size_t minCnt, size_t maxLength, size_t numWorkers = 1, bool exclusiveCnt = false) const
	{
		return hsd.extractPrefixes(minCnt, maxLength, numWorkers, exclusiveCnt);
	}
};

struct HSDatasetIterObject : py::CObject<HSDatasetIterObject>
{
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
		const size_t causalContextSize = obj->hsd.getCausalContextSize();
		const size_t windowSize = obj->hsd.getWindowSize();
		const size_t bs = batchSize * 4, ws = causalContextSize + windowSize;
		int64_t* inDataPtr = nullptr;
		int64_t* outDataPtr = nullptr;
		float* lmLProbsPtr = nullptr;
		int64_t* outNgramNodePtr = nullptr;
		int64_t* ulInDataPtr = nullptr;
		int64_t* ulOutDataPtr = nullptr;
		py::UniqueObj inData = py::newEmptyArray(inDataPtr, bs, ws);
		py::UniqueObj outData = py::newEmptyArray(outDataPtr, bs);
		py::UniqueObj lmLProbsData = py::newEmptyArray(lmLProbsPtr, bs);
		py::UniqueObj outNgramNodeData = py::newEmptyArray(outNgramNodePtr, bs);
		py::UniqueObj ulInData;
		py::UniqueObj ulOutData;

		if (obj->hsd.doesGenerateUnlikelihoods())
		{
			ulInData = py::newEmptyArray(ulInDataPtr, bs, ws);
			ulOutData = py::newEmptyArray(ulOutDataPtr, bs);
		}

		float restLm = 0;
		uint32_t restLmCnt = 0;
		size_t ulDataSize = 0;

		const size_t sz = obj->hsd.next(
			inDataPtr,
			outDataPtr,
			lmLProbsPtr,
			outNgramNodePtr,
			restLm,
			restLmCnt,
			ulInDataPtr,
			ulOutDataPtr,
			&ulDataSize
		);
		if (!sz) throw py::ExcPropagation{};

		//if (sz < batchSize)
		{
			py::UniqueObj slice{ PySlice_New(nullptr, py::buildPyValue(sz).get(), nullptr)};
			inData = py::UniqueObj{ PyObject_GetItem(inData.get(), slice.get())};
			outData = py::UniqueObj{ PyObject_GetItem(outData.get(), slice.get())};
			lmLProbsData = py::UniqueObj{ PyObject_GetItem(lmLProbsData.get(), slice.get())};
			outNgramNodeData = py::UniqueObj{ PyObject_GetItem(outNgramNodeData.get(), slice.get())};
			if (ulInData)
			{
				py::UniqueObj slice{ PySlice_New(nullptr, py::buildPyValue(ulDataSize).get(), nullptr) };
				ulInData = py::UniqueObj{ PyObject_GetItem(ulInData.get(), slice.get()) };
				ulOutData = py::UniqueObj{ PyObject_GetItem(ulOutData.get(), slice.get()) };
			}
		}
		if (ulInData)
		{
			return py::buildPyTuple(inData, outData, lmLProbsData, outNgramNodeData, restLm, restLmCnt, ulInData, ulOutData);
		}
		else
		{
			return py::buildPyTuple(inData, outData, lmLProbsData, outNgramNodeData, restLm, restLmCnt);
		}
	}
};

struct KNLangModelObject;

struct KNLangModelNextTokensResultObject : py::CObject<KNLangModelNextTokensResultObject>
{
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

struct KNLangModelEvaluateResultObject : py::CObject<KNLangModelEvaluateResultObject>
{
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

struct KNLangModelObject : py::CObject<KNLangModelObject>
{
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
			const int dtype = py::getArrayDtype(item);
			if (dtype < 0) throw py::ValueError{ "arrays must be a list of numpy arrays." };
			const size_t dims = py::getArrayNdim(item);
			if (dims != 1) throw py::ValueError{ "arrays must be a list of 1D numpy arrays." };
			const size_t len = py::getArraySize(item);
			if (dtype == py::NPY_UINT16 || dtype == py::NPY_INT16)
			{
				auto* ptr = (const uint16_t*)py::getArrayDataPtr(item);
				pfCnt.addArray(ptr, ptr + len);
			}
			else if (dtype == py::NPY_UINT32 || dtype == py::NPY_INT32)
			{
				auto* ptr = (const uint32_t*)py::getArrayDataPtr(item);
				pfCnt.addArray(ptr, ptr + len);
			}
			else if (dtype == py::NPY_UINT64 || dtype == py::NPY_INT64)
			{
				auto* ptr = (const uint64_t*)py::getArrayDataPtr(item);
				pfCnt.addArray(ptr, ptr + len);
			}
			else
			{
				throw py::ValueError{ "arrays must be a list of numpy arrays of uint16, uint32 or uint64." };
			}
		}, "arrays must be a list of numpy arrays.");

		auto lm = pfCnt.buildLM(minCf, bosTokenId, eosTokenId, unkTokenId, ArchType::balanced);

		auto* clsType = (PyTypeObject*)cls.get();
		py::UniqueCObj<KNLangModelObject> ret{ PyObject_New(KNLangModelObject, clsType) };
		new (ret.get()) KNLangModelObject{ };
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

	static py::UniqueObj load(py::UniqueObj cls, const string& path, size_t numWorkers)
	{
		auto lm = lm::KnLangModelBase::create(utils::MMap(path), ArchType::balanced);

		auto* clsType = (PyTypeObject*)cls.get();
		py::UniqueCObj<KNLangModelObject> ret{ PyObject_New(KNLangModelObject, clsType) };
		new (ret.get()) KNLangModelObject{ };
		ret->langModel = std::move(lm);
		ret->initClusterData();
		if (numWorkers >= 1)
		{
			ret->workers = std::make_unique<utils::ThreadPool>(numWorkers);
		}
		return ret;
	}

	void save(const string& path) const
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
		const int dtype = py::getArrayDtype(obj.get());
		if (dtype < 0) throw py::ValueError{ "obj must be a numpy array." };
		const size_t dims = py::getArrayNdim(obj.get());
		if (dims != 1) throw py::ValueError{ "obj must be a 1D numpy array." };
		const size_t len = py::getArraySize(obj.get());
		const void* inData = py::getArrayDataPtr(obj.get());

		uint32_t* idxData = nullptr;
		float* llData = nullptr;
		py::UniqueObj outIdx = py::newEmptyArray(idxData, len, topN);
		py::UniqueObj outLl = py::newEmptyArray(llData, len, topN);

		if (deferred)
		{
			auto ret = py::makeNewObject<KNLangModelNextTokensResultObject>();
			ret->inArray = move(obj);
			ret->outIdx = move(outIdx);
			ret->outLl = move(outLl);
			Py_INCREF(this);
			ret->parent = py::UniqueCObj<KNLangModelObject>{ (KNLangModelObject*)this };
			if (dtype == py::NPY_UINT16 || dtype == py::NPY_INT16)
			{
				ret->future = workers->enqueue([=](size_t threadIdx)
				{
					auto* ptr = (const uint16_t*)inData;
					langModel->predictTopN(ptr, ptr + len, topN, idxData, llData);
				});
			}
			else if (dtype == py::NPY_UINT32 || dtype == py::NPY_INT32)
			{
				ret->future = workers->enqueue([=](size_t threadIdx)
				{
					auto* ptr = (const uint32_t*)inData;
					langModel->predictTopN(ptr, ptr + len, topN, idxData, llData);
				});
			}
			else if (dtype == py::NPY_UINT64 || dtype == py::NPY_INT64)
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
			if (dtype == py::NPY_UINT16 || dtype == py::NPY_INT16)
			{
				auto* ptr = (const uint16_t*)inData;
				langModel->predictTopN(ptr, ptr + len, topN, idxData, llData);
			}
			else if (dtype == py::NPY_UINT32 || dtype == py::NPY_INT32)
			{
				auto* ptr = (const uint32_t*)inData;
				langModel->predictTopN(ptr, ptr + len, topN, idxData, llData);
			}
			else if (dtype == py::NPY_UINT64 || dtype == py::NPY_INT64)
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
		const int dtype = py::getArrayDtype(obj.get());
		if (dtype < 0) throw py::ValueError{ "obj must be a numpy array." };
		const size_t dims = py::getArrayNdim(obj.get());
		if (dims != 1) throw py::ValueError{ "obj must be a 1D numpy array." };
		const size_t len = py::getArraySize(obj.get());
		const void* inData = py::getArrayDataPtr(obj.get());

		float* llData = nullptr;
		py::UniqueObj outLl = py::newEmptyArray(llData, len);
		
		if (deferred)
		{
			auto ret = py::makeNewObject<KNLangModelEvaluateResultObject>();
			ret->inArray = move(obj);
			ret->outLl = move(outLl);
			Py_INCREF(this);
			ret->parent = py::UniqueCObj<KNLangModelObject>{ (KNLangModelObject*)this };
			if (dtype == py::NPY_UINT16 || dtype == py::NPY_INT16)
			{
				ret->future = workers->enqueue([=](size_t threadIdx)
				{
					auto* ptr = (const uint16_t*)inData;
					evaluateWithCluster(ptr, len, llData);
				});
			}
			else if (dtype == py::NPY_UINT32 || dtype == py::NPY_INT32)
			{
				ret->future = workers->enqueue([=](size_t threadIdx)
				{
					auto* ptr = (const uint32_t*)inData;
					evaluateWithCluster(ptr, len, llData);
				});
			}
			else if (dtype == py::NPY_UINT64 || dtype == py::NPY_INT64)
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
			if (dtype == py::NPY_UINT16 || dtype == py::NPY_INT16)
			{
				auto* ptr = (const uint16_t*)inData;
				evaluateWithCluster(ptr, len, llData);
			}
			else if (dtype == py::NPY_UINT32 || dtype == py::NPY_INT32)
			{
				auto* ptr = (const uint32_t*)inData;
				evaluateWithCluster(ptr, len, llData);
			}
			else if (dtype == py::NPY_UINT64 || dtype == py::NPY_INT64)
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

struct ContextSpan
{
	const uint32_t* data = nullptr;
	size_t size = 0;

	ContextSpan(const uint32_t* _data = nullptr, size_t _size = 0) : data(_data), size(_size) {}
};

template<class Ty>
void setValueFromAttr(Ty& val, PyObject* obj, const char* attr)
{
	py::UniqueObj ret{ PyObject_GetAttrString(obj, attr) };
	if (!ret) throw py::ExcPropagation{};
	py::toCpp<Ty>(ret.get(), val);
}

KiwiConfig toKiwiConfig(PyObject* obj)
{
	KiwiConfig c;
	setValueFromAttr(c.integrateAllomorph, obj, "integrate_allomorph");
	setValueFromAttr(c.cutOffThreshold, obj, "cutoff_threshold");
	setValueFromAttr(c.unkFormScoreScale, obj, "unk_form_score_scale");
	setValueFromAttr(c.unkFormScoreBias, obj, "unk_form_score_bias");
	setValueFromAttr(c.spacePenalty, obj, "space_penalty");
	setValueFromAttr(c.typoCostWeight, obj, "typo_cost_weight");
	setValueFromAttr(c.maxUnkFormSize, obj, "max_unk_form_size");
	setValueFromAttr(c.spaceTolerance, obj, "space_tolerance");
	return c;
}

struct KiwiObject : py::CObject<KiwiObject>
{
	KiwiBuilder builder;
	mutable std::shared_ptr<Kiwi> kiwi;
	TypoTransformerObject* typos = nullptr;
	float typoCostThreshold = 2.5f;
	Vector<vector<u16string>> contextForms;
	Vector<pair<Vector<uint32_t>, Vector<size_t>>> contextAnalyses;

#ifdef Py_GIL_DISABLED
	std::unique_ptr<std::shared_mutex> rwMutex;
#endif

	using _InitArgs = std::tuple<
		size_t,
		std::optional<string>,
		bool,
		bool,
		bool,
		bool,
		std::string,
		PyObject*,
		float,
		Dialect
	>;

	KiwiObject() = default;

	KiwiObject(size_t numThreads, 
		const std::optional<string>& modelPath = {}, 
		bool integrateAllomorph = true, 
		bool loadDefaultDict = true, 
		bool loadTypoDict = true, 
		bool loadMultiDict = true,
		const std::string& modelType = {},
		PyObject* _typos = nullptr, 
		float _typoCostThreshold = 2.5f,
		Dialect enabledDialects = Dialect::standard
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

		ModelType mtype = ModelType::none;
		if (modelType.empty() || modelType == "none")
		{
			mtype = ModelType::none;
		}
		else if (modelType == "largest")
		{
			mtype = ModelType::largest;
		}
		else if (modelType == "knlm")
		{
			mtype = ModelType::knlm;
		}
		else if (modelType == "sbg")
		{
			mtype = ModelType::sbg;
		}
		else if (modelType == "cong")
		{
			mtype = ModelType::cong;
		}
		else if (modelType == "cong-global")
		{
			mtype = ModelType::congGlobal;
		}
		else
		{
			throw py::ValueError{ "invalid model type: " + modelType };
		}

		builder = KiwiBuilder{ spath, numThreads, (BuildOption)boptions, mtype, enabledDialects };
#ifdef Py_GIL_DISABLED
		rwMutex = std::make_unique<std::shared_mutex>();
#endif
	}

	std::shared_ptr<Kiwi> doPrepare() const
	{
		if (auto k = kiwi) return k;
#ifdef Py_GIL_DISABLED
		std::unique_lock lock{ *rwMutex };
		if (kiwi) return kiwi;
#endif
		kiwi = std::make_shared<Kiwi>(builder.build(typos ? typos->tt : getDefaultTypoSet(DefaultTypoSet::withoutTypo), typoCostThreshold));
		return kiwi;
	}

	void convertContextToReadableForm(Kiwi* kiwi, const vector<uint32_t>& context, vector<u16string>& forms, pair<Vector<uint32_t>, Vector<size_t>>& analyses) const
	{
		Vector<ContextSpan> spans;
		const uint32_t delimiter = -1;
		size_t start = 0;
		for (size_t i = 0; i < context.size(); ++i)
		{
			if (context[i] == delimiter)
			{
				spans.emplace_back(context.data() + start, i - start);
				start = i + 1;
			}
		}
		if (start < context.size())
		{
			spans.emplace_back(context.data() + start, context.size() - start);
		}

		sort(spans.begin(), spans.end(), [](const ContextSpan& a, const ContextSpan& b)
		{
			if (a.size < b.size) return true;
			if (a.size > b.size) return false;
			for (size_t i = 0; i < a.size; ++i)
			{
				if (a.data[i] < b.data[i]) return true;
				if (a.data[i] > b.data[i]) return false;
			}
			return false;
		});

		for (auto& span : spans)
		{
			auto joiner = kiwi->newJoiner(false);
			for (size_t i = 0; i < span.size; ++i)
			{
				joiner.add(span.data[i]);
				analyses.first.emplace_back(span.data[i]);
			}
			forms.emplace_back(joiner.getU16());
			analyses.second.push_back(analyses.first.size());
		}
	}

	void prepareContextMap(Kiwi* kiwi, const lm::CoNgramModelBase* cong)
	{
		if (!contextForms.empty()) return;
#ifdef Py_GIL_DISABLED
		std::unique_lock lock{ *rwMutex };
		if (!contextForms.empty()) return;
#endif

		auto contextMap = cong->getContextWordMap();
		for (size_t i = 0; i < contextMap.size(); ++i)
		{
			vector<u16string> forms;
			pair<Vector<uint32_t>, Vector<size_t>> analyses;
			convertContextToReadableForm(kiwi, contextMap[i], forms, analyses);
			contextForms.emplace_back(std::move(forms));
			contextAnalyses.emplace_back(std::move(analyses));
		}
	}

	std::pair<uint32_t, bool> addUserWord(const string& word, const string& tag = "NNP", float score = 0, const std::optional<string>& origWord = {});
	bool addPreAnalyzedWord(const string& form, PyObject* oAnalyzed = nullptr, float score = 0, Dialect dialect = Dialect::standard);
	std::vector<std::pair<uint32_t, std::u16string>> addRule(const string& tag, PyObject* replacer, float score = 0);
	py::UniqueObj analyze(PyObject* text, size_t topN = 1, 
		Match matchOptions = Match::all, 
		bool echo = false, 
		PyObject* blockList = Py_None, 
		bool openEnding = false, 
		Dialect allowedDialects = Dialect::standard,
		float dialectCost = 3.f,
		PyObject* pretokenized = Py_None,
		PyObject* config = Py_None);
	py::UniqueObj extractAddWords(PyObject* sentences, size_t minCnt = 10, size_t maxWordLen = 10, float minScore = 0.25f, float posScore = -3, bool lmFilter = true);
	py::UniqueObj extractWords(PyObject* sentences, size_t minCnt, size_t maxWordLen = 10, float minScore = 0.25f, float posScore = -3, bool lmFilter = true) const;
	size_t loadUserDictionary(const string& path);
	py::UniqueObj getMorpheme(size_t id);
	py::UniqueObj join(PyObject* morphs, bool lmSearch = true, bool returnPositions = false);
	py::UniqueObj mostSimilarMorphemes(PyObject* retTy, PyObject* target, size_t topN);
	py::UniqueObj mostSimilarContexts(PyObject* retTy, PyObject* target, PyObject* contextId, size_t topN);
	py::UniqueObj predictNextMorpheme(PyObject* retTy, PyObject* prefix, PyObject* bgPrefix, float bgWeight, size_t topN);
	float morphemeSimilarity(PyObject* a, PyObject* b);
	float contextSimilarity(PyObject* a, PyObject* b);
	
	void convertHSData(
		PyObject* inputPathes, 
		const string& outputPath,
		PyObject* morphemeDefPath = nullptr,
		size_t morphemeDefMinCnt = 0,
		bool generateOovDict = false,
		PyObject* transform = nullptr) const;

	py::UniqueObj makeHSDataset(PyObject* inputPathes, 
		size_t batchSize, 
		size_t causalContextSize, 
		size_t windowSize, 
		size_t numWorkers, 
		float dropout = 0, 
		float dropoutOnHistory = 0,
		float nounAugmentingProb = 0,
		float emojiAugmentingProb = 0,
		float sbAugmentingProb = 0,
		size_t generateUnlikelihoods = -1,
		PyObject* tokenFilter = nullptr, 
		PyObject* windowFilter = nullptr, 
		float splitRatio = 0, 
		bool separateDefaultMorpheme = false, 
		PyObject* morphemeDefPath = nullptr,
		size_t morphemeDefMinCnt = 0,
		const std::vector<std::pair<size_t, std::vector<uint32_t>>>& contextualMapper = {},
		PyObject* transform = nullptr,
		size_t seed = 42) const;

	py::UniqueObj listAllScripts() const;

	py::UniqueObj getGlobalConfig() const
	{
		auto kiwiInst = doPrepare();
		auto config = kiwiInst->getGlobalConfig();
		static const char* keys[] = {
			"integrate_allomorph",
			"cutoff_threshold",
			"unk_form_score_scale",
			"unk_form_score_bias",
			"space_penalty",
			"typo_cost_weight",
			"max_unk_form_size",
			"space_tolerance",
		};

		return py::buildPyDict(
			keys,
			config.integrateAllomorph,
			config.cutOffThreshold,
			config.unkFormScoreScale,
			config.unkFormScoreBias,
			config.spacePenalty,
			config.typoCostWeight,
			config.maxUnkFormSize,
			config.spaceTolerance
		);
	}

	void setGlobalConfig(PyObject* config)
	{
		auto kiwiInst = doPrepare();
		kiwiInst->setGlobalConfig(toKiwiConfig(config));
	}

	size_t getNumWorkers() const
	{
		auto kiwiInst = doPrepare();
		return kiwiInst->getNumThreads();
	}

	const char* getModelType()
	{
		return modelTypeToStr(builder.getModelType());
	}
};

struct TokenObject : py::CObject<TokenObject>
{
	std::weak_ptr<Kiwi> kiwiInst;
	u16string _form, _raw_form;
	string _tag;
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
	uint16_t _dialect = 0;
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

py::UniqueObj resToPyList(vector<TokenResult>&& res, const KiwiObject* kiwiObj, const shared_ptr<Kiwi>& kiwiInst, vector<py::UniqueObj>&& userValues = {})
{
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
		const size_t resultHash = hashTokenInfo(p.first);
		for (auto& q : p.first)
		{
			size_t u32chrs = 0;
			for (auto u : q.str)
			{
				if ((u & 0xFC00) == 0xD800) u32chrs++;
			}

			auto tItem = py::makeNewObject<TokenObject>();
			tItem->kiwiInst = kiwiInst;
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
			tItem->_morphId = q.morph ? kiwiInst->morphToId(q.morph) : -1;
			tItem->_baseMorph = (q.morph && !!q.dialect) ? kiwiInst->idToMorph(q.morph->lmMorphemeId) : 
				(q.morph ? (q.morph->origMorphemeId ? kiwiInst->idToMorph(q.morph->origMorphemeId) : q.morph) : nullptr);
			tItem->_raw_form = q.typoCost ? kiwiInst->getTypoForm(q.typoFormId) : tItem->_form;
			tItem->_pairedToken = q.pairedToken;
			tItem->_dialect = (uint16_t)q.dialect;
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
					tItem->_tag = py::toCpp<string>(v);
				}
			}

			PyList_SetItem(rList.get(), jdx++, (PyObject*)tItem.release());
			u32offset += u32chrs;
		}
		PyList_SetItem(retList.get(), idx++, py::buildPyTuple(move(rList), p.second).release());
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
	py::UniqueCObj<KiwiObject> kiwi;
	std::vector<std::tuple<string, POSTag, uint8_t>> morphList;
	mutable std::weak_ptr<Kiwi> kiwiPtr;
	mutable std::unordered_set<const kiwi::Morpheme*> morphSet;

	using _InitArgs = std::tuple<py::UniqueCObj<KiwiObject>>;

	MorphemeSetObject() = default;

	MorphemeSetObject(py::UniqueCObj<KiwiObject>&& _kiwi)
	{
		kiwi = std::move(_kiwi);
	}

	void update(PyObject* morphs)
	{
		morphList.clear();
		morphSet.clear();

		py::foreach<PyObject*>(morphs, [&](PyObject* item)
		{
			if (PyTuple_Check(item) && (PyTuple_Size(item) == 2 || PyTuple_Size(item) == 3))
			{
				auto form = py::toCpp<string>(PyTuple_GetItem(item, 0));
				auto stag = py::toCpp<string>(PyTuple_GetItem(item, 1));
				uint8_t senseId = undefSenseId;
				if (PyTuple_Size(item) == 3)
				{
					senseId = (uint8_t)py::toCpp<size_t>(PyTuple_GetItem(item, 2));
				}
				POSTag tag = POSTag::unknown;
				if (!stag.empty())
				{
					tag = parseTag(stag.c_str());
				}
				morphList.emplace_back(form, tag, senseId);
			}
			else
			{
				throw py::ForeachFailed{};
			}
		}, "`morphs` must be an iterable of `tuple`.");
	}

	const std::unordered_set<const kiwi::Morpheme*>& getMorphemeSet() const
	{
		auto kiwiInst = kiwiPtr.lock();
		if (!kiwiInst)
		{
			morphSet.clear();
			kiwiPtr = kiwiInst = kiwi->doPrepare();
		}
		if (morphSet.empty())
		{
			for (auto& p : morphList)
			{
				auto form = utf8To16(std::get<0>(p));
				auto tag = std::get<1>(p);
				auto senseId = std::get<2>(p);
				auto morphs = kiwiInst->findMorphemes(form, tag, senseId);
				for (auto m : morphs)
				{
					morphSet.insert(m);
				}
			}
		}
		return morphSet;
	}
};

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
	py::UniqueCObj<KiwiObject> kiwi;
	std::shared_ptr<Kiwi> kiwiInst;
	kiwi::SwTokenizer tokenizer;

	using _InitArgs = std::tuple<py::UniqueCObj<KiwiObject>, string>;

	SwTokenizerObject() = default;

	SwTokenizerObject(py::UniqueCObj<KiwiObject>&& _kiwi, const string& path)
	{
		kiwi = std::move(_kiwi);
		kiwiInst = kiwi->doPrepare();
		std::ifstream ifs;
		tokenizer = kiwi::SwTokenizer::load(*kiwiInst, openFile(ifs, path));
	}

	void save(const string& path) const
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
		
		auto kiwiInst = kiwi->doPrepare();
		UnigramSwTrainer trainer{ *kiwiInst, cfg, trainCfg };
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
	py::UniqueCObj<KiwiObject> kiwi;
	std::shared_ptr<Kiwi> kiwiInst;
	py::UniqueCObj<MorphemeSetObject> blocklist;
	py::UniqueObj pretokenizedCallable;
#ifdef Py_GIL_DISABLED
	std::shared_lock<std::shared_mutex> lock;
#endif
	size_t topN = 1;
	AnalyzeOption options;
	KiwiConfig config;

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
			return resToPyList(move(v.first), kiwi.get(), kiwiInst, move(v.second));
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
			kiwiInst->asyncAnalyze(move(so.str), topN, 
				options,
				move(pretokenized.first),
				config
			),
			move(pretokenized.second)
		);
	}
};

using EncodeResult = pair<vector<uint32_t>, vector<pair<uint32_t, uint32_t>>>;

struct SwTokenizerResIter : public py::ResultIter<SwTokenizerResIter, EncodeResult>
{
	py::UniqueCObj<SwTokenizerObject> tokenizer;
	bool returnOffsets = false;
#ifdef Py_GIL_DISABLED
	std::shared_lock<std::shared_mutex> lock;
#endif

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
		if (returnOffsets) return py::buildPyTuple(resToPyList(move(get<0>(v)), tokenizer->kiwi.get(), tokenizer->kiwiInst), get<1>(v), get<2>(v));
		return py::buildPyTuple(resToPyList(move(get<0>(v)), tokenizer->kiwi.get(), tokenizer->kiwiInst), get<1>(v));
	}

	future<TokenEncodeResult> feedNext(py::SharedObj&& next)
	{
		if (!PyUnicode_Check(next)) throw py::ValueError{ "`tokenize_encode` requires an instance of `str` or an iterable of `str`." };
		auto* pool = tokenizer->kiwiInst->getThreadPool();
		if (!pool) throw py::RuntimeError{ "async mode is unavailable in num_workers == 0" };
		return pool->enqueue([&](size_t, const string& text)
		{
			vector<pair<uint32_t, uint32_t>> offsets;
			auto res = tokenizer->kiwiInst->analyze(text, 1, Match::allWithNormalizing | Match::zCoda);
			auto tokenIds = tokenizer->tokenizer.encode(res[0].first.data(), res[0].first.size(), returnOffsets ? &offsets : nullptr);
			if (returnOffsets) chrOffsetsToTokenOffsets(res[0].first, offsets);
			return make_tuple(move(res), move(tokenIds), move(offsets));
		}, py::toCpp<string>(next));
	}
};

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
		
	for (size_t i = 0; i < kiwiInst->getNumThreads() * 16; ++i)
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
#ifdef Py_GIL_DISABLED
	std::shared_lock lock{ *kiwi->rwMutex };
#endif
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
		auto res = kiwiInst->analyze(py::toCpp<string>(text), 1, Match::allWithNormalizing | Match::zCoda);
		auto tokenIds = tokenizer.encode(res[0].first.data(), res[0].first.size(), returnOffsets ? &offsets : nullptr);
		if (returnOffsets)
		{
			chrOffsetsToTokenOffsets(res[0].first, offsets);
			return py::buildPyTuple(resToPyList(move(res), kiwi.get(), kiwiInst), tokenIds, offsets);
		}
		else
		{
			return py::buildPyTuple(resToPyList(move(res), kiwi.get(), kiwiInst), tokenIds);
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

	for (size_t i = 0; i < kiwiInst->getNumThreads() * 16; ++i)
	{
		if (!ret->feed()) break;
	}
	return ret;
}

std::string SwTokenizerObject::decode(PyObject* ids, bool ignoreErrors) const
{
	return tokenizer.decode(py::toCpp<vector<uint32_t>>(ids), !!ignoreErrors);
}

std::pair<uint32_t, bool> KiwiObject::addUserWord(const string& word, const string& tag, float score, const std::optional<string>& origWord)
{	
#ifdef Py_GIL_DISABLED
	std::unique_lock lock{ *rwMutex };
#endif
	auto pos = parseTag(tag.c_str());
	std::pair<uint32_t, bool> added = std::make_pair(0, false);
	if (origWord)
	{
		added = builder.addWord(utf8To16(word), pos, score, utf8To16(*origWord));
	}
	else
	{
		added = builder.addWord(utf8To16(word), pos, score);
	}
	if (added.second) kiwi.reset();
	return added;
}

bool KiwiObject::addPreAnalyzedWord(const string& form, PyObject* oAnalyzed, float score, Dialect dialect)
{
	vector<tuple<u16string, POSTag, uint8_t>> analyzed;
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
			analyzed.emplace_back(str.substr(0, p), parseTag(str.substr(p + 1)), undefSenseId);
		}
		else if (PySequence_Check(item))
		{
			if (Py_SIZE(item) == 2)
			{
				auto p = py::toCpp<pair<u16string, string>>(item);
				analyzed.emplace_back(p.first, parseTag(p.second.c_str()), undefSenseId);
			}
			else if (Py_SIZE(item) == 3)
			{
				auto p = py::toCpp<tuple<u16string, string, uint8_t>>(item);
				analyzed.emplace_back(get<0>(p), parseTag(get<1>(p).c_str()), get<2>(p));
			}
			else if (Py_SIZE(item) == 4)
			{
				auto t = py::toCpp<tuple<u16string, string, size_t, size_t>>(item);
				analyzed.emplace_back(get<0>(t), parseTag(get<1>(t).c_str()), undefSenseId);
				positions.emplace_back(get<2>(t), get<3>(t));
			}
			else
			{
				auto t = py::toCpp<tuple<u16string, string, uint8_t, size_t, size_t>>(item);
				analyzed.emplace_back(get<0>(t), parseTag(get<1>(t).c_str()), get<2>(t));
				positions.emplace_back(get<3>(t), get<4>(t));
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
#ifdef Py_GIL_DISABLED
	std::unique_lock lock{ *rwMutex };
#endif
	auto added = builder.addPreAnalyzedWord(utf8To16(form), analyzed, positions, score, dialect);
	if (added) kiwi.reset();
	return added;
}

std::vector<std::pair<uint32_t, std::u16string>> KiwiObject::addRule(const string& tag, PyObject* replacer, float score)
{
	if (!PyCallable_Check(replacer)) throw py::ValueError{ "`replacer` must be an callable." };

#ifdef Py_GIL_DISABLED
	std::unique_lock lock{ *rwMutex };
#endif
	auto pos = parseTag(tag.c_str());
	auto added = builder.addRule(pos, [&](const u16string& input)
	{
		py::UniqueObj ret{ PyObject_CallFunctionObjArgs(replacer, py::UniqueObj{ py::buildPyValue(input) }.get(), nullptr) };
		if (!ret) throw py::ExcPropagation{};
		return py::toCpp<u16string>(ret.get());
	}, score);
	if (!added.empty()) kiwi.reset();
	return added;
}

size_t KiwiObject::loadUserDictionary(const string& path)
{
#ifdef Py_GIL_DISABLED
	std::unique_lock lock{ *rwMutex };
#endif
	size_t ret = builder.loadDictionary(path);
	if (ret) kiwi.reset();
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
		PyList_SetItem(retList.get(), idx++, v.release());
	}
	return retList;
}

py::UniqueObj KiwiObject::extractAddWords(PyObject* sentences, size_t minCnt, size_t maxWordLen, float minScore, float posScore, bool lmFilter)
{
#ifdef Py_GIL_DISABLED
	std::unique_lock lock{ *rwMutex };
#endif
	auto res = builder.extractAddWords(obj2reader(sentences), minCnt, maxWordLen, minScore, posScore, lmFilter);
	kiwi.reset();

	py::UniqueObj retList{ PyList_New(res.size()) };
	size_t idx = 0;
	for (auto& r : res)
	{
		auto v = py::buildPyTuple(utf16To8(r.form).c_str(), r.score, r.freq, r.posScore[POSTag::nnp]);
		if (!v) throw py::ExcPropagation{};
		PyList_SetItem(retList.get(), idx++, v.release());
	}
	return retList;
}

py::UniqueObj KiwiObject::analyze(PyObject* text, size_t topN, 
	Match matchOptions, bool echo, PyObject* blockList, bool openEnding, 
	Dialect allowedDialects, float dialectCost,
	PyObject* pretokenized, PyObject* config)
{
	auto kiwiInst = doPrepare();
	KiwiConfig cConfig = toKiwiConfig(config);

	if (PyUnicode_Check(text))
	{
		const unordered_set<const Morpheme*>* morphs = nullptr;
		pair<vector<PretokenizedSpan>, vector<py::UniqueObj>> pretokenizedSpans;
		if (blockList != Py_None) morphs = &((MorphemeSetObject*)blockList)->getMorphemeSet();
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
		auto res = kiwiInst->analyze(so.str, topN, AnalyzeOption{ matchOptions, morphs, openEnding, allowedDialects, dialectCost}, pretokenizedSpans.first, cConfig);
		if (res.size() > topN) res.erase(res.begin() + topN, res.end());
		return resToPyList(move(res), this, kiwiInst, move(pretokenizedSpans.second));
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
		ret->options = AnalyzeOption{ matchOptions, nullptr, openEnding, allowedDialects, dialectCost };
		ret->config = cConfig;
		ret->echo = !!echo;
		ret->kiwiInst = kiwiInst;

		if (blockList != Py_None)
		{
			ret->blocklist = py::UniqueCObj<MorphemeSetObject>{ (MorphemeSetObject*)blockList };
			ret->options.blocklist = &ret->blocklist->getMorphemeSet();
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

		for (size_t i = 0; i < kiwiInst->getNumThreads() * 16; ++i)
		{
			if (!ret->feed()) break;
		}
		return ret;
	}
}

py::UniqueObj KiwiObject::getMorpheme(size_t id)
{
	auto kiwiInst = doPrepare();

	auto ret = py::makeNewObject<TokenObject>();
	auto* morph = kiwiInst->idToMorph(id);
	if (!morph) throw py::ValueError{ "out of range" };
	auto joinedForm = joinHangul(morph->getForm());
	ret->kiwiInst = kiwiInst;
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
	auto kiwiInst = doPrepare();
	auto joiner = kiwiInst->newJoiner(!!lmSearch);
	size_t prevHash = 0;
	size_t prevEnd = 0;
	py::foreach<PyObject*>(morphs, [&](PyObject* item)
	{
		if (PyObject_IsInstance(item, (PyObject*)py::Type<TokenObject>))
		{
			auto& token = *((TokenObject*)item);
			cmb::Space space = cmb::Space::none;
			if (token.resultHash == prevHash)
			{
				space = token._pos <= prevEnd ? cmb::Space::no_space : cmb::Space::insert_space;
			}
			
			if (!token.kiwiInst.expired() && token._morph && token._morph->kform && !token._morph->kform->empty())
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
			string form = py::toCpp<string>(PyTuple_GetItem(item, 0));
			string tag = py::toCpp<string>(PyTuple_GetItem(item, 1));
			const char* p = strchr(tag.c_str(), '-');
			joiner.add(utf8To16(form), parseTag(tag.c_str()), p ? false : true);
			prevHash = 0;
			prevEnd = 0;
		}
		else if (PyTuple_Check(item) && PyTuple_Size(item) == 3)
		{
			string form = py::toCpp<string>(PyTuple_GetItem(item, 0));
			string tag = py::toCpp<string>(PyTuple_GetItem(item, 1));
			const char* p = strchr(tag.c_str(), '-');
			cmb::Space space = PyObject_IsTrue(PyTuple_GetItem(item, 2)) ? cmb::Space::insert_space : cmb::Space::no_space;
			joiner.add(utf8To16(form), parseTag(tag.c_str()), p ? false : true, space);
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

template<class E>
inline uint32_t convertToMorphId(const Kiwi* kiwi, PyObject* target, E&& errorMsg)
{
	if (PyUnicode_Check(target) || (PyTuple_Check(target) && (PyTuple_Size(target) == 2 || PyTuple_Size(target) == 3)))
	{
		u16string form;
		POSTag tag = POSTag::unknown;
		uint8_t senseId = undefSenseId;
		if (PyUnicode_Check(target))
		{
			form = py::toCpp<u16string>(target);
		}
		else
		{
			form = py::toCpp<u16string>(PyTuple_GetItem(target, 0));
			tag = parseTag(py::toCpp<u16string>(PyTuple_GetItem(target, 1)));
			if (PyTuple_Size(target) > 2)
			{
				senseId = py::toCpp<uint8_t>(PyTuple_GetItem(target, 2));
			}
		}

		auto cands = kiwi->findMorphemes(form, tag, senseId);
		if (cands.empty())
		{
			throw py::ValueError{ "No morpheme found for the given form: " + utf16To8(form) };
		}
		if (cands.size() > 1)
		{
			string errMsg = "Multiple morphemes found for the given form: ";
			for (auto c : cands)
			{
				errMsg += utf16To8(form);
				errMsg.push_back('/');
				errMsg += tagToString(c->tag);
				errMsg.push_back('_');
				errMsg.push_back('_');
				errMsg += to_string(c->senseId);
				errMsg.push_back(',');
				errMsg.push_back(' ');
			}
			errMsg.pop_back();
			errMsg.pop_back();
			throw py::ValueError{ errMsg };
		}
		return cands[0]->lmMorphemeId;
	}
	else if (PyLong_Check(target))
	{
		return py::toCpp<uint32_t>(target);
	}
	else
	{
		throw py::ValueError{ std::forward<E>(errorMsg)};
	}
}

inline Vector<uint32_t> convertToIds(const Kiwi* kiwi, PyObject* iterable)
{
	Vector<uint32_t> ids;
	py::foreach<PyObject*>(iterable, [&](PyObject* item)
	{
		ids.emplace_back(convertToMorphId(kiwi, item, "`prefix` must be an instance of `str`, `Tuple[str, str]`, `Tuple[str, str, int]` or `int`."));
	}, "`prefix` must be an iterable of `Tuple[str, str]`, `Tuple[str, str, int]` or `int`");
	return ids;
}

py::UniqueObj KiwiObject::mostSimilarMorphemes(PyObject* retTy, PyObject* target, size_t topN)
{
	auto kiwiInst = doPrepare();
	auto congLm = dynamic_cast<const lm::CoNgramModelBase*>(kiwiInst->getLangModel());
	if (!congLm)
	{
		throw py::ValueError{ "`most_similar_morphemes` is supported only for CoNgramModel." };
	}

	const uint32_t targetId = convertToMorphId(kiwiInst.get(), target, "`target` must be an instance of `str`, `Tuple[str, str]`, `Tuple[str, str, int]` or `int`.");
	Vector<pair<uint32_t, float>> output(topN);
	output.resize(congLm->mostSimilarWords(targetId, topN, output.data()));
	
	py::UniqueObj ret{ PyList_New(output.size()) };
	for (size_t i = 0; i < output.size(); ++i)
	{
		auto* morph = kiwiInst->idToMorph(output[i].first);
		PyList_SetItem(ret.get(), i, PyObject_CallObject(retTy, py::buildPyTuple(
			joinHangul(morph->getForm()), 
			tagToString(morph->tag),
			morph->senseId,
			output[i].first,
			output[i].second
		).get()));
	}
	return ret;
}

py::UniqueObj KiwiObject::mostSimilarContexts(PyObject* retTy, PyObject* target, PyObject* contextId, size_t topN)
{
	auto kiwiInst = doPrepare();
	auto congLm = dynamic_cast<const lm::CoNgramModelBase*>(kiwiInst->getLangModel());
	if (!congLm)
	{
		throw py::ValueError{ "`most_similar_contexts` is supported only for CoNgramModel." };
	}
	prepareContextMap(kiwiInst.get(), congLm);

	Vector<uint32_t> targetIds;
	if (target != Py_None)
	{
		targetIds = convertToIds(kiwiInst.get(), target);
	}
	
	const uint32_t targetContextId = target == Py_None ?
		PyLong_AsLong(contextId) :
		congLm->toContextId(targetIds.data(), targetIds.size());
	Vector<pair<uint32_t, float>> output(topN);
	if (topN > 1)
	{
		output.resize(congLm->mostSimilarContexts(targetContextId, topN - 1, output.data() + 1) + 1);
	}
	output[0].first = targetContextId;
	output[0].second = 1;

	py::UniqueObj ret{ PyList_New(output.size()) };
	for (size_t i = 0; i < output.size(); ++i)
	{
		auto* morph = kiwiInst->idToMorph(output[i].first);
		auto& forms = contextForms[output[i].first];
		auto& analysesData = contextAnalyses[output[i].first].first;
		auto& analysesPtr = contextAnalyses[output[i].first].second;

		py::UniqueObj analysisList{ PyList_New(analysesPtr.size()) };
		for (size_t j = 0; j < analysesPtr.size(); ++j)
		{
			const size_t start = j > 0 ? analysesPtr[j - 1] : 0;
			const size_t end = analysesPtr[j];
			py::UniqueObj morphs{ PyList_New(end - start) };
			for (size_t k = start; k < end; ++k)
			{
				auto* morph = kiwiInst->idToMorph(analysesData[k]);
				PyList_SetItem(morphs.get(), k - start, py::buildPyTuple(
					joinHangul(morph->getForm()),
					tagToString(morph->tag),
					morph->senseId
				).release());
			}
			PyList_SetItem(analysisList.get(), j, morphs.release());
		}
		PyList_SetItem(ret.get(), i, PyObject_CallObject(retTy, py::buildPyTuple(
			forms,
			analysisList.get(),
			output[i].first,
			output[i].second
		).get()));
	}
	return ret;
}

py::UniqueObj KiwiObject::predictNextMorpheme(PyObject* retTy, PyObject* prefix, PyObject* bgPrefix, float bgWeight, size_t topN)
{
	auto kiwiInst = doPrepare();
	auto congLm = dynamic_cast<const lm::CoNgramModelBase*>(kiwiInst->getLangModel());
	if (!congLm)
	{
		throw py::ValueError{ "`predict_next_morpheme` is supported only for CoNgramModel." };
	}

	Vector<uint32_t> prefixIds = convertToIds(kiwiInst.get(), prefix);
	Vector<uint32_t> bgPrefixIds;
	if (bgPrefix != Py_None)
	{
		bgPrefixIds = convertToIds(kiwiInst.get(), bgPrefix);
	}

	const uint32_t prefixContextId = congLm->toContextId(prefixIds.data(), prefixIds.size());
	Vector<pair<uint32_t, float>> output(topN);
	if (bgPrefixIds.empty())
	{
		output.resize(congLm->predictWordsFromContext(prefixContextId, topN, output.data()));
	}
	else
	{
		const uint32_t bgPrefixContextId = congLm->toContextId(bgPrefixIds.data(), bgPrefixIds.size());
		output.resize(congLm->predictWordsFromContextDiff(prefixContextId, bgPrefixContextId, bgWeight, topN, output.data()));
	}

	py::UniqueObj ret{ PyList_New(output.size()) };
	for (size_t i = 0; i < output.size(); ++i)
	{
		auto* morph = kiwiInst->idToMorph(output[i].first);
		PyList_SetItem(ret.get(), i, PyObject_CallObject(retTy, py::buildPyTuple(
			joinHangul(morph->getForm()),
			tagToString(morph->tag),
			morph->senseId,
			output[i].first,
			output[i].second
		).get()));
	}
	return ret;
}

float KiwiObject::morphemeSimilarity(PyObject* a, PyObject* b)
{
	auto kiwiInst = doPrepare();
	auto congLm = dynamic_cast<const lm::CoNgramModelBase*>(kiwiInst->getLangModel());
	if (!congLm)
	{
		throw py::ValueError{ "`morpheme_similarity` is supported only for CoNgramModel." };
	}

	const uint32_t aId = convertToMorphId(kiwiInst.get(), a, "`morpheme1` must be an instance of `str`, `Tuple[str, str]`, `Tuple[str, str, int]` or `int`.");
	const uint32_t bId = convertToMorphId(kiwiInst.get(), b, "`morpheme2` must be an instance of `str`, `Tuple[str, str]`, `Tuple[str, str, int]` or `int`.");

	return congLm->wordSimilarity(aId, bId);
}

float KiwiObject::contextSimilarity(PyObject* a, PyObject* b)
{
	auto kiwiInst = doPrepare();
	auto congLm = dynamic_cast<const lm::CoNgramModelBase*>(kiwiInst->getLangModel());
	if (!congLm)
	{
		throw py::ValueError{ "`morpheme_similarity` is supported only for CoNgramModel." };
	}

	const Vector<uint32_t> aId = convertToIds(kiwiInst.get(), a);
	const Vector<uint32_t> bId = convertToIds(kiwiInst.get(), b);

	const uint32_t aContextId = congLm->toContextId(aId.data(), aId.size());
	const uint32_t bContextId = congLm->toContextId(bId.data(), bId.size());
	return congLm->contextSimilarity(aContextId, bContextId);
}

void KiwiObject::convertHSData(
	PyObject* inputPathes,
	const string& outputPath,
	PyObject* morphemeDefPath,
	size_t morphemeDefMinCnt,
	bool generateOovDict,
	PyObject* transform
) const
{
	
	string morphemeDefPathStr;
	if (morphemeDefPath && morphemeDefPath != Py_None)
	{
		morphemeDefPathStr = py::toCpp<string>(morphemeDefPath);
	}

	vector<pair<pair<string, POSTag>, vector<pair<string, POSTag>>>> transformMap;
	if (transform && transform != Py_None)
	{
		py::foreach<PyObject*>(transform, [&](PyObject* item)
		{
			pair<string, POSTag> key;
			vector<pair<string, POSTag>> values;
			py::foreach<pair<string, string>>(item, [&](const pair<string, string>& token)
			{
				const POSTag tag = parseTag(token.second.c_str());
				if (key.first.empty())
				{
					key = make_pair(token.first, tag);
				}
				else
				{
					values.emplace_back(token.first, tag);
				}
			}, "`transform` must be an iterable of `List[Tuple[str, str]]`.");
			transformMap.emplace_back(key, move(values));
		}, "`transform` must be an iterable of `List[Tuple[str, str]]`.");
	}

	builder.convertHSData(py::toCpp<vector<string>>(inputPathes), 
		outputPath, 
		morphemeDefPathStr, 
		morphemeDefMinCnt, 
		generateOovDict,
		transformMap.empty() ? nullptr : &transformMap
	);
}

py::UniqueObj KiwiObject::makeHSDataset(PyObject* inputPathes, 
	size_t batchSize, 
	size_t causalContextSize, 
	size_t windowSize, 
	size_t numWorkers, 
	float dropout, 
	float dropoutOnHistory,
	float nounAugmentingProb,
	float emojiAugmentingProb,
	float sbAugmentingProb,
	size_t generateUnlikelihoods,
	PyObject* tokenFilter, 
	PyObject* windowFilter, 
	float splitRatio, 
	bool separateDefaultMorpheme, 
	PyObject* morphemeDefPath,
	size_t morphemeDefMinCnt,
	const std::vector<std::pair<size_t, std::vector<uint32_t>>>& contextualMapper,
	PyObject* transform,
	size_t seed
) const
{
	KiwiBuilder::TokenFilter tf, wf;
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
	if (windowFilter && windowFilter != Py_None)
	{
		wf = [&](const u16string& form, POSTag tag)
		{
			py::UniqueObj ret{ PyObject_CallObject(windowFilter, py::buildPyTuple(form, tagToString(tag)).get()) };
			if (!ret) throw py::ExcPropagation{};
			auto truth = PyObject_IsTrue(ret.get());
			if (truth < 0) throw py::ExcPropagation{};
			return !!truth;
		};
	}

	vector<pair<pair<string, POSTag>, vector<pair<string, POSTag>>>> transformMap;
	if (transform && transform != Py_None)
	{
		py::foreach<PyObject*>(transform, [&](PyObject* item)
		{
			pair<string, POSTag> key;
			vector<pair<string, POSTag>> values;
			py::foreach<pair<string, string>>(item, [&](const pair<string, string>& token)
			{
				const POSTag tag = parseTag(token.second.c_str());
				if (key.first.empty())
				{
					key = make_pair(token.first, tag);
				}
				else
				{
					values.emplace_back(token.first, tag);
				}
			}, "`transform` must be an iterable of `List[Tuple[str, str]]`.");
			transformMap.emplace_back(key, move(values));
		}, "`transform` must be an iterable of `List[Tuple[str, str]]`.");
	}

	string morphemeDefPathStr;
	if (morphemeDefPath && morphemeDefPath != Py_None)
	{
		morphemeDefPathStr = py::toCpp<string>(morphemeDefPath);
	}

	HSDataset anotherDataset;
	auto dataset = builder.makeHSDataset(py::toCpp<vector<string>>(inputPathes), 
		batchSize, 
		causalContextSize, 
		windowSize, 
		numWorkers, 
		HSDatasetOption {
			dropout,
			dropoutOnHistory,
			nounAugmentingProb,
			emojiAugmentingProb,
			sbAugmentingProb,
			generateUnlikelihoods,
		},
		tf, 
		wf, 
		splitRatio, 
		separateDefaultMorpheme, 
		morphemeDefPathStr,
		morphemeDefMinCnt,
		contextualMapper,
		&anotherDataset,
		transformMap.empty() ? nullptr : &transformMap);
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
	using _InitArgs = std::tuple<PyObject*, bool>;

	NgramExtractor ne;
	std::shared_ptr<Kiwi> kiwiInst;
#ifdef Py_GIL_DISABLED
	std::unique_ptr<std::shared_mutex> rwMutex;
#endif

	NgramExtractorObject() = default;

	NgramExtractorObject(PyObject* kiwi, bool gatherLmScore)
	{
		if (!PyObject_IsInstance(kiwi, (PyObject*)py::Type<KiwiObject>))
		{
			throw py::ValueError{ "`kiwi` must be an instance of `Kiwi`." };
		}
		kiwiInst = ((KiwiObject*)kiwi)->doPrepare();
		ne = NgramExtractor{ *kiwiInst.get(), gatherLmScore};
#ifdef Py_GIL_DISABLED
		rwMutex = std::make_unique<std::shared_mutex>();
#endif
	}

	size_t add(PyObject* texts)
	{
		if (PyUnicode_Check(texts))
		{
#ifdef Py_GIL_DISABLED
			std::unique_lock lock{ *rwMutex };
#endif
			return ne.addText(py::toCpp<u16string>(texts));
		}
		else
		{
			py::UniqueObj iter{ PyObject_GetIter(texts) };
#ifdef Py_GIL_DISABLED
			std::unique_lock lock{ *rwMutex };
#endif
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
#ifdef Py_GIL_DISABLED
		std::shared_lock lock{ *rwMutex };
#endif
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

PyMODINIT_FUNC PyInit__kiwipiepy()
{
	py::CustomExcHandler::add<kiwi::IOException, py::OSError>();
	py::CustomExcHandler::add<kiwi::SerializationException, py::ValueError>();
	py::CustomExcHandler::add<kiwi::FormatException, py::ValueError>();
	py::CustomExcHandler::add<kiwi::UnicodeException, py::ValueError>();
	py::CustomExcHandler::add<kiwi::UnknownMorphemeException, py::ValueError>();
	py::CustomExcHandler::add<kiwi::SwTokenizerException, py::ValueError>();
	py::CustomExcHandler::add<kiwi::Exception, py::Exception>();

	return gModule.init(
		py::define<TypoTransformerObject>("kiwipiepy._TypoTransformer", "_TypoTransformer", Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE)
		.template method<&TypoTransformerObject::generate>("generate")
		.template method<&TypoTransformerObject::copy>("copy")
		.template method<&TypoTransformerObject::update>("update")
		.template method<&TypoTransformerObject::scaleCost>("scale_cost")
		.template property<&TypoTransformerObject::getContinualTypoCost>("_continual_typo_cost")
		.template property<&TypoTransformerObject::getLengtheningTypoCost>("_lengthening_typo_cost")
		.template property<&TypoTransformerObject::getDefs>("_defs"),

		py::define<HSDatasetObject>("kiwipiepy._HSDataset", "_HSDataset", Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE)
		.template method<&HSDatasetObject::getVocabInfo>("get_vocab_info")
		.template method<&HSDatasetObject::getSent>("get_sent")
		.template method<&HSDatasetObject::estimVocabFrequency>("estim_vocab_frequency")
		.template method<&HSDatasetObject::extractPrefixes>("extract_prefixes")
		.template property<&HSDatasetObject::getVocabSize>("vocab_size")
		.template property<&HSDatasetObject::getKnlmVocabSize>("knlm_vocab_size")
		.template property<&HSDatasetObject::getNgramNodeSize>("ngram_node_size")
		.template property<&HSDatasetObject::getBatchSize>("batch_size")
		.template property<&HSDatasetObject::getWindowSize>("window_size")
		.template property<&HSDatasetObject::numSents>("num_sents")
		.template property<&HSDatasetObject::getWindowTokenValidness>("window_token_validness")
		.template sqLen<&HSDatasetObject::len>(),

		py::define<HSDatasetIterObject>("kiwipiepy._HSDatasetIter", "_HSDatasetIter", Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE),

		py::define<KNLangModelNextTokensResultObject>("kiwipiepy._KNLangModelNextTokensResult", "_KNLangModelNextTokensResult")
		.template sqLen<&KNLangModelNextTokensResultObject::len>()
		.template sqGetItem<&KNLangModelNextTokensResultObject::getitem>(),

		py::define<KNLangModelEvaluateResultObject>("kiwipiepy._KNLangModelEvaluateResult", "_KNLangModelEvaluateResult")
		.template method<&KNLangModelEvaluateResultObject::dir>("__dir__")
		.template mpLen<&KNLangModelEvaluateResultObject::len>()
		.template mpGetItem<&KNLangModelEvaluateResultObject::getitem>()
		.template getAttrO<&KNLangModelEvaluateResultObject::getattr>(),

		py::define<KNLangModelObject>("kiwipiepy._KNLangModel", "_KNLangModel", Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE)
		.template staticMethod<&KNLangModelObject::fromArrays>("from_arrays")
		.template staticMethod<&KNLangModelObject::load>("load")
		.template method<&KNLangModelObject::save>("save")
		.template method<&KNLangModelObject::nextTokens>("next_tokens")
		.template method<&KNLangModelObject::evaluate>("evaluate")
		.template property<&KNLangModelObject::ngramSize>("_ngram_size")
		.template property<&KNLangModelObject::vocabSize>("_vocab_size")
		.template property<&KNLangModelObject::numNodes>("_num_nodes")
		.template property<&KNLangModelObject::numWorkers>("_num_workers"),

		py::define<KiwiObject>("kiwipiepy._Kiwi", "_Kiwi", Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE)
		.template method<&KiwiObject::addUserWord>("add_user_word")
		.template method<&KiwiObject::addPreAnalyzedWord>("add_pre_analyzed_word")
		.template method<&KiwiObject::addRule>("add_rule")
		.template method<&KiwiObject::loadUserDictionary>("load_user_dictionary")
		.template method<&KiwiObject::extractWords>("extract_words")
		.template method<&KiwiObject::extractAddWords>("extract_add_words")
		.template method<&KiwiObject::analyze>("analyze")
		.template method<&KiwiObject::getMorpheme>("morpheme")
		.template method<&KiwiObject::join>("join")
		.template method<&KiwiObject::convertHSData>("convert_hsdata")
		.template method<&KiwiObject::makeHSDataset>("make_hsdataset")
		.template method<&KiwiObject::listAllScripts>("list_all_scripts")
		.template method<&KiwiObject::mostSimilarMorphemes>("most_similar_morphemes")
		.template method<&KiwiObject::mostSimilarContexts>("most_similar_contexts")
		.template method<&KiwiObject::predictNextMorpheme>("predict_next_morpheme")
		.template method<&KiwiObject::morphemeSimilarity>("morpheme_similarity")
		.template method<&KiwiObject::contextSimilarity>("context_similarity")
		.template property<&KiwiObject::getGlobalConfig, &KiwiObject::setGlobalConfig>("__global_config")
		.template property<&KiwiObject::typoCostThreshold, &KiwiObject::typoCostThreshold>("_typo_cost_threshold")
		.template property<&KiwiObject::getNumWorkers>("_num_workers")
		.template property<&KiwiObject::getModelType>("_model_type"),

		py::define<TokenObject>("kiwipiepy.Token", "Token")
		.template property<&TokenObject::_form>("form")
		.template property<&TokenObject::_tag>("tag")
		.template property<&TokenObject::_pos>("start")
		.template property<&TokenObject::_len>("len")
		.template property<&TokenObject::end>("end")
		.template property<&TokenObject::span>("span")
		.template property<&TokenObject::_morphId>("id")
		.template property<&TokenObject::_wordPosition>("word_position")
		.template property<&TokenObject::_sentPosition>("sent_position")
		.template property<&TokenObject::_subSentPosition>("sub_sent_position")
		.template property<&TokenObject::_lineNumber>("line_number")
		.template property<&TokenObject::baseForm>("base_form")
		.template property<&TokenObject::baseId>("base_id")
		.template property<&TokenObject::taggedForm>("tagged_form")
		.template property<&TokenObject::formTag>("form_tag")
		.template property<&TokenObject::_score>("score")
		.template property<&TokenObject::_typoCost>("typo_cost")
		.template property<&TokenObject::_raw_form>("raw_form")
		.template property<&TokenObject::regularity>("regularity")
		.template property<&TokenObject::lemma>("lemma")
		.template property<&TokenObject::_pairedToken>("paired_token")
		.template property<&TokenObject::_userValue>("user_value")
		.template property<&TokenObject::script>("script")
		.template property<&TokenObject::_sense>("sense")
		.template property<&TokenObject::_dialect>("dialect")
		.template sqLen<&TokenObject::len>()
		.template sqGetItem<&TokenObject::getitem>(),

		py::define<MorphemeSetObject>("kiwipiepy._MorphemeSet", "_MorphemeSet", Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE)
		.template method<&MorphemeSetObject::update>("_update"),

		py::define<SwTokenizerObject>("kiwipiepy._SwTokenizer", "_SwTokenizer", Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE)
		.template method<&SwTokenizerObject::encode>("encode")
		.template method<&SwTokenizerObject::encodeFromMorphs>("encode_from_morphs")
		.template method<&SwTokenizerObject::tokenizeAndEncode>("tokenize_encode")
		.template method<&SwTokenizerObject::decode>("decode")
		.template staticMethod<&SwTokenizerObject::train>("_train")
		.template method<&SwTokenizerObject::save>("save")
		.template property<&SwTokenizerObject::config>("_config")
		.template property<&SwTokenizerObject::vocab>("_vocab")
		.template property<&SwTokenizerObject::kiwi>("_kiwi")
		.template sqLen<&SwTokenizerObject::len>(),

		py::define<KiwiResIter>("kiwipiepy._ResIter", "_ResIter"),

		py::define<SwTokenizerResIter>("kiwipiepy._SwTokenizerResIter", "_SwTokenizerResIter"),

		py::define<SwTokenizerResTEIter>("kiwipiepy._SwTokenizerResTEIter", "_SwTokenizerResTEIter"),

		py::define<NgramExtractorObject>("kiwipiepy._NgramExtractor", "_NgramExtractor", Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE)
		.template method<&NgramExtractorObject::add>("add")
		.template method<&NgramExtractorObject::extract>("extract")
		.template staticMethod<&pyExtractSubstrings>("_extract_substrings")
	);
}
