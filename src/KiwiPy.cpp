#include <stdexcept>

#ifdef _DEBUG
#undef _DEBUG
#include "PyDoc.h"
#define _DEBUG
#else 
#include "PyDoc.h"
#endif

#include "core/Kiwi.h"

using namespace std;
using namespace kiwi;

#if PY_MAJOR_VERSION < 3
string PyUnicode_AsUTF8(PyObject* obj)
{
	PyObject* utf8Obj = PyUnicode_AsUTF8String(obj);
	string str = PyString_AsString(utf8Obj);
	Py_DECREF(utf8Obj);
	return str;
}

#endif

string getModuleFilename(PyObject* moduleObj)
{
#if PY_MAJOR_VERSION >= 3
	PyObject* filePath = PyModule_GetFilenameObject(moduleObj);
	string spath = PyUnicode_AsUTF8(filePath);
	Py_DECREF(filePath);
	return spath;
#else
	return PyModule_GetFilename(moduleObj);
#endif
}

static PyObject* gModule;

struct KiwiObject
{
	PyObject_HEAD;
	Kiwi* inst;
	bool owner;

	static void dealloc(KiwiObject* self)
	{
		delete self->inst;
		Py_TYPE(self)->tp_free((PyObject*)self);
	}

	static int init(KiwiObject *self, PyObject *args, PyObject *kwargs)
	{
		const char* modelPath = "./";
		size_t numThread = 0, options = 3;
		static const char* kwlist[] = { "num_workers", "model_path", "options", nullptr };
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|nsn", (char**)kwlist, &numThread, &modelPath, &options)) return -1;
		try
		{
			self->inst = nullptr;
			try
			{
				self->inst = new Kiwi{ modelPath, 0, numThread, options };
			}
			catch (const exception& e)
			{
				string spath = getModuleFilename(PyImport_AddModule("kiwipiepy"));
				spath = spath.substr(0, spath.rfind(spath.rfind('/') != spath.npos ? '/' : '\\') + 1);
				self->inst = new Kiwi{ (spath + modelPath).c_str(), 0, numThread, options };
			}
		}
		catch (const exception& e)
		{
			cerr << e.what() << endl;
			PyErr_SetString(PyExc_Exception, e.what());
			return -1;
		}
		return 0;
	}
};

static PyObject* kiwi__addUserWord(KiwiObject* self, PyObject* args, PyObject *kwargs);
static PyObject* kiwi__analyze(KiwiObject* self, PyObject* args, PyObject *kwargs);
static PyObject* kiwi__extractAddWords(KiwiObject* self, PyObject* args, PyObject *kwargs);
static PyObject* kiwi__extractFilterWords(KiwiObject* self, PyObject* args, PyObject *kwargs);
static PyObject* kiwi__extractWords(KiwiObject* self, PyObject* args, PyObject *kwargs);
static PyObject* kiwi__loadUserDictionary(KiwiObject* self, PyObject* args, PyObject *kwargs);
static PyObject* kiwi__perform(KiwiObject* self, PyObject* args, PyObject *kwargs);
static PyObject* kiwi__prepare(KiwiObject* self, PyObject* args, PyObject *kwargs);
static PyObject* kiwi__setCutOffThreshold(KiwiObject* self, PyObject* args, PyObject *kwargs);
static PyObject* kiwi__get_option(KiwiObject* self, PyObject* args, PyObject *kwargs);
static PyObject* kiwi__set_option(KiwiObject* self, PyObject* args, PyObject *kwargs);

static PyMethodDef Kiwi_methods[] =
{
	{ "addUserWord", (PyCFunction)kiwi__addUserWord, METH_VARARGS | METH_KEYWORDS, Kiwi_add_user_word__doc__ },
	{ "add_user_word", (PyCFunction)kiwi__addUserWord, METH_VARARGS | METH_KEYWORDS, Kiwi_add_user_word__doc__ },
	{ "loadUserDictionary", (PyCFunction)kiwi__loadUserDictionary, METH_VARARGS | METH_KEYWORDS, Kiwi_load_user_dictionary__doc__ },
	{ "load_user_dictionary", (PyCFunction)kiwi__loadUserDictionary, METH_VARARGS | METH_KEYWORDS, Kiwi_load_user_dictionary__doc__ },
	{ "extractWords", (PyCFunction)kiwi__extractWords, METH_VARARGS | METH_KEYWORDS, Kiwi_extract_words__doc__ },
	{ "extract_words", (PyCFunction)kiwi__extractWords, METH_VARARGS | METH_KEYWORDS, Kiwi_extract_words__doc__ },
	{ "extractFilterWords", (PyCFunction)kiwi__extractFilterWords, METH_VARARGS | METH_KEYWORDS, Kiwi_extract_filter_words__doc__ },
	{ "extract_filter_words", (PyCFunction)kiwi__extractFilterWords, METH_VARARGS | METH_KEYWORDS, Kiwi_extract_filter_words__doc__ },
	{ "extractAddWords", (PyCFunction)kiwi__extractAddWords, METH_VARARGS | METH_KEYWORDS, Kiwi_extract_add_words__doc__ },
	{ "extract_add_words", (PyCFunction)kiwi__extractAddWords, METH_VARARGS | METH_KEYWORDS, Kiwi_extract_add_words__doc__ },
	{ "perform", (PyCFunction)kiwi__perform, METH_VARARGS | METH_KEYWORDS, Kiwi_perform__doc__ },
	{ "setCutOffThreshold", (PyCFunction)kiwi__setCutOffThreshold, METH_VARARGS | METH_KEYWORDS, Kiwi_set_cutoff_threshold__doc__ },
	{ "set_cutoff_threshold", (PyCFunction)kiwi__setCutOffThreshold, METH_VARARGS | METH_KEYWORDS, Kiwi_set_cutoff_threshold__doc__ },
	{ "prepare", (PyCFunction)kiwi__prepare, METH_VARARGS | METH_KEYWORDS, Kiwi_prepare__doc__ },
	{ "analyze", (PyCFunction)kiwi__analyze, METH_VARARGS | METH_KEYWORDS, Kiwi_analyze__doc__ },
	{ "get_option", (PyCFunction)kiwi__get_option, METH_VARARGS | METH_KEYWORDS, Kiwi_get_option__doc__ },
	{ "set_option", (PyCFunction)kiwi__set_option, METH_VARARGS | METH_KEYWORDS, Kiwi_set_option__doc__ },
	{ nullptr }
};

static PyObject* kiwi__version(KiwiObject* self, void* closure);

static PyGetSetDef Kiwi_getsets[] = 
{
	{ (char*)"version", (getter)kiwi__version, nullptr, "get version", nullptr },
};

static PyTypeObject Kiwi_type = {
	PyVarObject_HEAD_INIT(nullptr, 0)
	"kiwipiepy.Kiwi",             /* tp_name */
	sizeof(KiwiObject), /* tp_basicsize */
	0,                         /* tp_itemsize */
	(destructor)KiwiObject::dealloc, /* tp_dealloc */
	0,                         /* tp_print */
	0,                         /* tp_getattr */
	0,                         /* tp_setattr */
	0,                         /* tp_reserved */
	0,                         /* tp_repr */
	0,                         /* tp_as_number */
	0,                         /* tp_as_sequence */
	0,                         /* tp_as_mapping */
	0,                         /* tp_hash  */
	0,                         /* tp_call */
	0,                         /* tp_str */
	0,                         /* tp_getattro */
	0,                         /* tp_setattro */
	0,                         /* tp_as_buffer */
	Py_TPFLAGS_DEFAULT,   /* tp_flags */
	"Kiwi()",           /* tp_doc */
	0,                         /* tp_traverse */
	0,                         /* tp_clear */
	0,                         /* tp_richcompare */
	0,                         /* tp_weaklistoffset */
	0,                         /* tp_iter */
	0,                         /* tp_iternext */
	Kiwi_methods,             /* tp_methods */
	0,						 /* tp_members */
	0,        /* tp_getset */
	0,                         /* tp_base */
	0,                         /* tp_dict */
	0,                         /* tp_descr_get */
	0,                         /* tp_descr_set */
	0,                         /* tp_dictoffset */
	(initproc)KiwiObject::init,      /* tp_init */
	PyType_GenericAlloc,
	PyType_GenericNew,
};

static PyObject* kiwi__addUserWord(KiwiObject* self, PyObject* args, PyObject *kwargs)
{
	const char* word;
	const char* tag = "NNP";
	float score = 0;
	static const char* kwlist[] = { "word", "tag", "score", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|sf", (char**)kwlist, &word, &tag, &score)) return nullptr;
	try
	{
		return Py_BuildValue("n", self->inst->addUserWord(Kiwi::toU16(word), makePOSTag(Kiwi::toU16(tag)), score));
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

static PyObject* kiwi__loadUserDictionary(KiwiObject* self, PyObject* args, PyObject *kwargs)
{
	const char* path;
	static const char* kwlist[] = { "dict_path", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s", (char**)kwlist, &path)) return nullptr;
	try
	{
		return Py_BuildValue("n", self->inst->loadUserDictionary(path));
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

static PyObject* kiwi__extractWords(KiwiObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject *argReader;
	size_t minCnt = 10, maxWordLen = 10;
	float minScore = 0.25f;
	static const char* kwlist[] = { "reader", "min_cnt", "max_word_len", "min_score", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|nnf", (char**)kwlist, &argReader, &minCnt, &maxWordLen, &minScore)) return nullptr;
	if (!PyCallable_Check(argReader)) return PyErr_SetString(PyExc_TypeError, "extractWords requires 1st parameter which is callable"), nullptr;
	try
	{
		auto res = self->inst->extractWords([argReader](size_t id) -> u16string
		{
			PyObject* argList = Py_BuildValue("(n)", id);
			PyObject* retVal = PyEval_CallObject(argReader, argList);
			Py_DECREF(argList);
			if (!retVal) throw bad_exception();
			if (PyObject_Not(retVal))
			{
				Py_DECREF(retVal);
				return {};
			}
			auto p = Kiwi::toU16(PyUnicode_AsUTF8(retVal));
			Py_DECREF(retVal);
			return p;
		}, minCnt, maxWordLen, minScore);

		PyObject* retList = PyList_New(res.size());
		size_t idx = 0;
		for (auto& r : res)
		{
			PyList_SetItem(retList, idx++, Py_BuildValue("(sfnf)", Kiwi::toU8(r.form).c_str(), r.score, r.freq, r.posScore[KPOSTag::NNP]));
		}
		return retList;
	}
	catch (const bad_exception& e)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}

	Py_INCREF(Py_None);
	return Py_None;
}

static PyObject* kiwi__extractFilterWords(KiwiObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject *argReader;
	size_t minCnt = 10, maxWordLen = 10;
	float minScore = 0.25f, posScore = -3;
	static const char* kwlist[] = { "reader", "min_cnt", "max_word_len", "min_score", "pos_score", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|nnff", (char**)kwlist, &argReader, &minCnt, &maxWordLen, &minScore, &posScore)) return nullptr;
	if (!PyCallable_Check(argReader)) return PyErr_SetString(PyExc_TypeError, "extractFilterWords requires 1st parameter which is callable"), nullptr;
	try
	{
		auto res = self->inst->extractWords([argReader](size_t id) -> u16string
		{
			PyObject* argList = Py_BuildValue("(n)", id);
			PyObject* retVal = PyEval_CallObject(argReader, argList);
			Py_DECREF(argList);
			if (!retVal) throw bad_exception();
			if (PyObject_Not(retVal))
			{
				Py_DECREF(retVal);
				return {};
			}
			auto p = Kiwi::toU16(PyUnicode_AsUTF8(retVal));
			Py_DECREF(retVal);
			return p;
		}, minCnt, maxWordLen, minScore);

		res = self->inst->filterExtractedWords(move(res), posScore);
		PyObject* retList = PyList_New(res.size());
		size_t idx = 0;
		for (auto& r : res)
		{
			PyList_SetItem(retList, idx++, Py_BuildValue("(sfnf)", Kiwi::toU8(r.form).c_str(), r.score, r.freq, r.posScore[KPOSTag::NNP]));
		}
		return retList;
	}
	catch (const bad_exception& e)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}

	Py_INCREF(Py_None);
	return Py_None;
}

static PyObject* kiwi__extractAddWords(KiwiObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject *argReader;
	size_t minCnt = 10, maxWordLen = 10;
	float minScore = 0.25f, posScore = -3;
	static const char* kwlist[] = { "reader", "min_cnt", "max_word_len", "min_score", "pos_score", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|nnff", (char**)kwlist, &argReader, &minCnt, &maxWordLen, &minScore, &posScore)) return nullptr;
	if (!PyCallable_Check(argReader)) return PyErr_SetString(PyExc_TypeError, "extractAddWords requires 1st parameter which is callable"), nullptr;
	try
	{
		auto res = self->inst->extractAddWords([argReader](size_t id) -> u16string
		{
			PyObject* argList = Py_BuildValue("(n)", id);
			PyObject* retVal = PyEval_CallObject(argReader, argList);
			Py_DECREF(argList);
			if (!retVal) throw bad_exception();
			if (PyObject_Not(retVal))
			{
				Py_DECREF(retVal);
				return {};
			}
			auto p = Kiwi::toU16(PyUnicode_AsUTF8(retVal));
			Py_DECREF(retVal);
			return p;
		}, minCnt, maxWordLen, minScore, posScore);

		PyObject* retList = PyList_New(res.size());
		size_t idx = 0;
		for (auto& r : res)
		{
			PyList_SetItem(retList, idx++, Py_BuildValue("(sfnf)", Kiwi::toU8(r.form).c_str(), r.score, r.freq, r.posScore[KPOSTag::NNP]));
		}
		return retList;
	}
	catch (const bad_exception& e)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}

	Py_INCREF(Py_None);
	return Py_None;
}

static PyObject* kiwi__setCutOffThreshold(KiwiObject* self, PyObject* args, PyObject *kwargs)
{
	float threshold;
	static const char* kwlist[] = { "threshold", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "f", (char**)kwlist, &threshold)) return nullptr;
	try
	{
		self->inst->setCutOffThreshold(threshold);
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}

	Py_INCREF(Py_None);
	return Py_None;
}

static PyObject* kiwi__prepare(KiwiObject* self, PyObject* args, PyObject *kwargs)
{
	static const char* kwlist[] = { nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "", (char**)kwlist)) return nullptr;
	try
	{
		return Py_BuildValue("n", self->inst->prepare());
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

static PyObject* kiwi__get_option(KiwiObject* self, PyObject* args, PyObject *kwargs)
{
	ssize_t option;
	static const char* kwlist[] = { "option", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "n", (char**)kwlist, &option)) return nullptr;
	try
	{
		return Py_BuildValue("n", self->inst->getOption(option));
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

static PyObject* kiwi__set_option(KiwiObject* self, PyObject* args, PyObject *kwargs)
{
	ssize_t option, value;
	static const char* kwlist[] = { "option", "value", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "nn", (char**)kwlist, &option, &value)) return nullptr;
	try
	{
		self->inst->setOption(option, value);
		Py_INCREF(Py_None);
		return Py_None;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

PyObject* resToPyList(const vector<KResult>& res)
{
	PyObject* retList = PyList_New(res.size());
	size_t idx = 0;
	for (auto& p : res)
	{
		PyObject* rList = PyList_New(p.first.size());
		size_t jdx = 0;
		for (auto& q : p.first)
		{
			PyList_SetItem(rList, jdx++, Py_BuildValue("(ssnn)", Kiwi::toU8(q.str()).c_str(), tagToString(q.tag()), (size_t)q.pos(), (size_t)q.len()));
		}
		PyList_SetItem(retList, idx++, Py_BuildValue("(Of)", rList, p.second));
	}
	return retList;
}

static PyObject* kiwi__analyze(KiwiObject* self, PyObject* args, PyObject *kwargs)
{
	size_t topN = 1;
	{
		char* text;
		static const char* kwlist[] = { "text", "top_n", nullptr };
		if (PyArg_ParseTupleAndKeywords(args, kwargs, "s|n", (char**)kwlist, &text, &topN))
		{
			try
			{
				auto res = self->inst->analyze(text, topN);
				return resToPyList(res);
			}
			catch (const exception& e)
			{
				PyErr_SetString(PyExc_Exception, e.what());
				return nullptr;
			}
		}
	}
	{
		PyObject* reader, *receiver;
		static const char* kwlist[] = { "reader", "receiver", "top_n", nullptr };
		if (PyArg_ParseTupleAndKeywords(args, kwargs, "OO|n", (char**)kwlist, &reader, &receiver, &topN))
		{
			try
			{
				if (!PyCallable_Check(reader)) return PyErr_SetString(PyExc_TypeError, "analyze requires 1st parameter which is callable"), nullptr;
				if (!PyCallable_Check(receiver)) return PyErr_SetString(PyExc_TypeError, "analyze requires 2nd parameter which is callable"), nullptr;
				self->inst->analyze(topN, [&reader](size_t id)->u16string
				{
					PyObject* argList = Py_BuildValue("(n)", id);
					PyObject* retVal = PyEval_CallObject(reader, argList);
					Py_DECREF(argList);
					if (!retVal) throw bad_exception();
					if (PyObject_Not(retVal))
					{
						Py_DECREF(retVal);
						return {};
					}
					auto p = Kiwi::toU16(PyUnicode_AsUTF8(retVal));
					Py_DECREF(retVal);
					return p;
				}, [&receiver](size_t id, vector<KResult>&& res)
				{
					PyObject* l = resToPyList(res);
					PyObject* argList = Py_BuildValue("(nO)", id, l);
					PyObject* ret = PyEval_CallObject(receiver, argList);
					if(!ret) throw bad_exception();
					Py_DECREF(ret);
					Py_DECREF(argList);
				});
				Py_INCREF(Py_None);
				return Py_None;
			}
			catch (const bad_exception& e)
			{
				return nullptr;
			}
			catch (const exception& e)
			{
				PyErr_SetString(PyExc_Exception, e.what());
				return nullptr;
			}
		}
	}
	return nullptr;
}


static PyObject* kiwi__perform(KiwiObject* self, PyObject* args, PyObject *kwargs)
{
	size_t topN = 1;
	PyObject* reader, *receiver;
	size_t minCnt = 10, maxWordLen = 10;
	float minScore = 0.25f, posScore = -3;
	static const char* kwlist[] = { "reader", "receiver", "top_n", "min_cnt", "max_word_len", "min_score", "pos_score", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|nnnff", (char**)kwlist, 
		&reader, &receiver, &topN, &minCnt, &maxWordLen, &minScore, &posScore)) return nullptr;
	try
	{
		if (!PyCallable_Check(reader)) return PyErr_SetString(PyExc_TypeError, "perform requires 1st parameter which is callable"), nullptr;
		if (!PyCallable_Check(receiver)) return PyErr_SetString(PyExc_TypeError, "perform requires 2nd parameter which is callable"), nullptr;

		self->inst->perform(topN, [&reader](size_t id)->u16string
		{
			PyObject* argList = Py_BuildValue("(n)", id);
			PyObject* retVal = PyEval_CallObject(reader, argList);
			Py_DECREF(argList);
			if (!retVal) throw bad_exception();
			if (PyObject_Not(retVal))
			{
				Py_DECREF(retVal);
				return {};
			}
			auto p = Kiwi::toU16(PyUnicode_AsUTF8(retVal));
			Py_DECREF(retVal);
			return p;
		}, [&receiver](size_t id, vector<KResult>&& res)
		{
			PyObject* l = resToPyList(res);
			PyObject* argList = Py_BuildValue("(nO)", id, l);
			PyObject* ret = PyEval_CallObject(receiver, argList);
			if (!ret) throw bad_exception();
			Py_DECREF(ret);
			Py_DECREF(argList);
		}, minCnt, maxWordLen, minScore, posScore);
		Py_INCREF(Py_None);
		return Py_None;
	}
	catch (const bad_exception& e)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
	return nullptr;
}

static PyObject* kiwi__version(KiwiObject* self, void* closure)
{
	try
	{
		return Py_BuildValue("n", self->inst->getVersion());
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

PyObject* moduleInit()
{
#if PY_MAJOR_VERSION >= 3
	static PyModuleDef mod =
	{
		PyModuleDef_HEAD_INIT,
		"_kiwipiepy",
		"Kiwi API for Python",
		-1,
		nullptr
	};

	gModule = PyModule_Create(&mod);
#else
	gModule = Py_InitModule3("_kiwipiepy", nullptr, "Kiwi API for Python");
#endif
	if (PyType_Ready(&Kiwi_type) < 0) return nullptr;
	Py_INCREF(&Kiwi_type);
	PyModule_AddObject(gModule, "Kiwi", (PyObject*)&Kiwi_type);
	return gModule;
}

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit__kiwipiepy()
{
	return moduleInit();
}
#else
PyMODINIT_FUNC init_kiwipiepy()
{
	moduleInit();
}
#endif
