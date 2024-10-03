#pragma once
#include <type_traits>
#include <vector>
#include <map>
#include <unordered_map>
#include <tuple>
#include <set>
#include <limits>
#include <exception>
#include <string>
#include <functional>
#include <iostream>
#include <cstring>
#include <deque>
#include <future>
#include <optional>
#include <variant>
#include <numeric>
#include <typeinfo>
#include <typeindex>

#ifdef _DEBUG
#undef _DEBUG
#include <Python.h>
#define _DEBUG
#else
#include <Python.h>
#endif

#include <frameobject.h>

#ifdef USE_NUMPY
#ifdef MAIN_MODULE
#else
#define NO_IMPORT_ARRAY
#endif
#define PY_ARRAY_UNIQUE_SYMBOL KIWIPIEPY_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#endif

#if defined(__clang__)
#define PY_STRONG_INLINE inline
#elif defined(__GNUC__) || defined(__GNUG__)
#define PY_STRONG_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
#define PY_STRONG_INLINE __forceinline 
#endif

namespace py
{
	template<class Ty = PyObject>
	struct UniqueCObj
	{
		Ty* obj = nullptr;
		
		UniqueCObj() {}

		explicit UniqueCObj(Ty* _obj) : obj(_obj) {}

		~UniqueCObj()
		{
			Py_XDECREF(obj);
		}

		UniqueCObj(const UniqueCObj&) = delete;
		UniqueCObj& operator=(const UniqueCObj&) = delete;

		UniqueCObj(UniqueCObj&& o) noexcept
		{
			std::swap(obj, o.obj);
		}

		UniqueCObj& operator=(UniqueCObj&& o) noexcept
		{
			std::swap(obj, o.obj);
			return *this;
		}

		operator UniqueCObj<PyObject>()
		{
			return UniqueCObj<PyObject>{ (PyObject*)release() };
		}

		Ty* get() const
		{
			return obj;
		}

		Ty* release()
		{
			auto o = obj;
			obj = nullptr;
			return o;
		}

		operator bool() const
		{
			return !!obj;
		}

		explicit operator Ty* () const
		{
			return obj;
		}

		Ty* operator->()
		{
			return obj;
		}

		const Ty* operator->() const
		{
			return obj;
		}
	};

	template<class Ty = PyObject>
	struct SharedCObj
	{
		Ty* obj = nullptr;

		SharedCObj() {}

		SharedCObj(Ty* _obj) : obj(_obj) {}

		~SharedCObj()
		{
			Py_XDECREF(obj);
		}

		SharedCObj(const SharedCObj& o)
			: obj(o.obj)
		{
			Py_INCREF(obj);
		}

		SharedCObj& operator=(const SharedCObj& o)
		{
			Py_XDECREF(obj);
			obj = o.obj;
			Py_INCREF(obj);
			return *this;
		}

		SharedCObj(SharedCObj&& o) noexcept
		{
			std::swap(obj, o.obj);
		}

		SharedCObj& operator=(SharedCObj&& o) noexcept
		{
			std::swap(obj, o.obj);
			return *this;
		}

		Ty* get() const
		{
			return obj;
		}

		operator bool() const
		{
			return !!obj;
		}

		operator Ty* () const
		{
			return obj;
		}

		Ty* operator->()
		{
			return obj;
		}

		const Ty* operator->() const
		{
			return obj;
		}
	};

	using UniqueObj = UniqueCObj<>;
	using SharedObj = SharedCObj<>;

	template<class Ty>
	struct StringWithOffset
	{
		Ty str;
		std::vector<size_t> offsets;
	};

	class ForeachFailed : public std::runtime_error
	{
	public:
		ForeachFailed() : std::runtime_error{ "" }
		{
		}
	};

	class ExcPropagation : public std::runtime_error
	{
	public:
		ExcPropagation() : std::runtime_error{ "" }
		{
		}
	};

	class BaseException : public std::runtime_error
	{
	public:
		using std::runtime_error::runtime_error;

		virtual PyObject* pytype() const
		{
			return PyExc_BaseException;
		}
	};

	class Exception : public BaseException
	{
	public:
		using BaseException::BaseException;

		virtual PyObject* pytype() const
		{
			return PyExc_Exception;
		}
	};

	class StopIteration : public Exception
	{
	public:
		using Exception::Exception;

		virtual PyObject* pytype() const
		{
			return PyExc_StopIteration;
		}
	};

	class StopAsyncIteration : public Exception
	{
	public:
		using Exception::Exception;

		virtual PyObject* pytype() const
		{
			return PyExc_StopAsyncIteration;
		}
	};

	class ArithmeticError : public Exception
	{
	public:
		using Exception::Exception;

		virtual PyObject* pytype() const
		{
			return PyExc_ArithmeticError;
		}
	};

	class AssertionError : public Exception
	{
	public:
		using Exception::Exception;

		virtual PyObject* pytype() const
		{
			return PyExc_AssertionError;
		}
	};

	class AttributeError : public Exception
	{
	public:
		using Exception::Exception;

		virtual PyObject* pytype() const
		{
			return PyExc_AttributeError;
		}
	};

	class BufferError : public Exception
	{
	public:
		using Exception::Exception;

		virtual PyObject* pytype() const
		{
			return PyExc_BufferError;
		}
	};

	class EOFError : public Exception
	{
	public:
		using Exception::Exception;

		virtual PyObject* pytype() const
		{
			return PyExc_EOFError;
		}
	};

	class ImportError : public Exception
	{
	public:
		using Exception::Exception;

		virtual PyObject* pytype() const
		{
			return PyExc_ImportError;
		}
	};

	class LookupError : public Exception
	{
	public:
		using Exception::Exception;

		virtual PyObject* pytype() const
		{
			return PyExc_LookupError;
		}
	};

	class IndexError : public LookupError
	{
	public:
		using LookupError::LookupError;

		virtual PyObject* pytype() const
		{
			return PyExc_IndexError;
		}
	};

	class KeyError : public LookupError
	{
	public:
		using LookupError::LookupError;

		virtual PyObject* pytype() const
		{
			return PyExc_KeyError;
		}
	};

	class MemoryError : public Exception
	{
	public:
		using Exception::Exception;

		virtual PyObject* pytype() const
		{
			return PyExc_MemoryError;
		}
	};

	class NameError : public Exception
	{
	public:
		using Exception::Exception;

		virtual PyObject* pytype() const
		{
			return PyExc_NameError;
		}
	};

	class OSError : public Exception
	{
	public:
		using Exception::Exception;

		virtual PyObject* pytype() const
		{
			return PyExc_OSError;
		}
	};

	class ReferenceError : public Exception
	{
	public:
		using Exception::Exception;

		virtual PyObject* pytype() const
		{
			return PyExc_ReferenceError;
		}
	};

	class RuntimeError : public Exception
	{
	public:
		using Exception::Exception;

		virtual PyObject* pytype() const
		{
			return PyExc_RuntimeError;
		}
	};

	class SyntaxError : public Exception
	{
	public:
		using Exception::Exception;

		virtual PyObject* pytype() const
		{
			return PyExc_SyntaxError;
		}
	};

	class SystemError : public Exception
	{
	public:
		using Exception::Exception;

		virtual PyObject* pytype() const
		{
			return PyExc_SystemError;
		}
	};

	class TypeError : public Exception
	{
	public:
		using Exception::Exception;

		virtual PyObject* pytype() const
		{
			return PyExc_TypeError;
		}
	};

	class ValueError : public Exception
	{
	public:
		using Exception::Exception;

		virtual PyObject* pytype() const
		{
			return PyExc_ValueError;
		}
	};

	template<typename _Fn>
	auto handleExc(_Fn&& fn)
		-> typename std::enable_if<std::is_pointer<decltype(fn())>::value, decltype(fn())>::type;

	template<typename _Fn>
	auto handleExc(_Fn&& fn)
		-> typename std::enable_if<std::is_integral<decltype(fn())>::value, decltype(fn())>::type;

	class ConversionFail : public ValueError
	{
	public:
		using ValueError::ValueError;

		template<typename _Ty,
			typename = typename std::enable_if<std::is_constructible<std::function<std::string()>, _Ty>::value>::type
		>
		ConversionFail(_Ty&& callable) : ValueError{ callable() }
		{
		}
	};

	template<typename _Ty, typename = void>
	struct ValueBuilder;

	template<typename _Ty>
	inline UniqueObj buildPyValue(_Ty&& v)
	{
		return ValueBuilder<
			typename std::remove_const<typename std::remove_reference<_Ty>::type>::type
		>{}(std::forward<_Ty>(v));
	}

	template<typename _Ty, typename _FailMsg>
	inline _Ty toCppWithException(PyObject* obj, _FailMsg&& fail)
	{
		_Ty ret;
		if (!obj || !ValueBuilder<_Ty>{}._toCpp(obj, ret)) throw ConversionFail{ std::forward<_FailMsg>(fail) };
		return ret;
	}

	template<typename _Ty>
	inline _Ty getAttr(PyObject* obj, const char* attr)
	{
		py::UniqueObj item{ PyObject_GetAttrString(obj, attr) };
		return toCppWithException<_Ty>(item.get(), [&]() { return std::string{ "Failed to get attribute " } + attr; });
	}

	inline std::string repr(PyObject* o)
	{
		UniqueObj r{ PyObject_Repr(o) };
		if (!r) throw ExcPropagation{};
		return toCppWithException<std::string>(r.get(), "");
	}

	inline std::string reprWithNestedError(PyObject* o)
	{
		PyObject* type, * value, * traceback;
		PyErr_Fetch(&type, &value, &traceback);
		PyErr_Clear();
		UniqueObj r{ PyObject_Repr(o) };
		if (!r) throw ExcPropagation{};
		PyErr_Restore(type, value, traceback);
		return toCppWithException<std::string>(r.get(), "");
	}

	template<typename _Ty>
	inline std::string reprFromCpp(_Ty&& o)
	{
		UniqueObj p{ py::buildPyValue(std::forward<_Ty>(o)) };
		UniqueObj r{ PyObject_Repr(p.get()) };
		if (!r) throw ExcPropagation{};
		return toCppWithException<std::string>(r.get(), "");
	}

	template<typename _Ty>
	inline _Ty toCpp(PyObject* obj)
	{
		if (!obj) throw ConversionFail{ "cannot convert null pointer into appropriate C++ type" };
		_Ty v;
		if (!ValueBuilder<_Ty>{}._toCpp(obj, v)) throw ConversionFail{ "cannot convert " + reprWithNestedError(obj) + " into appropriate C++ type" };
		return v;
	}

	template<typename _Ty>
	inline bool toCpp(PyObject* obj, _Ty& out)
	{
		if (!obj) return false;
		return ValueBuilder<_Ty>{}._toCpp(obj, out);
	}

	template<typename _Ty>
	struct ValueBuilder<_Ty,
		typename std::enable_if<std::is_integral<_Ty>::value || std::is_enum<_Ty>::value>::type>
	{
		UniqueObj operator()(_Ty v)
		{
			return UniqueObj{ PyLong_FromLongLong(v) };
		}

		bool _toCpp(PyObject* obj, _Ty& out)
		{
			long long v = PyLong_AsLongLong(obj);
			if (v == -1 && PyErr_Occurred()) return false;
			out = (_Ty)v;
			return true;
		}
	};

	template<typename _Ty>
	struct ValueBuilder<_Ty,
		typename std::enable_if<std::is_floating_point<_Ty>::value>::type>
	{
		UniqueObj operator()(_Ty v)
		{
			return UniqueObj{ PyFloat_FromDouble(v) };
		}

		bool _toCpp(PyObject* obj, _Ty& out)
		{
			double v = PyFloat_AsDouble(obj);
			if (v == -1 && PyErr_Occurred()) return false;
			out = (_Ty)v;
			return true;
		}
	};

	template<>
	struct ValueBuilder<std::string>
	{
		UniqueObj operator()(const std::string& v)
		{
			return UniqueObj{ PyUnicode_FromStringAndSize(v.data(), v.size()) };
		}

		bool _toCpp(PyObject* obj, std::string& out)
		{
			Py_ssize_t size;
			const char* str = PyUnicode_AsUTF8AndSize(obj, &size);
			if (!str) return false;
			out = { str, str + size };
			return true;
		}
	};

	template<>
	struct ValueBuilder<std::u16string>
	{
		UniqueObj operator()(const std::u16string& v)
		{
			return UniqueObj{ PyUnicode_DecodeUTF16((const char*)v.data(), v.size() * 2, nullptr, nullptr) };
		}

		bool _toCpp(PyObject* obj, std::u16string& out)
		{
			UniqueObj uobj{ PyUnicode_FromObject(obj) };
			if (!uobj) return false;
			size_t len = PyUnicode_GET_LENGTH(uobj.get());

			switch (PyUnicode_KIND(uobj.get()))
			{
			case PyUnicode_1BYTE_KIND:
			{
				auto* p = PyUnicode_1BYTE_DATA(uobj.get());
				out.resize(len);
				std::copy(p, p + len, &out[0]);
				break;
			}
			case PyUnicode_2BYTE_KIND:
			{
				auto* p = PyUnicode_2BYTE_DATA(uobj.get());
				out.resize(len);
				std::copy(p, p + len, &out[0]);
				break;
			}
			case PyUnicode_4BYTE_KIND:
			{
				auto* p = PyUnicode_4BYTE_DATA(uobj.get());
				out.reserve(len);
				for (size_t i = 0; i < len; ++i)
				{
					auto c = p[i];
					if (c < 0x10000)
					{
						out.push_back(c);
					}
					else
					{
						out.push_back(0xD800 - (0x10000 >> 10) + (c >> 10));
						out.push_back(0xDC00 + (c & 0x3FF));
					}
				}
				break;
			}
			default:
				return false;
			}
			return true;
		}
	};

	template<>
	struct ValueBuilder<StringWithOffset<std::u16string>>
	{
		bool _toCpp(PyObject* obj, StringWithOffset<std::u16string>& out)
		{
			UniqueObj uobj{ PyUnicode_FromObject(obj) };
			if (!uobj) return false;
			size_t len = PyUnicode_GET_LENGTH(uobj.get());

			switch (PyUnicode_KIND(uobj.get()))
			{
			case PyUnicode_1BYTE_KIND:
			{
				auto* p = PyUnicode_1BYTE_DATA(uobj.get());
				out.str.resize(len);
				std::copy(p, p + len, &out.str[0]);
				out.offsets.resize(len + 1);
				std::iota(out.offsets.begin(), out.offsets.end(), 0);
				break;
			}
			case PyUnicode_2BYTE_KIND:
			{
				auto* p = PyUnicode_2BYTE_DATA(uobj.get());
				out.str.resize(len);
				std::copy(p, p + len, &out.str[0]);
				out.offsets.resize(len + 1);
				std::iota(out.offsets.begin(), out.offsets.end(), 0);
				break;
			}
			case PyUnicode_4BYTE_KIND:
			{
				auto* p = PyUnicode_4BYTE_DATA(uobj.get());
				out.str.reserve(len);
				out.offsets.reserve(len);
				for (size_t i = 0; i < len; ++i)
				{
					auto c = p[i];
					if (c < 0x10000)
					{
						out.offsets.emplace_back(out.str.size());
						out.str.push_back(c);
					}
					else
					{
						out.offsets.emplace_back(out.str.size());
						out.str.push_back(0xD800 - (0x10000 >> 10) + (c >> 10));
						out.str.push_back(0xDC00 + (c & 0x3FF));
					}
				}
				out.offsets.emplace_back(out.str.size());
				break;
			}
			default:
				return false;
			}
			return true;
		}
	};

	template<>
	struct ValueBuilder<const char*>
	{
		UniqueObj operator()(const char* v)
		{
			return UniqueObj{ PyUnicode_FromString(v) };
		}

		bool _toCpp(PyObject* obj, const char*& out)
		{
			const char* p = PyUnicode_AsUTF8(obj);
			if (!p) return false;
			out = p;
			return true;
		}
	};

	template<size_t len>
	struct ValueBuilder<char[len]>
	{
		UniqueObj operator()(const char(&v)[len])
		{
			return UniqueObj{ PyUnicode_FromStringAndSize(v, len - 1) };
		}
	};

	template<>
	struct ValueBuilder<bool>
	{
		UniqueObj operator()(bool v)
		{
			return UniqueObj{ PyBool_FromLong(v) };
		}

		bool _toCpp(PyObject* obj, bool& out)
		{
			if (!obj) return false;
			out = !!PyObject_IsTrue(obj);
			return true;
		}
	};

	template<>
	struct ValueBuilder<std::nullptr_t>
	{
		UniqueObj operator()(std::nullptr_t)
		{
			Py_INCREF(Py_None);
			return UniqueObj{ Py_None };
		}
	};

	template<>
	struct ValueBuilder<PyObject*>
	{
		UniqueObj operator()(PyObject* v)
		{
			if (!v) v = Py_None;
			Py_INCREF(v);
			return UniqueObj{ v };
		}

		bool _toCpp(PyObject* obj, PyObject*& out)
		{
			out = obj;
			return true;
		}
	};

	template<typename Ty>
	struct ValueBuilder<UniqueCObj<Ty>>
	{
		UniqueObj operator()(UniqueCObj<Ty>&& v)
		{
			if (v)
			{
				Py_INCREF(v.get());
				return UniqueObj{ (PyObject*)v.get() };
			}
			else
			{
				Py_INCREF(Py_None);
				return UniqueObj{ Py_None };
			}
		}

		UniqueObj operator()(const UniqueCObj<Ty>& v)
		{
			if (v)
			{
				Py_INCREF(v.get());
				return UniqueObj{ (PyObject*)v.get() };
			}
			else
			{
				Py_INCREF(Py_None);
				return UniqueObj{ Py_None };
			}
		}

		bool _toCpp(PyObject* obj, UniqueCObj<Ty>& out);
	};
	
	template<typename Ty>
	struct ValueBuilder<SharedCObj<Ty>>
	{
		UniqueObj operator()(SharedCObj<Ty>&& v)
		{
			if (v)
			{
				Py_INCREF(v);
				return UniqueObj{ (PyObject*)v.get() };
			}
			else
			{
				Py_INCREF(Py_None);
				return UniqueObj{ Py_None };
			}
		}

		UniqueObj operator()(const SharedCObj<Ty>& v)
		{
			if (v)
			{
				Py_INCREF(v);
				return UniqueObj{ (PyObject*)v.get() };
			}
			else
			{
				Py_INCREF(Py_None);
				return UniqueObj{ Py_None };
			}
		}
	};

	template<typename _Ty1, typename _Ty2>
	struct ValueBuilder<std::pair<_Ty1, _Ty2>>
	{
		UniqueObj operator()(const std::pair<_Ty1, _Ty2>& v)
		{
			UniqueObj ret{ PyTuple_New(2) };
			size_t id = 0;
			PyTuple_SET_ITEM(ret.get(), id++, buildPyValue(std::get<0>(v)).release());
			PyTuple_SET_ITEM(ret.get(), id++, buildPyValue(std::get<1>(v)).release());
			return ret;
		}

		bool _toCpp(PyObject* obj, std::pair<_Ty1, _Ty2>& out)
		{
			if (Py_SIZE(obj) != 2) throw ConversionFail{ "input is not tuple with len=2" };
			if (!toCpp<_Ty1>(UniqueObj{ PySequence_ITEM(obj, 0) }.get(), out.first)) return false;
			if (!toCpp<_Ty2>(UniqueObj{ PySequence_ITEM(obj, 1) }.get(), out.second)) return false;
			return true;
		}
	};

	template<typename... _Tys>
	struct ValueBuilder<std::tuple<_Tys...>>
	{
	private:
		void setValue(PyObject* o, const std::tuple<_Tys...>& v, std::integer_sequence<size_t>)
		{
		}

		template<size_t i, size_t ...rest>
		void setValue(PyObject* o, const std::tuple<_Tys...>& v, std::integer_sequence<size_t, i, rest...>)
		{
			PyTuple_SET_ITEM(o, i, buildPyValue(std::get<i>(v)).release());
			return setValue(o, v, std::integer_sequence<size_t, rest...>{});
		}

		template<size_t n, size_t ...idx>
		bool getValue(PyObject* o, std::tuple<_Tys...>& out, std::integer_sequence<size_t, n, idx...>)
		{
			if (!toCpp<typename std::tuple_element<n, std::tuple<_Tys...>>::type>(UniqueObj{ PySequence_ITEM(o, n) }.get(), std::get<n>(out))) return false;
			return getValue(o, out, std::integer_sequence<size_t, idx...>{});
		}

		bool getValue(PyObject* o, std::tuple<_Tys...>& out, std::integer_sequence<size_t>)
		{
			return true;
		}

	public:
		UniqueObj operator()(const std::tuple<_Tys...>& v)
		{
			UniqueObj ret{ PyTuple_New(sizeof...(_Tys)) };
			size_t id = 0;
			setValue(ret.get(), v, std::make_index_sequence<sizeof...(_Tys)>{});
			return ret;
		}

		bool _toCpp(PyObject* obj, std::tuple<_Tys...>& out)
		{
			if (Py_SIZE(obj) != sizeof...(_Tys)) return false;
			getValue(obj, out, std::make_index_sequence<sizeof...(_Tys)>{});
			return true;
		}
	};

	template<typename _Ty1, typename _Ty2>
	struct ValueBuilder<std::unordered_map<_Ty1, _Ty2>>
	{
		UniqueObj operator()(const std::unordered_map<_Ty1, _Ty2>& v)
		{
			UniqueObj ret{ PyDict_New() };
			for (auto& p : v)
			{
				if (PyDict_SetItem(ret.get(), buildPyValue(p.first).get(), buildPyValue(p.second).get())) return UniqueObj{ nullptr };
			}
			return ret;
		}

		bool _toCpp(PyObject* obj, std::unordered_map<_Ty1, _Ty2>& out)
		{
			PyObject* key, * value;
			Py_ssize_t pos = 0;
			while (PyDict_Next(obj, &pos, &key, &value)) 
			{
				_Ty1 k;
				_Ty2 v;
				if (!toCpp<_Ty1>(key, k)) return false;
				if (!toCpp<_Ty2>(value, v)) return false;
				out.emplace(std::move(k), std::move(v));
			}
			if (PyErr_Occurred()) return false;
			return true;
		}
	};

#if __cplusplus >= 201700L
	template<typename _Ty>
	struct ValueBuilder<std::optional<_Ty>>
	{
		UniqueObj operator()(const std::optional<_Ty>& v)
		{
			if (v) return buildPyValue(*v);
			return buildPyValue(nullptr);
		}

		bool _toCpp(PyObject* obj, std::optional<_Ty>& out)
		{
			if (obj != Py_None)
			{
				_Ty v;
				if (!toCpp<_Ty>(obj, v)) return false;
				out = std::move(v);
				return true;
			}
			out = {};
			return true;
		}
	};

	template<typename Ty, typename... Ts>
	struct ValueBuilder<std::variant<Ty, Ts...>>
	{
		UniqueObj operator()(const std::variant<Ty, Ts...>& v)
		{
			return std::visit([](auto&& t)
			{
				return py::buildPyValue(std::forward<decltype(t)>(t));
			}, v);
		}

		bool _toCpp(PyObject* obj, std::variant<Ty, Ts...>& out)
		{
			Ty v;
			if (toCpp<Ty>(obj, v))
			{
				out = std::move(v);
				return true;
			}

			if constexpr (sizeof...(Ts) > 0)
			{
				std::variant<Ts...> v2;
				if (toCpp<std::variant<Ts...>>(obj, v2))
				{
					out = std::visit([](auto&& t) -> std::variant<Ty, Ts...>
					{
						return std::forward<decltype(t)>(t);
					}, std::move(v2));
					return true;
				}
			}
			else
			{
			}
			return false;
		}
	};

#endif

#ifdef USE_NUMPY
	namespace detail
	{
		template<typename _Ty>
		struct NpyType
		{
			enum {
				npy_type = -1,
			};
		};

		template<>
		struct NpyType<int8_t>
		{
			enum {
				type = NPY_INT8,
				signed_type = type,
				npy_type = type,
			};
		};

		template<>
		struct NpyType<uint8_t>
		{
			enum {
				type = NPY_UINT8,
				signed_type = NPY_INT8,
				npy_type = type,
			};
		};

		template<>
		struct NpyType<int16_t>
		{
			enum {
				type = NPY_INT16,
				signed_type = type,
				npy_type = type,
			};
		};

		template<>
		struct NpyType<uint16_t>
		{
			enum {
				type = NPY_UINT16,
				signed_type = NPY_INT16,
				npy_type = type,
			};
		};

		template<>
		struct NpyType<int32_t>
		{
			enum {
				type = NPY_INT32,
				signed_type = type,
				npy_type = type,
			};
		};

		template<>
		struct NpyType<uint32_t>
		{
			enum {
				type = NPY_UINT32,
				signed_type = NPY_INT32,
				npy_type = type,
			};
		};

		template<>
		struct NpyType<int64_t>
		{
			enum {
				type = NPY_INT64,
				signed_type = type,
				npy_type = type,
			};
		};

		template<>
		struct NpyType<uint64_t>
		{
			enum {
				type = NPY_UINT64,
				signed_type = NPY_INT64,
				npy_type = type,
			};
		};

#ifdef __APPLE__
		template<>
		struct NpyType<long>
		{
			enum {
				type = NPY_INT64,
				signed_type = type,
				npy_type = type,
			};
		};

		template<>
		struct NpyType<unsigned long>
		{
			enum {
				type = NPY_UINT64,
				signed_type = NPY_INT64,
				npy_type = type,
			};
		};
#endif

		template<>
		struct NpyType<float>
		{
			enum {
				type = NPY_FLOAT,
				signed_type = type,
				npy_type = type,
			};
		};

		template<>
		struct NpyType<double>
		{
			enum {
				type = NPY_DOUBLE,
				signed_type = type,
				npy_type = type,
			};
		};
	}

	struct cast_to_signed_t {};
	static constexpr cast_to_signed_t cast_to_signed{};

	template<typename _Ty>
	struct numpy_able : std::integral_constant<bool, std::is_arithmetic<_Ty>::value> {};

#else
	template<typename _Ty>
	struct numpy_able : std::false_type {};
#endif

	template<class _Ty>
	struct numpy_pair_test : std::false_type {};

	template<class _Ty>
	struct numpy_pair_test<std::pair<_Ty, _Ty>> : numpy_able<_Ty> {};

	struct force_list_t {};
	static constexpr force_list_t force_list{};

#ifdef USE_NUMPY
	template<typename _Ty>
	struct ValueBuilder<std::vector<_Ty>,
		typename std::enable_if<numpy_able<_Ty>::value>::type>
	{
		UniqueObj operator()(const std::vector<_Ty>& v)
		{
			npy_intp size = v.size();
			UniqueObj obj{ PyArray_EMPTY(1, &size, detail::NpyType<_Ty>::type, 0) };
			std::memcpy(PyArray_DATA((PyArrayObject*)obj.get()), v.data(), sizeof(_Ty) * size);
			return obj;
		}

		bool _toCpp(PyObject* obj, std::vector<_Ty>& out)
		{
			if (detail::NpyType<_Ty>::npy_type >= 0 && PyArray_Check(obj) && PyArray_TYPE((PyArrayObject*)obj) == detail::NpyType<_Ty>::npy_type)
			{
				_Ty* ptr = (_Ty*)PyArray_GETPTR1((PyArrayObject*)obj, 0);
				out = std::vector<_Ty>{ ptr, ptr + PyArray_Size(obj) };
				return true;
			}
			else
			{
				UniqueObj iter{ PyObject_GetIter(obj) }, item;
				if (!iter) return false;
				std::vector<_Ty> v;
				while ((item = UniqueObj{ PyIter_Next(iter.get()) }))
				{
					_Ty i;
					if (!toCpp<_Ty>(item.get(), i)) return false;
					v.emplace_back(std::move(i));
				}
				if (PyErr_Occurred())
				{
					return false;
				}
				out = std::move(v);
				return true;
			}
		}
	};

	template<typename _Ty>
	struct ValueBuilder<std::vector<std::pair<_Ty, _Ty>>,
		typename std::enable_if<numpy_able<_Ty>::value>::type>
	{
		UniqueObj operator()(const std::vector<std::pair<_Ty, _Ty>>& v)
		{
			npy_intp size[2] = { (npy_intp)v.size(), 2 };
			UniqueObj obj{ PyArray_EMPTY(2, size, detail::NpyType<_Ty>::type, 0) };
			std::memcpy(PyArray_DATA((PyArrayObject*)obj.get()), v.data(), sizeof(_Ty) * v.size() * 2);
			return obj;
		}
	};
#endif

	template<typename _Ty>
	struct ValueBuilder<std::vector<_Ty>,
		typename std::enable_if<!numpy_able<_Ty>::value && !numpy_pair_test<_Ty>::value>::type>
	{
		UniqueObj operator()(const std::vector<_Ty>& v)
		{
			UniqueObj ret{ PyList_New(v.size()) };
			size_t id = 0;
			for (auto& e : v)
			{
				PyList_SET_ITEM(ret.get(), id++, buildPyValue(e).release());
			}
			return ret;
		}

		bool _toCpp(PyObject* obj, std::vector<_Ty>& out)
		{
			UniqueObj iter{ PyObject_GetIter(obj) }, item;
			if (!iter) return false;
			std::vector<_Ty> v;
			while ((item = UniqueObj{ PyIter_Next(iter.get()) }))
			{
				_Ty i;
				if (!toCpp<_Ty>(item.get(), i)) return false;
				v.emplace_back(std::move(i));
			}
			if (PyErr_Occurred())
			{
				return false;
			}
			out = std::move(v);
			return true;
		}
	};

	template<typename T, typename Out, typename Msg>
	inline void transform(PyObject* iterable, Out out, Msg&& failMsg)
	{
		if (!iterable) throw ConversionFail{ std::forward<Msg>(failMsg) };
		UniqueObj iter{ PyObject_GetIter(iterable) }, item;
		if (!iter) throw ConversionFail{ std::forward<Msg>(failMsg) };
		while ((item = UniqueObj{ PyIter_Next(iter.get()) }))
		{
			*out++ = toCpp<T>(item);
		}
		if (PyErr_Occurred())
		{
			throw ConversionFail{ std::forward<Msg>(failMsg) };
		}
	}

	template<typename T, typename Fn, typename Msg>
	inline void foreach(PyObject* iterable, Fn&& fn, Msg&& failMsg)
	{
		if (!iterable) throw ConversionFail{ std::forward<Msg>(failMsg) };
		UniqueObj iter{ PyObject_GetIter(iterable) }, item;
		if (!iter) throw ConversionFail{ std::forward<Msg>(failMsg) };
		try
		{
			while ((item = UniqueObj{ PyIter_Next(iter.get()) }))
			{
				fn(toCpp<T>(item.get()));
			}
		}
		catch (const ForeachFailed&)
		{
			throw ConversionFail{ std::forward<Msg>(failMsg) };
		}

		if (PyErr_Occurred())
		{
			throw ExcPropagation{};
		}
	}

#if __cplusplus >= 201701L
	template<typename T, typename Fn, typename Msg>
	inline void foreachVisit(PyObject* iterable, Fn&& fn, Msg&& failMsg)
	{
		if (!iterable) throw ConversionFail{ std::forward<Msg>(failMsg) };
		UniqueObj iter{ PyObject_GetIter(iterable) }, item;
		if (!iter) throw ConversionFail{ std::forward<Msg>(failMsg) };
		try
		{
			while ((item = UniqueObj{ PyIter_Next(iter.get()) }))
			{
				std::visit(fn, toCpp<T>(item.get()));
			}
		}
		catch (const ForeachFailed&)
		{
			throw ConversionFail{ std::forward<Msg>(failMsg) };
		}

		if (PyErr_Occurred())
		{
			throw ExcPropagation{};
		}
	}
#endif

	template<typename T, typename Fn, typename Msg>
	inline void foreachWithPy(PyObject* iterable, Fn&& fn, Msg&& failMsg)
	{
		if (!iterable) throw ConversionFail{ std::forward<Msg>(failMsg) };
		UniqueObj iter{ PyObject_GetIter(iterable) }, item;
		if (!iter) throw ConversionFail{ std::forward<Msg>(failMsg) };
		try
		{
			while ((item = UniqueObj{ PyIter_Next(iter.get()) }))
			{
				fn(toCpp<T>(item.get()), item.get());
			}
		}
		catch (const ForeachFailed&)
		{
			throw ConversionFail{ std::forward<Msg>(failMsg) };
		}

		if (PyErr_Occurred())
		{
			throw ConversionFail{ std::forward<Msg>(failMsg) };
		}
	}

#ifdef USE_NUMPY
	template<typename _Ty>
	inline typename std::enable_if<numpy_able<_Ty>::value, UniqueObj>::type
		buildPyValue(const std::vector<_Ty>& v, cast_to_signed_t)
	{
		npy_intp size = v.size();
		UniqueObj obj{ PyArray_EMPTY(1, &size, detail::NpyType<_Ty>::signed_type, 0) };
		std::memcpy(PyArray_DATA((PyArrayObject*)obj.get()), v.data(), sizeof(_Ty) * size);
		return obj;
	}
#endif

	template<typename _Ty>
	inline typename std::enable_if<
		!numpy_able<typename std::iterator_traits<_Ty>::value_type>::value,
		UniqueObj
	>::type buildPyValue(_Ty first, _Ty last)
	{
		UniqueObj ret{ PyList_New(std::distance(first, last)) };
		size_t id = 0;
		for (; first != last; ++first)
		{
			PyList_SET_ITEM(ret.get(), id++, buildPyValue(*first).release());
		}
		return ret;
	}

	template<typename _Ty>
	inline UniqueObj buildPyValue(const std::vector<_Ty>& v, force_list_t)
	{
		UniqueObj ret{ PyList_New(v.size()) };
		for (size_t i = 0; i < v.size(); ++i)
		{
			PyList_SET_ITEM(ret.get(), i, buildPyValue(v[i]).release());
		}
		return ret;
	}

	template<typename _Ty, typename _Tx>
	inline typename std::enable_if<
		!numpy_able<
		typename std::result_of<_Tx(typename std::iterator_traits<_Ty>::value_type)>::type
		>::value,
		UniqueObj
	>::type buildPyValueTransform(_Ty first, _Ty last, _Tx tx)
	{
		UniqueObj ret{ PyList_New(std::distance(first, last)) };
		size_t id = 0;
		for (; first != last; ++first)
		{
			PyList_SET_ITEM(ret.get(), id++, buildPyValue(tx(*first)).release());
		}
		return ret;
	}

	template<typename _Ty, typename _Tx>
	inline UniqueObj buildPyValueTransform(_Ty&& container, _Tx tx)
	{
		return buildPyValueTransform(std::begin(container), std::end(container), tx);
	}

#ifdef USE_NUMPY
	template<typename _Ty>
	inline typename std::enable_if<
		numpy_able<typename std::iterator_traits<_Ty>::value_type>::value,
		UniqueObj
	>::type buildPyValue(_Ty first, _Ty last)
	{
		using value_type = typename std::iterator_traits<_Ty>::value_type;
		npy_intp size = std::distance(first, last);
		UniqueObj ret{ PyArray_EMPTY(1, &size, detail::NpyType<value_type>::type, 0) };
		size_t id = 0;
		for (; first != last; ++first, ++id)
		{
			*(value_type*)PyArray_GETPTR1((PyArrayObject*)ret.get(), id) = *first;
		}
		return ret;
	}

	template<typename _Ty, typename _Tx>
	inline typename std::enable_if<
		numpy_able<
		typename std::result_of<_Tx(typename std::iterator_traits<_Ty>::value_type)>::type
		>::value,
		UniqueObj
	>::type buildPyValueTransform(_Ty first, _Ty last, _Tx tx)
	{
		using value_type = decltype(tx(*first));
		npy_intp size = std::distance(first, last);
		UniqueObj ret{ PyArray_EMPTY(1, &size, detail::NpyType<value_type>::type, 0) };
		size_t id = 0;
		for (; first != last; ++first, ++id)
		{
			*(value_type*)PyArray_GETPTR1((PyArrayObject*)ret.get(), id) = tx(*first);
		}
		return ret;
	}
#endif

	namespace detail
	{
		inline void setDictItem(PyObject* dict, const char** keys)
		{

		}

		template<typename _Ty, typename... _Rest>
		inline void setDictItem(PyObject* dict, const char** keys, _Ty&& value, _Rest&& ... rest)
		{
			{
				UniqueObj v{ buildPyValue(std::forward<_Ty>(value)) };
				PyDict_SetItemString(dict, keys[0], v.get());
			}
			return setDictItem(dict, keys + 1, std::forward<_Rest>(rest)...);
		}

		inline void setDictItem(PyObject* dict, const UniqueObj* keys)
		{

		}

		template<typename _Ty, typename... _Rest>
		inline void setDictItem(PyObject* dict, const UniqueObj* keys, _Ty&& value, _Rest&& ... rest)
		{
			{
				UniqueObj v{ buildPyValue(std::forward<_Ty>(value)) };
				PyDict_SetItem(dict, keys[0].get(), v.get());
			}
			return setDictItem(dict, keys + 1, std::forward<_Rest>(rest)...);
		}

		template<typename _Ty>
		struct IsNull
		{
			bool operator()(const _Ty& v)
			{
				return false;
			}
		};

		template<typename _Ty>
		struct IsNull<_Ty*>
		{
			bool operator()(_Ty* v)
			{
				return !v;
			}
		};

		template<>
		struct IsNull<std::nullptr_t>
		{
			bool operator()(std::nullptr_t v)
			{
				return true;
			}
		};

		template<class _Ty>
		inline bool isNull(_Ty&& v)
		{
			return IsNull<typename std::remove_reference<_Ty>::type>{}(std::forward<_Ty>(v));
		}

		inline void setDictItemSkipNull(PyObject* dict, const char** keys)
		{

		}

		template<typename _Ty, typename... _Rest>
		inline void setDictItemSkipNull(PyObject* dict, const char** keys, _Ty&& value, _Rest&& ... rest)
		{
			if (!isNull(value))
			{
				UniqueObj v{ buildPyValue(value) };
				PyDict_SetItemString(dict, keys[0], v.get());
			}
			return setDictItemSkipNull(dict, keys + 1, std::forward<_Rest>(rest)...);
		}

		inline void setDictItemSkipNull(PyObject* dict, const UniqueObj* keys)
		{

		}

		template<typename _Ty, typename... _Rest>
		inline void setDictItemSkipNull(PyObject* dict, const UniqueObj* keys, _Ty&& value, _Rest&& ... rest)
		{
			if (!isNull(value))
			{
				UniqueObj v{ buildPyValue(value) };
				PyDict_SetItem(dict, keys[0].get(), v.get());
			}
			return setDictItemSkipNull(dict, keys + 1, std::forward<_Rest>(rest)...);
		}

		template<size_t _n>
		inline void setTupleItem(PyObject* tuple)
		{
		}

		template<size_t _n, typename _Ty, typename... _Rest>
		inline void setTupleItem(PyObject* tuple, _Ty&& first, _Rest&&... rest)
		{
			PyTuple_SET_ITEM(tuple, _n, buildPyValue(std::forward<_Ty>(first)).release());
			return setTupleItem<_n + 1>(tuple, std::forward<_Rest>(rest)...);
		}
	}

	template<typename... _Rest>
	inline UniqueObj buildPyDict(const char** keys, _Rest&&... rest)
	{
		UniqueObj dict{ PyDict_New() };
		detail::setDictItem(dict, keys, std::forward<_Rest>(rest)...);
		return dict;
	}

	template<typename... _Rest>
	inline UniqueObj buildPyDictSkipNull(const char** keys, _Rest&&... rest)
	{
		UniqueObj dict{ PyDict_New() };
		detail::setDictItemSkipNull(dict, keys, std::forward<_Rest>(rest)...);
		return dict;
	}

	template<typename... _Rest>
	inline UniqueObj buildPyDict(const UniqueObj* keys, _Rest&&... rest)
	{
		UniqueObj dict{ PyDict_New() };
		detail::setDictItem(dict.get(), keys, std::forward<_Rest>(rest)...);
		return dict;
	}

	template<typename... _Rest>
	inline UniqueObj buildPyDictSkipNull(const UniqueObj* keys, _Rest&&... rest)
	{
		UniqueObj dict{ PyDict_New() };
		detail::setDictItemSkipNull(dict.get(), keys, std::forward<_Rest>(rest)...);
		return dict;
	}

	template<typename _Ty>
	inline void setPyDictItem(PyObject* dict, const char* key, _Ty&& value)
	{
		UniqueObj v{ buildPyValue(value) };
		PyDict_SetItemString(dict, key, v.get());
	}

	template<typename... _Rest>
	inline UniqueObj buildPyTuple(_Rest&&... rest)
	{
		UniqueObj tuple{ PyTuple_New(sizeof...(_Rest)) };
		detail::setTupleItem<0>(tuple.get(), std::forward<_Rest>(rest)...);
		return tuple;
	}

	template<typename Ty>
	class TypeWrapper;

	template<class Derived>
	struct CObject
	{
		PyObject_HEAD;

		static PyObject* _new(PyTypeObject* subtype, PyObject* args, PyObject* kwargs)
		{
			return handleExc([&]()
			{
				py::UniqueObj ret{ subtype->tp_alloc(subtype, 0) };
				new ((Derived*)ret.get()) Derived;
				return ret.release();
			});
		}

		static void dealloc(Derived* self)
		{
			self->~Derived();
			Py_TYPE(self)->tp_free((PyObject*)self);
		}

		using _InitArgs = std::tuple<>;

	private:
		template<class InitArgs, size_t ...idx>
		PY_STRONG_INLINE static void initFromPython(Derived* self, PyObject* args, std::index_sequence<idx...>)
		{
			*self = Derived{ toCpp<std::tuple_element_t<idx, InitArgs>>(PyTuple_GET_ITEM(args, idx))... };
		}

		static int init(Derived* self, PyObject* args, PyObject* kwargs)
		{
			return handleExc([&]() -> int
			{
				using InitArgs = typename Derived::_InitArgs;
				if constexpr (std::tuple_size_v<InitArgs> == 0)
				{
					if (args && PyTuple_GET_SIZE(args) != 0)
					{
						throw TypeError{ "function takes " + std::to_string(std::tuple_size_v<InitArgs>) + " arguments (" + std::to_string(PyTuple_GET_SIZE(args)) + " given)" };
					}
				}

				if (std::tuple_size_v<InitArgs> != PyTuple_GET_SIZE(args))
				{
					throw TypeError{ "function takes " + std::to_string(std::tuple_size_v<InitArgs>) + " arguments (" + std::to_string(PyTuple_GET_SIZE(args)) + " given)" };
				}
				if (kwargs)
				{
					throw TypeError{ "function takes positional arguments only" };
				}

				auto temp = self->ob_base;
				initFromPython<InitArgs>(self, args, std::make_index_sequence<std::tuple_size_v<InitArgs>>{});
				self->ob_base = temp;
				return 0;
			});
		}

		static constexpr const char* _name = "";
		static constexpr const char* _name_in_module = "";
		static constexpr const char* _doc = "";
		static constexpr int _flags = Py_TPFLAGS_DEFAULT;

		friend class TypeWrapper<Derived>;
	};

	template<class Derived, class RetTy, class Future = std::future<RetTy>>
	struct ResultIter : public CObject<Derived>
	{
		using ReturnTy = RetTy;
		using FutureTy = Future;
		UniqueObj inputIter;
		std::deque<Future> futures;
		std::deque<SharedObj> inputItems;
		bool echo = false;

		ResultIter() = default;
		ResultIter(ResultIter&&) = default;
		ResultIter& operator=(ResultIter&&) = default;

		ResultIter(const ResultIter&) = delete;
		ResultIter& operator=(const ResultIter&) = delete;

		~ResultIter()
		{
			waitQueue();
		}

		void waitQueue()
		{
			while (!futures.empty())
			{
				auto f = std::move(futures.front());
				futures.pop_front();
				f.get();
			}
		}

		py::UniqueCObj<Derived> iter() const
		{
			Py_INCREF(this);
			return py::UniqueCObj<Derived>{ static_cast<Derived*>(const_cast<ResultIter*>(this)) };
		}

		py::UniqueObj iternext()
		{
			if (!feed() && futures.empty()) throw py::ExcPropagation{};
			auto f = std::move(futures.front());
			futures.pop_front();
			if (echo)
			{
				auto input = std::move(inputItems.front());
				inputItems.pop_front();
				return buildPyTuple(static_cast<Derived*>(this)->buildPy(f.get()), input);
			}
			else
			{
				return static_cast<Derived*>(this)->buildPy(f.get());
			}
		}

		bool feed()
		{
			SharedObj item{ PyIter_Next(inputIter.get()) };
			if (!item)
			{
				if (PyErr_Occurred()) throw ExcPropagation{};
				return false;
			}
			if (echo) inputItems.emplace_back(item);
			futures.emplace_back(static_cast<Derived*>(this)->feedNext(std::move(item)));
			return true;
		}

		Future feedNext(py::SharedObj&& next)
		{
			return {};
		}

		UniqueObj buildPy(RetTy&& v)
		{
			return py::buildPyValue(std::move(v));
		}
	};

	class Module
	{
		std::map<const char*, PyTypeObject*> types;
		PyModuleDef def;
		PyObject* mod = nullptr;

	public:
		Module(const char* name, const char* doc)
		{
			def.m_base = PyModuleDef_HEAD_INIT;
			def.m_name = name;
			def.m_doc = doc;
			def.m_size = -1;
			def.m_methods = nullptr;
			def.m_slots = nullptr;
			def.m_traverse = nullptr;
			def.m_clear = nullptr;
			def.m_free = nullptr;
		}

		template<class Fn>
		Module(const char* name, const char* doc, Fn&& fn)
			: Module{ name, doc }
		{
			fn(def);
		}

		PyObject* init()
		{
			mod = PyModule_Create(&def);
			addToModule();
			return mod;
		}

		void registerType(PyTypeObject* type, const char* name)
		{
			types[name] = type;
		}

		void addToModule()
		{
			for (auto& p : types)
			{
				if (PyType_Ready(p.second) < 0) throw ExcPropagation{};
				Py_INCREF(p.second);
				PyModule_AddObject(mod, p.first, (PyObject*)p.second);
			}
		}
	};

	template<typename Ty>
	class TypeWrapper
	{
	public:
		static PyTypeObject obj;

		template<class Fn>
		TypeWrapper(Module& tm, Fn&& fn);

		static constexpr PyObject* getTypeObj() { return (PyObject*)&obj; }
	};

	template<typename Ty>
	PyTypeObject* Type = &TypeWrapper<Ty>::obj;

	template<typename Ty> PyTypeObject TypeWrapper<Ty>::obj = {
		PyVarObject_HEAD_INIT(nullptr, 0)
	};

	template<class Ty = PyObject>
	inline UniqueCObj<Ty> makeNewObject(PyTypeObject* type)
	{
		UniqueCObj<Ty> ret{ (Ty*)type->tp_new(type, buildPyTuple().get(), nullptr)};
		return ret;
	}

	template<class Ty>
	inline UniqueCObj<Ty> makeNewObject()
	{
		return makeNewObject<Ty>(Type<Ty>);
	}

	template<class Ty>
	bool ValueBuilder<UniqueCObj<Ty>>::_toCpp(PyObject* obj, UniqueCObj<Ty>& out)
	{
		out = UniqueCObj<Ty>{ (Ty*)obj };
		if (!std::is_same_v<Ty, PyObject> && !PyObject_IsInstance(obj, (PyObject*)Type<Ty>))
		{
			return false;
		}
		Py_INCREF(obj);
		return true;
	}

	class CustomExcHandler
	{
		static std::unordered_map<std::type_index, PyObject*> handlers;

	public:
		template<class CustomExc, class PyExc>
		static void add()
		{
			handlers[std::type_index(typeid(CustomExc))] = PyExc{ "" }.pytype();
		}

		static const std::unordered_map<std::type_index, PyObject*>& get()
		{
			return handlers;
		}
	};

	std::unordered_map<std::type_index, PyObject*> CustomExcHandler::handlers;

	namespace detail
	{
		inline void setPyError(PyObject* errType, const char* errMsg)
		{
			if (PyErr_Occurred())
			{
				PyObject* exc, * val, * tb, * val2;
				PyErr_Fetch(&exc, &val, &tb);
				PyErr_NormalizeException(&exc, &val, &tb);
				if (tb)
				{
					PyException_SetTraceback(val, tb);
					Py_DECREF(tb);
				}
				Py_DECREF(exc);
				PyObject* et = errType;
				val2 = PyObject_CallFunctionObjArgs(et, py::UniqueObj{ buildPyValue(errMsg) }.get(), nullptr);
				PyException_SetCause(val2, val);
				PyErr_SetObject(et, val2);
				Py_DECREF(val2);
			}
			else
			{
				PyErr_SetString(errType, errMsg);
			}
		}
	}

	template<typename _Fn>
	PY_STRONG_INLINE auto handleExc(_Fn&& fn)
		-> typename std::enable_if<std::is_pointer<decltype(fn())>::value, decltype(fn())>::type
	{
		try
		{
			return fn();
		}
		catch (const ExcPropagation&)
		{
		}
		catch (const BaseException& e)
		{
			detail::setPyError(e.pytype(), e.what());
		}
		catch (const std::exception& e)
		{
			auto customHandlers = CustomExcHandler{}.get();
			auto it = customHandlers.find(std::type_index(typeid(e)));
			if (it == customHandlers.end())
			{
				throw;
			}
			detail::setPyError(it->second, e.what());
		}
		/*catch (const std::exception& e)
		{
			std::cerr << "Uncaughted c++ exception: " << e.what() << std::endl;
			PyErr_SetString(PyExc_RuntimeError, e.what());
		}*/
		return nullptr;
	}

	template<typename _Fn>
	PY_STRONG_INLINE auto handleExc(_Fn&& fn)
		-> typename std::enable_if<std::is_same<decltype(fn()), UniqueObj>::value, decltype(fn())>::type
	{
		try
		{
			return fn();
		}
		catch (const ExcPropagation&)
		{
		}
		catch (const BaseException& e)
		{
			detail::setPyError(e.pytype(), e.what());
		}
		catch (const std::exception& e)
		{
			auto customHandlers = CustomExcHandler{}.get();
			auto it = customHandlers.find(std::type_index(typeid(e)));
			if (it == customHandlers.end())
			{
				throw;
			}
			detail::setPyError(it->second, e.what());
		}
		/*catch (const std::exception& e)
		{
			std::cerr << "Uncaughted c++ exception: " << e.what() << std::endl;
			PyErr_SetString(PyExc_RuntimeError, e.what());
		}*/
		return UniqueObj{ nullptr };
	}

	template<typename _Fn>
	PY_STRONG_INLINE auto handleExc(_Fn&& fn)
		-> typename std::enable_if<std::is_integral<decltype(fn())>::value, decltype(fn())>::type
	{
		try
		{
			return fn();
		}
		catch (const ExcPropagation&)
		{
		}
		catch (const BaseException& e)
		{
			detail::setPyError(e.pytype(), e.what());
		}
		catch (const std::exception& e)
		{
			auto customHandlers = CustomExcHandler{}.get();
			auto it = customHandlers.find(std::type_index(typeid(e)));
			if (it == customHandlers.end())
			{
				throw;
			}
			detail::setPyError(it->second, e.what());
		}
		/*catch (const std::exception& e)
		{
			std::cerr << "Uncaughted c++ exception: " << e.what() << std::endl;
			PyErr_SetString(PyExc_RuntimeError, e.what());
		}*/
		return -1;
	}

	namespace detail
	{
		template <typename T>
		struct IsFunctionObjectImpl
		{
		private:
			using Yes = char(&)[1];
			using No = char(&)[2];

			struct Fallback
			{
				void operator()();
			};

			struct Derived : T, Fallback
			{
			};

			template <typename U, U>
			struct Check;

			template <typename>
			static Yes Test(...);

			template <typename C>
			static No Test(Check<void(Fallback::*)(), &C::operator()>*);

		public:
			static constexpr bool value{ sizeof(Test<Derived>(0)) == sizeof(Yes) };
		};
	}

	template <typename T>
	struct IsFunctionObject : std::conditional<
		std::is_class<T>::value,
		detail::IsFunctionObjectImpl<T>,
		std::false_type
	>::type
	{
	};

	namespace detail
	{
		template <typename T>
		struct CppWrapperImpl;

		/* global function object */
		template <typename R, typename... Ts>
		struct CppWrapperImpl<R(Ts...)>
		{
			using Type = R(Ts...);
			using FunctionPointerType = R(*)(Ts...);
			using ReturnType = R;
			using ClassType = void;
			using ArgsTuple = std::tuple<Ts...>;

			template <std::size_t N>
			using Arg = typename std::tuple_element<N, ArgsTuple>::type;

			static const std::size_t nargs{ sizeof...(Ts) };

			template<Type func, size_t ...idx>
			static constexpr auto callFromPython(void*, PyObject* args, PyObject* kwargs, std::index_sequence<idx...>)
			{
				if (PyTuple_GET_SIZE(args) != std::tuple_size_v<ArgsTuple>)
				{
					throw TypeError{ "function takes " + std::to_string(std::tuple_size_v<ArgsTuple>) + " arguments (" + std::to_string(PyTuple_GET_SIZE(args)) + " given)" };
				}
				if (kwargs)
				{
					throw TypeError{ "function takes positional arguments only" };
				}

				return func(toCpp<std::remove_const_t<std::remove_reference_t<Arg<idx>>>>(PyTuple_GET_ITEM(args, idx))...);
			}
		};

		/* global function pointer */
		template <typename R, typename... Ts>
		struct CppWrapperImpl<R(*)(Ts...)>
		{
			using Type = R(*)(Ts...);
			using FunctionPointerType = R(*)(Ts...);
			using ReturnType = R;
			using ClassType = void;
			using ArgsTuple = std::tuple<Ts...>;

			template <std::size_t N>
			using Arg = typename std::tuple_element<N, ArgsTuple>::type;

			static const std::size_t nargs{ sizeof...(Ts) };

			template<Type func, size_t ...idx>
			static constexpr auto call(void*, PyObject* args, PyObject* kwargs, std::index_sequence<idx...>)
			{
				if (PyTuple_GET_SIZE(args) != std::tuple_size_v<ArgsTuple>)
				{
					throw TypeError{ "function takes " + std::to_string(std::tuple_size_v<ArgsTuple>) + " arguments (" + std::to_string(PyTuple_GET_SIZE(args)) + " given)" };
				}
				if (kwargs)
				{
					throw TypeError{ "function takes positional arguments only" };
				}

				return func(toCpp<std::remove_const_t<std::remove_reference_t<Arg<idx>>>>(PyTuple_GET_ITEM(args, idx))...);
			}
		};

		/* member function pointer */
		template <typename C, typename R, typename... Ts>
		struct CppWrapperImpl<R(C::*)(Ts...)>
		{
			using Type = R(C::*)(Ts...);
			using FunctionPointerType = R(*)(C*, Ts...);
			using ReturnType = R;
			using ClassType = C;
			using ArgsTuple = std::tuple<Ts...>;

			template <std::size_t N>
			using Arg = typename std::tuple_element<N, ArgsTuple>::type;

			static constexpr std::size_t nargs{ sizeof...(Ts) };

			template<Type func, size_t ...idx>
			static constexpr auto call(ClassType* self, PyObject* args, PyObject* kwargs, std::index_sequence<idx...>)
			{
				if (PyTuple_GET_SIZE(args) != std::tuple_size_v<ArgsTuple>)
				{
					throw TypeError{ "function takes " + std::to_string(std::tuple_size_v<ArgsTuple>) + " arguments (" + std::to_string(PyTuple_GET_SIZE(args)) + " given)" };
				}
				if (kwargs)
				{
					throw TypeError{ "function takes positional arguments only" };
				}

				return (self->*func)(toCpp<std::remove_const_t<std::remove_reference_t<Arg<idx>>>>(PyTuple_GET_ITEM(args, idx))...);
			}

			template<Type func, size_t _nargs = nargs>
			static constexpr std::enable_if_t<_nargs == 0, ReturnType> len(ClassType* self)
			{
				static_assert(std::is_integral_v<ReturnType>, "len() must return integral type");
				return (self->*func)();
			}

			template<Type func, size_t _nargs = nargs>
			static constexpr std::enable_if_t<_nargs == 1, ReturnType> ssizearg(ClassType* self, Py_ssize_t idx)
			{
				static_assert(std::is_integral_v<Arg<0>>, "ssizearg() must take one integral argument");
				return (self->*func)(idx);
			}

			template<Type func, size_t _nargs = nargs>
			static constexpr std::enable_if_t<_nargs == 1, ReturnType> binary(ClassType* self, PyObject* arg)
			{
				return (self->*func)(toCpp<Arg<0>>(arg));
			}

			template<Type func, size_t _nargs = nargs>
			static constexpr std::enable_if_t<_nargs == 0, ReturnType> repr(ClassType* self)
			{
				return (self->*func)();
			}

			template<Type func, size_t _nargs = nargs>
			static constexpr std::enable_if_t<_nargs == 0, ReturnType> get(ClassType* self)
			{
				return (self->*func)();
			}

			template<Type func, size_t _nargs = nargs>
			static constexpr std::enable_if_t<_nargs == 1, int> set(ClassType* self, PyObject* val)
			{
				(self->*func)(toCpp<Arg<0>>(val));
				return 0;
			}
		};

		/* const member function pointer */
		template <typename C, typename R, typename... Ts>
		struct CppWrapperImpl<R(C::*)(Ts...) const>
		{
			using Type = R(C::*)(Ts...) const;
			using FunctionPointerType = R(*)(C*, Ts...);
			using ReturnType = R;
			using ClassType = C;
			using ArgsTuple = std::tuple<Ts...>;

			template <std::size_t N>
			using Arg = typename std::tuple_element<N, ArgsTuple>::type;

			static constexpr std::size_t nargs{ sizeof...(Ts) };

			template<Type func, size_t ...idx>
			static constexpr auto call(const ClassType* self, PyObject* args, PyObject* kwargs, std::index_sequence<idx...>)
			{
				if (PyTuple_GET_SIZE(args) != std::tuple_size_v<ArgsTuple>)
				{
					throw TypeError{ "function takes " + std::to_string(std::tuple_size_v<ArgsTuple>) + " arguments (" + std::to_string(PyTuple_GET_SIZE(args)) + " given)" };
				}
				if (kwargs)
				{
					throw TypeError{ "function takes positional arguments only" };
				}

				return (self->*func)(toCpp<std::remove_const_t<std::remove_reference_t<Arg<idx>>>>(PyTuple_GET_ITEM(args, idx))...);
			}

			template<Type func, size_t _nargs = nargs>
			static constexpr std::enable_if_t<_nargs == 0, ReturnType> len(const ClassType* self)
			{
				static_assert(std::is_integral_v<ReturnType>, "len() must return integral type");
				return (self->*func)();
			}

			template<Type func, size_t _nargs = nargs>
			static constexpr std::enable_if_t<_nargs == 1, ReturnType> ssizearg(const ClassType* self, Py_ssize_t idx)
			{
				static_assert(std::is_integral_v<Arg<0>>, "ssizearg() must take one integral argument");
				return (self->*func)(idx);
			}

			template<Type func, size_t _nargs = nargs>
			static constexpr std::enable_if_t<_nargs == 1, ReturnType> binary(ClassType* self, PyObject* arg)
			{
				return (self->*func)(toCpp<Arg<0>>(arg));
			}

			template<Type func, size_t _nargs = nargs>
			static constexpr std::enable_if_t<_nargs == 0, ReturnType> repr(const ClassType* self)
			{
				return (self->*func)();
			}

			template<Type func, size_t _nargs = nargs>
			static constexpr std::enable_if_t<_nargs == 0, ReturnType> get(const ClassType* self)
			{
				return (self->*func)();
			}
		};

		/* member variable pointer */
		template <typename C, typename R>
		struct CppWrapperImpl<R(C::*)>
		{
			using Type = R(C::*);
			using ReturnType = R;
			using ClassType = C;

			template<Type ptr>
			static constexpr const ReturnType& get(ClassType* self)
			{
				return self->*ptr;
			}

			template<Type ptr>
			static constexpr int set(ClassType* self, PyObject* val)
			{
				self->*ptr = toCpp<ReturnType>(val);
				return 0;
			}
		};

		template<class Base>
		struct CppWrapperInterface : public Base
		{
			using T = typename Base::Type;

			template<T func>
			static constexpr PyCFunction call()
			{
				return (PyCFunction)(PyCFunctionWithKeywords)[](PyObject* self, PyObject* args, PyObject* kwargs) -> PyObject*
				{
					return handleExc([&]() -> PyObject*
					{
						if constexpr (std::is_same_v<typename Base::ReturnType, void>)
						{
							Base::template call<func>((typename Base::ClassType*)self, args, kwargs, std::make_index_sequence<std::tuple_size_v<typename Base::ArgsTuple>>{});
							return buildPyValue(nullptr).release();
						}
						else
						{
							return buildPyValue(Base::template call<func>((typename Base::ClassType*)self, args, kwargs, std::make_index_sequence<std::tuple_size_v<typename Base::ArgsTuple>>{})).release();
						}
					});
				};
			}

			template<T ptr>
			static constexpr lenfunc len()
			{
				return (lenfunc)[](PyObject* self) -> Py_ssize_t
				{
					return handleExc([&]()
					{
						return Base::template len<ptr>((typename Base::ClassType*)self);
					});
				};
			}

			template<T ptr>
			static constexpr reprfunc repr()
			{
				return (reprfunc)[](PyObject* self) -> PyObject*
				{
					return handleExc([&]()
					{
						return buildPyValue(Base::template repr<ptr>((typename Base::ClassType*)self)).release();
					});
				};
			}

			template<T ptr>
			static constexpr ssizeargfunc ssizearg()
			{
				return (ssizeargfunc)[](PyObject* self, Py_ssize_t idx) -> PyObject*
				{
					return handleExc([&]()
					{
						return buildPyValue(Base::template ssizearg<ptr>((typename Base::ClassType*)self, idx)).release();
					});
				};
			}

			template<T ptr>
			static constexpr binaryfunc binary()
			{
				return (binaryfunc)[](PyObject* self, PyObject* arg) -> PyObject*
				{
					return handleExc([&]()
					{
						return buildPyValue(Base::template binary<ptr>((typename Base::ClassType*)self, arg)).release();
					});
				};
			}

			template<T ptr>
			static constexpr getter get()
			{
				return (getter)[](PyObject* self, void* closure) -> PyObject*
				{
					return handleExc([&]()
					{
						return buildPyValue(Base::template get<ptr>((typename Base::ClassType*)self)).release();
					});
				};
			}

			template<T ptr>
			static constexpr setter set()
			{
				return (setter)[](PyObject* self, PyObject* val, void* closure) -> int
				{
					return handleExc([&]()
					{
						return Base::template set<ptr>((typename Base::ClassType*)self, val);
					});
				};
			}
		};

#define _PY_DETAILED_TEST_MEMFN(name, func) \
		template <typename T>\
		class name\
		{\
			using one = char;\
			struct two { char x[2]; };\
			template <typename C> static one test(decltype(&C::func));\
			template <typename C> static two test(...);\
		public:\
			enum { value = sizeof(test<T>(0)) == sizeof(char) };\
		};

		_PY_DETAILED_TEST_MEMFN(HasRepr, repr);
		_PY_DETAILED_TEST_MEMFN(HasIter, iter);
		_PY_DETAILED_TEST_MEMFN(HasIterNext, iternext);
		_PY_DETAILED_TEST_MEMFN(HasLen, len);
	}

	template <typename T, typename = void>
	struct CppWrapper : detail::CppWrapperInterface<detail::CppWrapperImpl<T>>
	{
	};

	template <typename T>
	struct CppWrapper<T, typename std::enable_if<IsFunctionObject<T>::value>::type> :
		detail::CppWrapperInterface<detail::CppWrapperImpl<decltype(&T::operator())>>
	{
	};

	template<class Ty>
	template<class Fn>
	TypeWrapper<Ty>::TypeWrapper(Module& tm, Fn&& fn)
	{
		obj.tp_basicsize = sizeof(Ty);
		obj.tp_dealloc = (destructor)Ty::dealloc;
		obj.tp_new = (newfunc)Ty::_new;
		obj.tp_alloc = PyType_GenericAlloc;
		obj.tp_flags = Ty::_flags;
		obj.tp_name = Ty::_name;
		obj.tp_doc = Ty::_doc;
		obj.tp_init = (initproc)Ty::init;
		
		if constexpr (detail::HasIter<Ty>::value)
		{
			obj.tp_iter = CppWrapper<decltype(&Ty::iter)>::template repr<&Ty::iter>();
		}

		if constexpr (detail::HasIterNext<Ty>::value)
		{
			obj.tp_iternext = CppWrapper<decltype(&Ty::iternext)>::template repr<&Ty::iternext>();
		}

		if constexpr (detail::HasRepr<Ty>::value)
		{
			obj.tp_repr = CppWrapper<decltype(&Ty::repr)>::template repr<&Ty::repr>();
		}

		fn(obj);
		tm.registerType(&obj, Ty::_name_in_module);
	}
}

#define PY_METHOD(P) py::CppWrapper<decltype(P)>::call<P>()
#define PY_GETTER(P) py::CppWrapper<decltype(P)>::get<P>()
#define PY_SETTER(P) py::CppWrapper<decltype(P)>::set<P>()
#define PY_LENFUNC(P) py::CppWrapper<decltype(P)>::len<P>()
#define PY_SSIZEARGFUNC(P) py::CppWrapper<decltype(P)>::ssizearg<P>()
#define PY_BINARYFUNC(P) py::CppWrapper<decltype(P)>::binary<P>()
#define PY_REPRFUNC(P) py::CppWrapper<decltype(P)>::repr<P>()
