#include "LatticeGuide.h"

#include <algorithm>
#include <map>
#include <string_view>
#include <stdexcept>

using namespace std;

namespace kiwi
{
	namespace guide
	{
		namespace
		{
			// 잘못된 바이트가 있으면 nullopt(strict) 또는 그 바이트를 건너뛴다(ignore)
			optional<u32string> decodeUtf8(const string& s, bool ignoreErrors)
			{
				u32string out;
				size_t i = 0;
				while (i < s.size())
				{
					const auto c = (uint8_t)s[i];
					size_t len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
					char32_t cp = len == 1 ? c : len == 2 ? (c & 0x1F) : len == 3 ? (c & 0x0F) : (c & 0x07);
					bool ok = len > 0 && i + len <= s.size();
					for (size_t k = 1; ok && k < len; ++k)
					{
						const auto cc = (uint8_t)s[i + k];
						if ((cc >> 6) != 2) ok = false;
						else cp = (cp << 6) | (cc & 0x3F);
					}
					if (ok)
					{
						const char32_t minCp[] = { 0, 0, 0x80, 0x800, 0x10000 };
						ok = cp >= minCp[len] && cp <= 0x10FFFF && !(cp >= 0xD800 && cp <= 0xDFFF);
					}
					if (!ok)
					{
						if (!ignoreErrors) return nullopt;
						++i;
						continue;
					}
					out.push_back(cp);
					i += len;
				}
				return out;
			}

			string encodeUtf8(const u32string& s)
			{
				string out;
				for (auto cp : s)
				{
					if (cp < 0x80) out.push_back((char)cp);
					else if (cp < 0x800)
					{
						out.push_back((char)(0xC0 | (cp >> 6)));
						out.push_back((char)(0x80 | (cp & 0x3F)));
					}
					else if (cp < 0x10000)
					{
						out.push_back((char)(0xE0 | (cp >> 12)));
						out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
						out.push_back((char)(0x80 | (cp & 0x3F)));
					}
					else
					{
						out.push_back((char)(0xF0 | (cp >> 18)));
						out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
						out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
						out.push_back((char)(0x80 | (cp & 0x3F)));
					}
				}
				return out;
			}

			// 한글 음절만 초·중·종성으로 나눈다(Python unicodedata NFD는 다른 글자의 정규 분해도 하지만 격자 비용에서는 드물다)
			void appendJamo(u32string& out, char32_t c)
			{
				if (c >= 0xAC00 && c <= 0xD7A3)
				{
					const auto s = c - 0xAC00;
					out.push_back(0x1100 + s / 588);
					out.push_back(0x1161 + (s % 588) / 28);
					if (s % 28) out.push_back(0x11A7 + s % 28);
				}
				else out.push_back(c);
			}

			// 형태 bs의 글자 경계마다 그 뒤의 나머지(자모 편집으로 쓰던 형태를 마무리하는 후보)
			void addSuffixes(vector<string>& outs, const string& bs)
			{
				for (size_t a = 0; a < bs.size(); ++a)
				{
					if (((unsigned char)bs[a] & 0xC0) != 0x80) outs.emplace_back(bs.substr(a));
				}
			}

			u32string jamoOf(const u32string& s)
			{
				u32string out;
				for (auto c : s) appendJamo(out, c);
				return out;
			}

			string noSpace(const string& b)
			{
				string out;
				out.reserve(b.size());
				for (auto ch : b) if (ch != ' ') out.push_back(ch);
				return out;
			}

			bool startsWith(const string& s, const string& prefix)
			{
				return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
			}


			bool isSpaceLike(char32_t c)
			{
				return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
			}

			bool isBlankLike(char32_t c)
			{
				return isSpaceLike(c) || c == 0xA0 || (0x2000 <= c && c <= 0x200D) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F
					|| c == 0x2060 || c == 0x2800 || c == 0x3000 || c == 0x3164 || c == 0xFEFF;
			}

			// generative_msa special.model_form(f, True): 양쪽에 글자가 있는 _를 공백으로
			u32string modelForm(const u32string& f)
			{
				u32string out = f;
				for (size_t i = 1; i + 1 < f.size(); ++i)
				{
					if (f[i] != '_') continue;
					const bool l = f[i - 1] != '_' && !isSpaceLike(f[i - 1]);
					const bool r = f[i + 1] != '_' && !isSpaceLike(f[i + 1]);
					if (l && r) out[i] = ' ';
				}
				return out;
			}

			// 자모 편집은 한글 자모(초·중·종성, 호환 자모, 확장 자모)에만 쓴다. 한글이 아닌 글자는 그대로 일치할 때만 지나간다(HARD).
			// strict = false이면(예산을 넘겨 형태를 끝내는 경로) 모든 글자를 비용 1로 편집한다
			constexpr int HARD = 1 << 20;

			bool isJamo(char32_t c)
			{
				return (0x1100 <= c && c <= 0x11FF) || (0x3131 <= c && c <= 0x318E) || (0xA960 <= c && c <= 0xA97F) || (0xD7B0 <= c && c <= 0xD7FF);
			}

			int indel(char32_t c, bool strict = true) { return !strict || isJamo(c) ? 1 : HARD; }

			// 원문 글자를 맞추지 않고 건너뛰는 비용. 보이지 않는 공백류 글자는 정답에 형태소로 남지 않고, _는 모델 출력 규칙(model_form)에서
			// 공백으로 쓸 수 있으므로(^_^ → ^ ^, 형태 안의 공백은 무시된다) 0
			int skipSrc(char32_t c, bool strict = true) { return c == '_' || isBlankLike(c) ? 0 : indel(c, strict); }

			int subst(char32_t a, char32_t b, bool strict = true)
			{
				if (a == b) return 0;
				return !strict || (isJamo(a) && isJamo(b)) ? 1 : HARD;
			}

			// 편집 거리. cap을 넘으면 cap + 1
			int lev(const u32string& a, const u32string& b, int cap, bool strict = true)
			{
				vector<int> d(b.size() + 1);
				d[0] = 0;
				for (size_t j = 1; j <= b.size(); ++j) d[j] = min(d[j - 1] + skipSrc(b[j - 1], strict), HARD);
				for (size_t i = 1; i <= a.size(); ++i)
				{
					int p = d[0];
					d[0] = min(d[0] + indel(a[i - 1], strict), HARD);
					int lo = d[0];
					for (size_t j = 1; j <= b.size(); ++j)
					{
						const int old = d[j];
						d[j] = min({ d[j] + indel(a[i - 1], strict), d[j - 1] + skipSrc(b[j - 1], strict), p + subst(a[i - 1], b[j - 1], strict), HARD });
						p = old;
						lo = min(lo, d[j]);
					}
					if (lo > cap) return cap + 1;
				}
				return min(d[b.size()], cap + 1);
			}

			// a를 b의 앞부분들(0..|b|글자)과 비교한 편집 거리 행
			vector<int> levRow(u32string_view a, u32string_view b)
			{
				vector<int> d(b.size() + 1);
				d[0] = 0;
				for (size_t j = 1; j <= b.size(); ++j) d[j] = min(d[j - 1] + skipSrc(b[j - 1]), HARD);
				for (size_t i = 1; i <= a.size(); ++i)
				{
					int p = d[0];
					d[0] = min(d[0] + indel(a[i - 1]), HARD);
					for (size_t j = 1; j <= b.size(); ++j)
					{
						const int old = d[j];
						d[j] = min({ d[j] + indel(a[i - 1]), d[j - 1] + skipSrc(b[j - 1]), p + subst(a[i - 1], b[j - 1]), HARD });
						p = old;
					}
				}
				return d;
			}

			// 편집 거리 행 d 뒤에 '조건 ok를 만족하는 아무 글자 하나'를 덧붙인 행. 그 글자는 b[j-1]과 같을 수 있으면 비용 0,
			// 아니면 jamo(그 글자가 한글 자모인가)에 따라 삽입·치환 비용을 센다
			template<class Ok>
			vector<int> stepAny(const vector<int>& d, u32string_view b, Ok ok, bool jamo)
			{
				vector<int> e(d.size());
				const int ins = jamo ? 1 : HARD;
				e[0] = min(d[0] + ins, HARD);
				for (size_t j = 1; j < d.size(); ++j)
				{
					const int sub = ok(b[j - 1]) ? 0 : (jamo && isJamo(b[j - 1]) ? 1 : HARD);
					e[j] = min({ d[j] + ins, e[j - 1] + skipSrc(b[j - 1]), d[j - 1] + sub, HARD });
				}
				return e;
			}

			// 바이트열 끝에서 끊긴 UTF-8 글자: 끊긴 바이트 수와, 이어 쓰면 될 수 있는 코드 포인트 범위. 끊기지 않았으면 len = 0
			struct Tail
			{
				size_t len = 0;
				char32_t lo = 0, hi = 0;
			};

			Tail incompleteTail(const string& s)
			{
				Tail t;
				size_t k = 0;
				while (k < 3 && k < s.size() && ((uint8_t)s[s.size() - 1 - k] >> 6) == 2) ++k;
				if (k >= s.size()) return t;
				const auto lead = (uint8_t)s[s.size() - 1 - k];
				const size_t need = lead < 0x80 ? 1 : (lead >> 5) == 6 ? 2 : (lead >> 4) == 14 ? 3 : (lead >> 3) == 30 ? 4 : 0;
				if (!need || k + 1 >= need) return t;
				char32_t v = need == 2 ? (lead & 0x1F) : need == 3 ? (lead & 0x0F) : (lead & 0x07);
				for (size_t j = 0; j < k; ++j) v = (v << 6) | ((uint8_t)s[s.size() - k + j] & 0x3F);
				const size_t rest = need - 1 - k;
				t.len = k + 1;
				t.lo = v << (6 * rest);
				t.hi = t.lo | (((char32_t)1 << (6 * rest)) - 1);
				return t;
			}

			// 편집 거리 행 row 뒤에 범위 [lo, hi]의 글자 하나를 덧붙였을 때의 최소 비용(그 글자를 무엇으로 마무리하든 이보다 싸지 않다).
			// 비용은 같은 글자인지와 자모인지로만 정해지고, 덧붙인 자모 하나는 한 정렬 경로에서 원문 글자 하나에만 맞춰지므로
			// 초성 집합 × 중성 집합 × 종성 집합 꼴의 묶음은 stepAny 세 번으로 정확히 센다. 연속 범위의 음절은 이런 묶음 다섯 개 이하로 나뉜다
			int tailCost(const vector<int>& row, u32string_view b, char32_t lo, char32_t hi, bool skipFirst = false)
			{
				int best = HARD;
				const size_t from = skipFirst ? 1 : 0;
				const auto rowMin = [&](const vector<int>& d) { return d.size() > from ? *min_element(d.begin() + from, d.end()) : HARD; };
				const auto inRange = [&](char32_t c) { return lo <= c && c <= hi; };
				const auto isSyllable = [](char32_t c) { return 0xAC00 <= c && c <= 0xD7A3; };

				// 음절이 아닌 글자: 자모는 편집할 수 있고, 나머지는 그대로 일치할 때만 지나간다
				best = min(best, rowMin(stepAny(row, b, [&](char32_t c) { return inRange(c) && !isSyllable(c) && !isJamo(c); }, false)));
				static constexpr pair<char32_t, char32_t> jamoBlocks[] = { { 0x1100, 0x11FF }, { 0x3131, 0x318E }, { 0xA960, 0xA97F }, { 0xD7B0, 0xD7FF } };
				if (any_of(begin(jamoBlocks), end(jamoBlocks), [&](const pair<char32_t, char32_t>& r) { return r.first <= hi && lo <= r.second; }))
				{
					best = min(best, rowMin(stepAny(row, b, [&](char32_t c) { return inRange(c) && isJamo(c); }, true)));
				}

				const char32_t a = max(lo, (char32_t)0xAC00), z = min(hi, (char32_t)0xD7A3);
				if (a > z) return best;
				// 초성 [l0, l1] × 중성 [v0, v1] × 종성 [t0, t1](0은 종성 없음)
				const auto block = [&](int l0, int l1, int v0, int v1, int t0, int t1)
				{
					if (l0 > l1 || v0 > v1 || t0 > t1) return;
					const auto eL = stepAny(row, b, [&](char32_t c) { return (char32_t)(0x1100 + l0) <= c && c <= (char32_t)(0x1100 + l1); }, true);
					const auto eV = stepAny(eL, b, [&](char32_t c) { return (char32_t)(0x1161 + v0) <= c && c <= (char32_t)(0x1161 + v1); }, true);
					if (t0 == 0) best = min(best, rowMin(eV));
					const int ta = max(t0, 1);
					if (ta <= t1) best = min(best, rowMin(stepAny(eV, b, [&](char32_t c) { return (char32_t)(0x11A7 + ta) <= c && c <= (char32_t)(0x11A7 + t1); }, true)));
				};
				const int sa = a - 0xAC00, sz = z - 0xAC00;
				const int La = sa / 588, Va = sa % 588 / 28, Ta = sa % 28;
				const int Lz = sz / 588, Vz = sz % 588 / 28, Tz = sz % 28;
				if (La == Lz && Va == Vz) block(La, La, Va, Va, Ta, Tz);
				else if (La == Lz)
				{
					block(La, La, Va, Va, Ta, 27);
					block(La, La, Va + 1, Vz - 1, 0, 27);
					block(La, La, Vz, Vz, 0, Tz);
				}
				else
				{
					block(La, La, Va, Va, Ta, 27);
					block(La, La, Va + 1, 20, 0, 27);
					block(La + 1, Lz - 1, 0, 20, 0, 27);
					block(Lz, Lz, 0, Vz - 1, 0, 27);
					block(Lz, Lz, Vz, Vz, 0, Tz);
				}
				return best;
			}

			bool isHangul(char32_t c) { return (0xAC00 <= c && c <= 0xD7A3) || isJamo(c); }
		}

		LatticeTokens::LatticeTokens(vector<string> _tokenBytes, vector<string> knownWords)
			: tokenBytes{ move(_tokenBytes) }
		{
			vector<map<uint8_t, uint32_t>> children(1);
			vector<vector<uint32_t>> toks(1);
			for (size_t i = 0; i < tokenBytes.size(); ++i)
			{
				uint32_t node = 0;
				for (auto ch : tokenBytes[i])
				{
					auto it = children[node].find((uint8_t)ch);
					if (it == children[node].end())
					{
						const auto id = (uint32_t)children.size();
						children.emplace_back();
						toks.emplace_back();
						it = children[node].emplace((uint8_t)ch, id).first;
					}
					node = it->second;
				}
				if (node) toks[node].emplace_back((uint32_t)i);
			}
			trie.resize(children.size());
			for (size_t i = 0; i < children.size(); ++i)
			{
				auto& n = trie[i];
				n.edgeBegin = (uint32_t)trieEdges.size();
				trieEdges.insert(trieEdges.end(), children[i].begin(), children[i].end());
				n.edgeEnd = (uint32_t)trieEdges.size();
				n.tokBegin = (uint32_t)trieTokens.size();
				trieTokens.insert(trieTokens.end(), toks[i].begin(), toks[i].end());
				n.tokEnd = (uint32_t)trieTokens.size();
			}
			for (auto& w : knownWords)
			{
				if (!w.empty() && known.emplace(w).second) knownSorted.emplace_back(w);
			}
			sort(knownSorted.begin(), knownSorted.end());
		}

		uint32_t LatticeTokens::child(uint32_t node, uint8_t ch) const
		{
			const auto first = trieEdges.begin() + trie[node].edgeBegin, last = trieEdges.begin() + trie[node].edgeEnd;
			const auto it = lower_bound(first, last, ch, [](const pair<uint8_t, uint32_t>& e, uint8_t c) { return e.first < c; });
			return it != last && it->first == ch ? it->second : 0;
		}

		bool LatticeTokens::hasKnownPrefix(const string& prefix) const
		{
			auto it = lower_bound(knownSorted.begin(), knownSorted.end(), prefix);
			return it != knownSorted.end() && startsWith(*it, prefix);
		}

		Lattice::Lattice(const string& text, const vector<LatticeGroupSpec>& specs, int _budget,
			shared_ptr<const LatticeTokens> _tokens, bool _anyForm)
			: budget{ _budget }, tokens{ move(_tokens) }, anyForm{ _anyForm }
		{
			auto t = decodeUtf8(text, false);
			if (!t) throw invalid_argument{ "`text` must be valid UTF-8." };
			T = move(*t);
			n = T.size();
			groupsAt.resize(n + 1);
			for (auto& g : specs)
			{
				if (g.start > n || g.end > n || g.forms.empty()) continue;
				groupsAt[g.start].emplace_back((uint32_t)groups.size());
				auto& added = groups.emplace_back(Group{ g.start, g.end, g.forms, {} });
				for (auto& f : added.bytes)
				{
					auto u = decodeUtf8(f, true);
					added.jamo.emplace_back(jamoOf(*u));
				}
			}
			J.resize(n);
			off.reserve(n + 1);
			jOff.reserve(n + 1);
			size_t b = 0;
			for (size_t i = 0; i < n; ++i)
			{
				appendJamo(J[i], T[i]);
				off.emplace_back(b);
				jOff.emplace_back(Jn.size());
				if (T[i] != ' ') Jn += J[i];
				if (T[i] != ' ')
				{
					const auto cb = encodeUtf8(u32string(1, T[i]));
					Tn += cb;
					b += cb.size();
				}
			}
			off.emplace_back(b);
			jOff.emplace_back(Jn.size());
			for (size_t i = n + 1; i-- > 0;) byte2pos[off[i]] = (uint32_t)i;
			eend.resize(n + 1);
			uint32_t e = (uint32_t)n;
			for (size_t i = n + 1; i-- > 0;)
			{
				if (i < n && T[i] == ' ') e = (uint32_t)i;
				eend[i] = e;
			}
		}

		void Lattice::put(State& h, Key key, int c)
		{
			for (auto& [k, v] : h)
			{
				if (k == key)
				{
					if (c < v) v = c;
					return;
				}
			}
			h.emplace_back(key, c);
		}

		void Lattice::settle(State& h, uint32_t p, int c, bool inserted) const
		{
			put(h, { p, -1, 0, inserted }, c);
			if (p < n && T[p] == ' ')
			{
				while (p < n && T[p] == ' ') ++p;
				put(h, { p, -1, 0 }, 0);
			}
		}

		u32string Lattice::jamoSpan(uint32_t a, uint32_t b) const
		{
			u32string out;
			for (uint32_t i = a; i < b && i < n; ++i) if (T[i] != ' ') out += J[i];
			return out;
		}

		bool Lattice::symbolAt(uint32_t i) const
		{
			// 어절 끝(공백 앞) 상태에서도 다음 어절 첫 글자를 본다(.. ?눈 에서 ..와 ? 사이에 한글을 끼워 넣지 못하게)
			while (i < n && (T[i] == ' ' || isBlankLike(T[i]))) ++i;
			return i < n && !isHangul(T[i]);
		}

		uint32_t Lattice::reach(uint32_t i, size_t k) const
		{
			uint32_t j = i;
			for (size_t cnt = 0; j < n && cnt < k; ++j) if (T[j] != ' ') ++cnt;
			return j;
		}

		Lattice::State Lattice::start() const
		{
			State h;
			settle(h, 0, 0);
			return h;
		}

		bool Lattice::atEnd(const State& h) const
		{
			return any_of(h.begin(), h.end(), [&](const pair<Key, int>& p) { return p.first.pos == n && p.first.group < 0; });
		}

		bool Lattice::viable(const State& h, const string& buf) const
		{
			return Viability{ *this, h, string{}, buf.size() }(buf);
		}

		Lattice::Viability::Viability(const Lattice& _lat, const State& _h, const string& _buf, size_t maxTokenBytes)
			: lat{ &_lat }, h{ &_h }, buf{ _buf }, nb0{ noSpace(_buf) }
		{
			p0 = nb0.size() - incompleteTail(nb0).len;
			if (auto core = decodeUtf8(nb0.substr(0, p0), false))
			{
				core0 = true;
				E0 = jamoOf(*core);
			}
			// 새로 완성되는 글자의 자모 수는 그 바이트 수(buf의 끊긴 바이트 + 토큰) 이하
			maxLen = E0.size() + (nb0.size() - p0) + maxTokenBytes;
		}

		// 편집 대상마다 buf의 완성된 글자까지의 행을 세고, 그 최솟값이 예산을 넘는 대상은 뺀다
		// (자모를 덧붙인 행의 값과 tailCost는 모두 원래 행의 최솟값 이상이다)
		void Lattice::Viability::build()
		{
			built = true;
			auto& L = *lat;
			const auto add = [&](u32string_view b, int limit, bool noPure, bool span, uint32_t pos, int slack)
			{
				auto row = levRow(E0, b);
				if (*min_element(row.begin(), row.end()) > limit) return;
				targets.push_back(Target{ b, move(row), limit, noPure, span, pos, slack });
			};
			for (auto& [key, c] : *h)
			{
				const int limit = L.budget - c;
				if (key.group >= 0)
				{
					add(L.groups[key.group].jamo[key.k], limit, false, false, 0, 0);
					continue;
				}
				const auto i = key.pos;
				const int slack = max(limit, 0);
				const auto sym = L.symbolAt(i);
				const auto e = L.reach(i, maxLen + 1 + slack);
				add(u32string_view{ L.Jn }.substr(L.jOff[i], L.jOff[e] - L.jOff[i]), limit, sym, true, i, slack);
				for (auto gi : L.groupsAt[i]) if (L.usable(key, gi)) add(L.groups[gi].jamo[0], limit, sym, false, 0, 0);
			}
		}

		bool Lattice::Viability::operator()(const string& tokenBytes)
		{
			auto& L = *lat;
			const auto nb = nb0 + noSpace(tokenBytes);
			if (nb.empty()) return false;
			const auto tail = incompleteTail(nb);
			const size_t pe = nb.size() - tail.len;
			// 끊긴 글자가 buf의 완성된 부분까지 거슬러 올라가면 기준 행을 쓸 수 없다
			if (pe < p0) return L.viable(*h, buf + tokenBytes);

			// 편집 없이 원문·묶음 형태를 그대로 쓰는 경로
			for (auto& [key, c] : *h)
			{
				if (key.group >= 0)
				{
					if (startsWith(L.groups[key.group].bytes[key.k], nb)) return true;
					continue;
				}
				const auto i = key.pos;
				if (L.off[i] + nb.size() <= L.Tn.size() && L.Tn.compare(L.off[i], nb.size(), nb) == 0) return true;
				for (auto gi : L.groupsAt[i]) if (L.usable(key, gi) && startsWith(L.groups[gi].bytes[0], nb)) return true;
			}

			// 자모 편집 경로
			if (!core0 || !L.knownPrefix(nb)) return false;
			const auto r = decodeUtf8(nb.substr(p0, pe - p0), false);
			if (!r) return false;
			const auto added = jamoOf(*r);
			const size_t len = E0.size() + added.size();
			if (len > maxLen) return L.viable(*h, buf + tokenBytes);
			if (!built) build();
			for (auto& t : targets)
			{
				const size_t m = t.span ? L.jOff[L.reach(t.pos, len + 1 + t.slack)] - L.jOff[t.pos] : t.b.size();
				const auto b = t.b.substr(0, m);
				d.assign(t.row.begin(), t.row.begin() + m + 1);
				e.resize(m + 1);
				bool dead = false;
				for (auto ch : added)
				{
					e[0] = min(d[0] + indel(ch), HARD);
					int lo = e[0];
					for (size_t j = 1; j <= m; ++j)
					{
						e[j] = min({ d[j] + indel(ch), e[j - 1] + skipSrc(b[j - 1]), d[j - 1] + subst(ch, b[j - 1]), HARD });
						lo = min(lo, e[j]);
					}
					swap(d, e);
					if (lo > t.limit)
					{
						dead = true;
						break;
					}
				}
				if (dead) continue;
				const size_t from = t.noPure ? 1 : 0;
				const int cost = tail.len ? tailCost(d, b, tail.lo, tail.hi, t.noPure)
					: (d.size() > from ? *min_element(d.begin() + from, d.end()) : HARD);
				if (cost <= t.limit) return true;
			}
			return false;
		}

		Lattice::State Lattice::advance(const State& h, const string& buf, bool relaxed) const
		{
			const auto nb = noSpace(buf);
			State out;
			if (nb.empty()) return out;
			const auto w = decodeUtf8(nb, false);
			optional<u32string> jw;
			for (auto& [key, c] : h)
			{
				const int rest = relaxed ? 64 : budget - c;
				const auto clampCost = [&](int x) { return relaxed ? min(x, budget) : x; };
				// 묶음의 k번째 형태를 쓴 뒤로 나아간다(비용 d는 그 형태를 자모 편집한 비용)
				const auto stepGroup = [&](uint32_t pos, uint32_t gi, uint32_t k, int d)
				{
					auto& g = groups[gi];
					if (k + 1 < g.bytes.size()) put(out, { pos, (int32_t)gi, k + 1 }, clampCost(c + d));
					else settle(out, g.end, clampCost(c + d), g.start == g.end);
				};
				const auto groupEdit = [&](uint32_t gi, uint32_t k) -> int
				{
					if (!w || !known(nb)) return rest + 1;
					if (!jw) jw = jamoOf(*w);
					return lev(*jw, groups[gi].jamo[k], rest, !relaxed);
				};
				if (key.group >= 0)
				{
					auto& g = groups[key.group];
					if (g.bytes[key.k] == nb) stepGroup(key.pos, key.group, key.k, 0);
					else if (const int d = groupEdit(key.group, key.k); d <= rest) stepGroup(key.pos, key.group, key.k, d);
					continue;
				}
				const auto i = key.pos;
				if (off[i] + nb.size() <= Tn.size() && Tn.compare(off[i], nb.size(), nb) == 0)
				{
					auto it = byte2pos.find(off[i] + nb.size());
					if (it != byte2pos.end())
					{
						const auto j = it->second;
						const bool crossed = any_of(T.begin() + i, T.begin() + max(i, j), [](char32_t ch) { return ch == ' '; });
						settle(out, j, crossed ? 0 : c);
					}
				}
				for (auto gi : groupsAt[i])
				{
					if (!usable(key, gi)) continue;
					if (groups[gi].bytes[0] == nb) stepGroup(i, gi, 0, 0);
					else if (const int d = groupEdit(gi, 0); d <= rest) stepGroup(i, gi, 0, d);
				}
				if (w && known(nb))
				{
					if (!jw) jw = jamoOf(*w);
					// 공백을 넘어 다음 어절까지 맞출 수 있다. 다른 어절에 도착하면 그 어절에는 이 형태의 편집 비용만 넘긴다
					const uint32_t far = reach(i, w->size() + rest);
					// 다음 원문 글자가 기호이면 원문을 소비하지 않는 형태(한글 삽입)는 쓸 수 없다: 모델이 빠뜨린 기호를 먼저 쓰게 한다
					for (uint32_t j = relaxed || symbolAt(i) ? i + 1 : i; j <= far; ++j)
					{
						if (j > i && T[j - 1] == ' ') continue;
						const int d = lev(*jw, jamoSpan(i, j), rest, !relaxed);
						if (d > rest) continue;
						const bool crossed = j > eend[i];
						settle(out, j, clampCost((crossed ? 0 : c) + d));
					}
				}
			}
			return out;
		}

		vector<string> Lattice::continuations(const State& h, const string& buf) const
		{
			const auto nb = noSpace(buf);
			vector<string> outs;
			for (auto& [key, c] : h)
			{
				if (key.group >= 0)
				{
					auto& bs = groups[key.group].bytes[key.k];
					if (startsWith(bs, nb)) outs.emplace_back(bs.substr(nb.size()));
					else if (!nb.empty() && knownPrefix(nb)) addSuffixes(outs, bs); // 묶음 형태를 자모 편집해 쓰던 형태의 마무리
					continue;
				}
				const auto i = key.pos;
				if (off[i] + nb.size() <= Tn.size() && Tn.compare(off[i], nb.size(), nb) == 0)
				{
					const size_t a = off[i] + nb.size(), e = min(off[eend[i]] + 64, Tn.size());
					outs.emplace_back(e > a ? Tn.substr(a, e - a) : string{});
				}
				else if (!nb.empty() && knownPrefix(nb))
				{
					// 자모 편집으로 쓰던 형태: 원문의 남은 부분을 옮겨 쓰는 마무리 토큰도 후보로(공백을 빼고 최대 32글자 안의 각 위치에서)
					const uint32_t far = reach(i, 32);
					for (uint32_t j = i; j <= far; ++j)
					{
						if (j > i && T[j - 1] == ' ') continue;
						const size_t a = off[j], e = min(a + 64, Tn.size());
						if (e > a) outs.emplace_back(Tn.substr(a, e - a));
					}
				}
				for (auto gi : groupsAt[i])
				{
					if (!usable(key, gi)) continue;
					auto& bs = groups[gi].bytes[0];
					if (startsWith(bs, nb)) outs.emplace_back(bs.substr(nb.size()));
					else if (!nb.empty() && knownPrefix(nb)) addSuffixes(outs, bs);
				}
			}
			return outs;
		}

		vector<pair<uint32_t, uint32_t>> Lattice::copySpans(const State& h, const string& buf) const
		{
			vector<pair<uint32_t, uint32_t>> out;
			const auto nb = noSpace(buf);
			if (nb.empty()) return out;
			for (auto& [key, c] : h)
			{
				const auto i = key.pos;
				if (key.group >= 0 || off[i] + nb.size() > Tn.size() || Tn.compare(off[i], nb.size(), nb) != 0) continue;
				auto it = byte2pos.find(off[i] + nb.size());
				if (it != byte2pos.end()) out.emplace_back(i, it->second);
			}
			return out;
		}

		Lattice::State Lattice::after(const vector<string>& forms) const
		{
			auto h = start();
			for (auto& f : forms)
			{
				auto u = decodeUtf8(f, true);
				h = advance(h, encodeUtf8(modelForm(*u)));
				if (h.empty()) break;
			}
			return h;
		}

		LatticeFollower::LatticeFollower(shared_ptr<const Lattice> _lat, shared_ptr<const LatticeTokens> _tokens,
			vector<uint32_t> _tags, uint32_t _eos, const vector<string>& prefix,
			FormTagTable _formTags)
			: lat{ move(_lat) }, tokens{ move(_tokens) }, tags{ move(_tags) }, tagSet{ tags.begin(), tags.end() },
			eos{ _eos }, formTags{ move(_formTags) }
		{
			H = lat->after(prefix);
			dead = H.empty();
		}

		vector<uint32_t> LatticeFollower::tagsFor(const string& buf) const
		{
			const auto nb = noSpace(buf);
			auto it = formTags.find(nb);
			if (it != formTags.end())
			{
				unordered_set<uint32_t> restricted;
				bool matched = false;
				for (auto& [i, j] : lat->copySpans(H, nb))
				{
					for (auto& sp : it->second)
					{
						if (sp.start != i || sp.end != j) continue;
						restricted.insert(sp.tags.begin(), sp.tags.end());
						matched = true;
					}
				}
				if (matched)
				{
					vector<uint32_t> out;
					for (auto t : tags) if (restricted.count(t)) out.emplace_back(t);
					return out;
				}
			}
			return tags;
		}

		void LatticeFollower::consume(uint32_t token)
		{
			prepared = false;
			viability.reset();
			if (dead || token == eos) return;
			if (tagSet.count(token))
			{
				auto next = lat->advance(H, buf, relaxedNext);
				relaxedNext = false;
				if (next.empty()) dead = true;
				H = move(next);
				buf.clear();
				done = -1;
			}
			else if (auto b = tokens->bytesOf(token))
			{
				buf += *b;
				done = -1;
			}
			else dead = true;
		}

		void LatticeFollower::prepare()
		{
			if (dead || prepared) return;
			prepared = true;
			prepOut.clear();
			prepSeen.clear();
			if (!buf.empty())
			{
				if (done < 0) done = lat->advance(H, buf).empty() ? 0 : 1;
				if (done) prepOut = tagsFor(noSpace(buf));
			}
			else if (lat->atEnd(H)) prepOut.emplace_back(eos);

			vector<uint32_t> cands;
			const auto add = [&](uint32_t t) { if (prepSeen.emplace(t).second) cands.emplace_back(t); };
			// 글자 중간에서 끊겼으면 그 글자를 정확히 마무리하는 토큰도 후보로 넣는다(상위 토큰에 없으면 막히지 않게)
			const auto nb = noSpace(buf);
			if (const auto tail = incompleteTail(nb); tail.len)
			{
				const auto head = nb.substr(nb.size() - tail.len);
				const uint8_t lead = head[0];
				const size_t need = (lead >> 4) == 14 ? 3 : (lead >> 3) == 30 ? 4 : 2;
				tokens->forEachContinuation(need - tail.len, [&](LatticeTokens::TokenRange ts)
				{
					// 완성한 글자가 올바른 UTF-8이어야 한다(과다 길이, 서로게이트, U+10FFFF 초과 제외)
					if (!decodeUtf8(head + *tokens->bytesOf(*ts.first), false)) return;
					for (auto t : ts) add(t);
				});
			}
			for (auto& rest : lat->continuations(H, buf))
			{
				tokens->forEachPrefix(rest, 64, [&](LatticeTokens::TokenRange ts) { for (auto t : ts) add(t); });
			}
			// top 토큰은 보통 짧으므로 16바이트까지는 기준 행을 같이 쓰게 한다
			size_t maxBytes = 16;
			for (auto t : cands) maxBytes = max(maxBytes, tokens->bytesOf(t)->size());
			viability = make_unique<Lattice::Viability>(*lat, H, buf, maxBytes);
			for (auto t : cands) if ((*viability)(*tokens->bytesOf(t))) prepOut.emplace_back(t);
		}

		optional<vector<uint32_t>> LatticeFollower::allowed(const vector<uint32_t>& top)
		{
			if (dead) return nullopt;
			prepare();
			vector<uint32_t> out = prepOut;
			vector<uint32_t> checked;
			for (auto t : top)
			{
				const auto b = tokens->bytesOf(t);
				if (!b || prepSeen.count(t) || find(checked.begin(), checked.end(), t) != checked.end()) continue;
				checked.emplace_back(t);
				if ((*viability)(*b)) out.emplace_back(t);
			}
			if (out.empty())
			{
				// 이어 쓸 수 있는 토큰이 없다. 제약을 풀면 반복에 빠질 수 있으므로, 쓰던 형태가 온전하면 예산을 넘겨서라도
				// 가장 가까운 원문 위치에서 끝내게 하고(그 뒤로는 다시 격자 제약), 그럴 수 없으면 종료 토큰만 허용한다
				if (!buf.empty() && decodeUtf8(noSpace(buf), false) && !lat->advance(H, buf, true).empty())
				{
					relaxedNext = true;
					++relaxedCount;
					out = tagsFor(noSpace(buf));
					if (!out.empty()) return out;
					out.insert(out.end(), tags.begin(), tags.end());
					return out;
				}
				stuck = true;
				return vector<uint32_t>{ eos };
			}
			return out;
		}
	}
}
