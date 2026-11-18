#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include "ByteGrammar.h"

using namespace std;

namespace kiwi
{
	namespace guide
	{
		namespace
		{
			constexpr uint32_t none = numeric_limits<uint32_t>::max();
			constexpr uint32_t afterSenseKey = none - 1; // 번호 자리 다음 노드로 가는 내부 키(토큰이 아님)
			constexpr uint32_t specialBase = 256; // 자동자의 키: 형태는 바이트(0~255), 태그·번호 토큰은 specialBase + 토큰 ID
		}

		ByteGrammar::ByteGrammar(vector<string> _tokenBytes, const vector<uint32_t>& specialTokens, vector<uint32_t> _eosTokens)
			: tokenBytes{ move(_tokenBytes) }, eosTokens{ move(_eosTokens) }
		{
			auto& tb = tokenBytes;
			vocabSize_ = tb.size();
			for (auto t : specialTokens) vocabSize_ = max(vocabSize_, (size_t)t + 1);
			for (auto t : eosTokens) vocabSize_ = max(vocabSize_, (size_t)t + 1);
			isSpecial.resize(vocabSize_);
			for (auto t : specialTokens)
			{
				if (t < tb.size() && !tb[t].empty()) throw invalid_argument{ "A special token must not be a normal token." };
				isSpecial[t] = 1;
			}

			unordered_map<uint64_t, uint32_t> edges;
			vocabChildren.emplace_back();
			vocabToken.emplace_back(-1);
			for (size_t i = 0; i < tb.size(); ++i)
			{
				uint32_t node = 0;
				for (auto ch : tb[i])
				{
					auto key = ((uint64_t)node << 8) | (uint8_t)ch;
					auto it = edges.find(key);
					if (it == edges.end())
					{
						it = edges.emplace(key, (uint32_t)vocabChildren.size()).first;
						vocabChildren[node].emplace_back((uint8_t)ch, it->second);
						vocabChildren.emplace_back();
						vocabToken.emplace_back(-1);
					}
					node = it->second;
				}
				if (node) vocabToken[node] = (int64_t)i;
			}
			for (auto& c : vocabChildren) sort(c.begin(), c.end());
		}

		unique_ptr<GrammarMatcher> ByteGrammar::makeMatcher(uint32_t numStates, const vector<GraphEdge>& edges) const
		{
			return make_unique<GrammarMatcher>(shared_from_this(), numStates, edges);
		}

		GrammarMatcher::GrammarMatcher(shared_ptr<const ByteGrammar> _grammar, uint32_t numStates, const vector<GraphEdge>& edges)
			: grammar{ move(_grammar) }
		{
			if (!numStates) throw invalid_argument{ "`numStates` must be positive." };
			const auto& g = *grammar;
			auto a = make_shared<Automaton>();
			auto& children = a->children;
			auto& eps = a->eps;
			auto& finals = a->finals;

			// 쓸 수 없는 간선(모르는 태그)을 빼고, 시작에서 닿고 끝에 닿는 간선만 남긴다.
			vector<uint8_t> usable(edges.size());
			for (size_t i = 0; i < edges.size(); ++i)
			{
				auto& e = edges[i];
				bool ok = !e.morphs.empty() && e.from < numStates && e.to < numStates && e.from < e.to;
				for (auto& m : e.morphs)
				{
					if (!ok) break;
					ok = !m.form.empty() && !m.tagTokens.empty();
					for (auto t : m.tagTokens) ok = ok && t < g.vocabSize_ && g.isSpecial[t];
					for (auto s : m.senseTokens) ok = ok && s < g.vocabSize_ && g.isSpecial[s];
				}
				usable[i] = ok;
			}
			vector<uint8_t> fwd(numStates), bwd(numStates);
			fwd[0] = 1;
			bwd[numStates - 1] = 1;
			vector<size_t> order(edges.size());
			iota(order.begin(), order.end(), 0);
			sort(order.begin(), order.end(), [&](size_t a, size_t b) { return edges[a].from < edges[b].from; });
			for (auto i : order) if (usable[i] && fwd[edges[i].from]) fwd[edges[i].to] = 1;
			for (auto it = order.rbegin(); it != order.rend(); ++it) if (usable[*it] && bwd[edges[*it].to]) bwd[edges[*it].from] = 1;

			unordered_map<uint64_t, uint32_t> childMap;
			const auto newNode = [&]()
			{
				children.emplace_back();
				eps.emplace_back();
				finals.emplace_back(0);
				return (uint32_t)children.size() - 1;
			};
			const auto getChild = [&](uint32_t node, uint32_t key)
			{
				auto it = childMap.find(((uint64_t)node << 32) | key);
				if (it != childMap.end()) return it->second;
				const auto c = newNode();
				childMap.emplace(((uint64_t)node << 32) | key, c);
				return c;
			};
			vector<uint32_t> roots(numStates, none);
			const auto root = [&](uint32_t s)
			{
				if (roots[s] == none) roots[s] = newNode();
				return roots[s];
			};
			a->start = root(0);
			for (size_t i = 0; i < edges.size(); ++i)
			{
				auto& e = edges[i];
				if (!usable[i] || !fwd[e.from] || !bwd[e.to])
				{
					++a->droppedEdges;
					continue;
				}
				uint32_t cur = root(e.from);
				for (auto& m : e.morphs)
				{
					for (auto ch : m.form) cur = getChild(cur, (uint8_t)ch);
					if (m.tagTokens.size() == 1) cur = getChild(cur, specialBase + m.tagTokens[0]);
					else
					{
						// 태그 후보가 여럿이면 이 간선만의 노드에서 갈라졌다 모은다. 같은 형태의 다른 간선과 태그 노드를 나누면 뒤 경로가 섞인다.
						const auto branch = newNode(), joined = newNode();
						eps[cur].emplace_back(branch);
						for (auto t : m.tagTokens) childMap.emplace(((uint64_t)branch << 32) | (specialBase + t), joined);
						cur = joined;
					}
					if (!m.senseTokens.empty())
					{
						const auto after = getChild(cur, afterSenseKey);
						if (!m.senseRequired) eps[cur].emplace_back(after);
						for (auto s : m.senseTokens) childMap.emplace(((uint64_t)cur << 32) | (specialBase + s), after);
						cur = after;
					}
				}
				const auto next = root(e.to); // root()가 노드를 늘리면 eps가 재할당되므로 먼저 구한다
				eps[cur].emplace_back(next);
			}
			if (bwd[0]) finals[root(numStates - 1)] = 1;

			for (auto& [key, c] : childMap) children[key >> 32].emplace_back((uint32_t)key, c);
			for (auto& c : children) sort(c.begin(), c.end());
			for (auto& e : eps)
			{
				sort(e.begin(), e.end());
				e.erase(unique(e.begin(), e.end()), e.end());
			}
			nfa = move(a);
			reset();
		}

		uint32_t GrammarMatcher::child(uint32_t node, uint32_t key) const
		{
			auto& c = nfa->children[node];
			auto it = lower_bound(c.begin(), c.end(), make_pair(key, (uint32_t)0));
			return it != c.end() && it->first == key ? it->second : none;
		}

		void GrammarMatcher::close(vector<uint32_t>& nodes) const
		{
			vector<uint32_t> stack = nodes;
			while (!stack.empty())
			{
				const auto n = stack.back();
				stack.pop_back();
				for (auto e : nfa->eps[n])
				{
					nodes.emplace_back(e);
					stack.emplace_back(e);
				}
			}
			sort(nodes.begin(), nodes.end());
			nodes.erase(unique(nodes.begin(), nodes.end()), nodes.end());
		}

		void GrammarMatcher::reset()
		{
			active = { nfa->start };
			close(active);
			terminated = false;
		}

		bool GrammarMatcher::canTerminate() const
		{
			return any_of(active.begin(), active.end(), [&](uint32_t n) { return nfa->finals[n]; });
		}

		bool GrammarMatcher::accept(uint32_t token)
		{
			auto& g = *grammar;
			if (terminated || token >= g.vocabSize_) return false;
			if (find(g.eosTokens.begin(), g.eosTokens.end(), token) != g.eosTokens.end())
			{
				if (!canTerminate()) return false;
				terminated = true;
				return true;
			}
			vector<uint32_t> next;
			if (g.isSpecial[token])
			{
				for (auto n : active)
				{
					const auto c = child(n, specialBase + token);
					if (c != none) next.emplace_back(c);
				}
			}
			else
			{
				auto& bytes = g.tokenBytes;
				if (token >= bytes.size() || bytes[token].empty()) return false;
				for (auto n : active)
				{
					uint32_t cur = n;
					for (auto ch : bytes[token])
					{
						cur = child(cur, (uint8_t)ch);
						if (cur == none) break;
					}
					if (cur != none) next.emplace_back(cur);
				}
			}
			if (next.empty()) return false;
			close(next);
			active = move(next);
			return true;
		}

		void GrammarMatcher::collectFormTokens(uint32_t fn, uint32_t vn, vector<uint32_t>& ret) const
		{
			auto& g = *grammar;
			for (auto [key, fc] : nfa->children[fn])
			{
				if (key >= specialBase) break;
				auto& vc = g.vocabChildren[vn];
				auto it = lower_bound(vc.begin(), vc.end(), make_pair((uint8_t)key, (uint32_t)0));
				if (it == vc.end() || it->first != key) continue;
				if (g.vocabToken[it->second] >= 0) ret.emplace_back((uint32_t)g.vocabToken[it->second]);
				collectFormTokens(fc, it->second, ret);
			}
		}

		vector<uint32_t> GrammarMatcher::allowedTokens() const
		{
			auto& g = *grammar;
			vector<uint32_t> ret;
			if (terminated) return ret;
			for (auto n : active)
			{
				for (auto [key, c] : nfa->children[n])
				{
					if (key != afterSenseKey && key >= specialBase) ret.emplace_back(key - specialBase);
				}
				collectFormTokens(n, 0, ret);
			}
			if (canTerminate()) ret.insert(ret.end(), g.eosTokens.begin(), g.eosTokens.end());
			sort(ret.begin(), ret.end());
			ret.erase(unique(ret.begin(), ret.end()), ret.end());
			return ret;
		}
	}
}
