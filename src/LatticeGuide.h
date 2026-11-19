#pragma once
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace kiwi
{
	namespace guide
	{
		/**
		 * @brief 정렬 격자 가이드가 쓰는 토큰 표(토큰 ID별 바이트, 바이트열 → 토큰들)와 아는 단어 목록. 요청마다 만드는 LatticeFollower가 공유한다.
		 */
		class LatticeTokens
		{
		public:
			/**
			 * @param tokenBytes 토큰 ID별 바이트열. 일반 토큰이 아니면(추가 토큰) 빈 문자열
			 * @param knownWords 아는 단어(공백·띄어쓰기 _를 뺀 NFC). 격자에서 자모 편집 경로를 쓸 수 있는 형태. any_form 요청에서는 쓰지 않는다
			 */
			LatticeTokens(std::vector<std::string> tokenBytes, std::vector<std::string> knownWords);

			const std::string* bytesOf(uint32_t token) const
			{
				return token < tokenBytes.size() && !tokenBytes[token].empty() ? &tokenBytes[token] : nullptr;
			}
			struct TokenRange
			{
				const uint32_t* first = nullptr;
				const uint32_t* last = nullptr;
				const uint32_t* begin() const { return first; }
				const uint32_t* end() const { return last; }
			};

			/** bytes의 길이 maxLen 이하 앞부분 중 토큰인 것마다, 짧은 것부터 f(그 토큰들) */
			template<class Fn>
			void forEachPrefix(std::string_view bytes, size_t maxLen, Fn&& f) const
			{
				uint32_t node = 0;
				for (size_t i = 0; i < bytes.size() && i < maxLen; ++i)
				{
					if (!(node = child(node, (uint8_t)bytes[i]))) return;
					const auto r = tokensAt(node);
					if (r.first != r.last) f(r);
				}
			}
			/** 바이트열이 UTF-8 연속 바이트(0x80~0xBF) depth개인 토큰마다 f(그 토큰들) */
			template<class Fn>
			void forEachContinuation(size_t depth, Fn&& f) const { walkContinuation(0, depth, f); }
			bool isKnown(const std::string& word) const { return known.count(word) > 0; }
			bool hasKnownPrefix(const std::string& prefix) const;

		private:
			// 토큰 바이트열의 trie. 노드 0이 뿌리이고, 자식은 바이트 순으로 정렬된 trieEdges[edgeBegin, edgeEnd)
			struct TrieNode
			{
				uint32_t edgeBegin = 0, edgeEnd = 0, tokBegin = 0, tokEnd = 0;
			};

			// 없으면 0(뿌리는 누구의 자식도 아니다)
			uint32_t child(uint32_t node, uint8_t ch) const;
			TokenRange tokensAt(uint32_t node) const
			{
				auto& n = trie[node];
				return { trieTokens.data() + n.tokBegin, trieTokens.data() + n.tokEnd };
			}
			template<class Fn>
			void walkContinuation(uint32_t node, size_t depth, Fn& f) const
			{
				if (!depth)
				{
					const auto r = tokensAt(node);
					if (r.first != r.last) f(r);
					return;
				}
				for (uint32_t i = trie[node].edgeBegin; i < trie[node].edgeEnd; ++i)
				{
					const auto [ch, c] = trieEdges[i];
					if (ch > 0xBF) break;
					if (ch >= 0x80) walkContinuation(c, depth - 1, f);
				}
			}

			std::vector<std::string> tokenBytes;
			std::vector<TrieNode> trie;
			std::vector<std::pair<uint8_t, uint32_t>> trieEdges;
			std::vector<uint32_t> trieTokens;
			std::unordered_set<std::string> known;
			std::vector<std::string> knownSorted;
		};

		struct LatticeGroupSpec
		{
			uint32_t start = 0, end = 0;
			std::vector<std::string> forms;
		};

		/** 원문 [start, end) 구간을 그대로 옮긴 형태에 허용하는 태그 */
		struct FormTagSpan
		{
			uint32_t start = 0, end = 0;
			std::unordered_set<uint32_t> tags;
		};
		using FormTagTable = std::unordered_map<std::string, std::vector<FormTagSpan>>;

		/**
		 * @brief 문장 하나의 정렬 격자(generative_msa gmsa.lattice.Lattice와 같은 동작).
		 * 상태는 (원문 위치, Kiwi 묶음 진행) → 그 어절에서 쓴 자모 편집 비용. 형태 하나는 원문 복사(공백 무시), Kiwi 묶음(했 → 하+었),
		 * 아는 단어(anyForm이면 모든 형태)의 자모 편집으로 원문을 소비한다. 어절마다 비용 합은 budget 이하.
		 * 자모 편집 형태는 공백을 넘어 다음 어절까지 맞출 수 있고(여러 어절 고유명사 등), 그때 도착한 어절에는 그 형태의 편집 비용만 넘긴다.
		 * 자모 편집은 원문 글자뿐 아니라 묶음(그래프 간선)의 형태에도 잴 수 있다(원문 어여어어 → 묶음 어여 → 형태 어요, 비용 1).
		 * 편집은 한글 자모에만 쓰고, 한글이 아닌 글자는 그대로 일치해야 한다(예산을 넘겨 형태를 끝내는 경로는 예외).
		 * 원문의 보이지 않는 공백류 글자(NBSP, 폭 없는 공백, 점자 공백 등, 정답에 형태소로 남지 않는다)와 _(모델 출력 규칙에서 공백으로 쓸 수 있다)는
		 * 편집에서 비용 없이 건너뛸 수 있다.
		 * 다음 원문 글자가 기호이면 원문을 소비하지 않는 편집 형태(한글 삽입)는 쓸 수 없다(모델이 빠뜨린 기호를 먼저 쓰게 한다).
		 */
		class Lattice
		{
		public:
			struct Key
			{
				uint32_t pos = 0;
				int32_t group = -1; /**< 진행 중인 묶음(groups 번호), 없으면 -1 */
				uint32_t k = 0;     /**< 묶음에서 다음에 쓸 형태 번호 */
				bool inserted = false; /**< 이 위치에서 길이 0인 묶음(원문에 글자가 없는 형태소)을 이미 썼다. 다시 쓰지 못한다 */
				bool operator==(const Key& o) const { return pos == o.pos && group == o.group && k == o.k && inserted == o.inserted; }
			};
			using State = std::vector<std::pair<Key, int>>;

			Lattice(const std::string& text, const std::vector<LatticeGroupSpec>& groups, int budget,
				std::shared_ptr<const LatticeTokens> tokens, bool anyForm);

			State start() const;
			bool atEnd(const State& h) const;
			bool viable(const State& h, const std::string& buf) const;

			/**
			 * @brief 한 상태(h, buf)에서 여러 토큰의 viable(h, buf + 토큰)을 판정한다. 상태 키·편집 대상마다 buf까지의 편집 거리 행을
			 * 처음 쓸 때 한 번만 세고, 토큰마다 새로 완성된 글자의 자모만 덧붙인다. h는 이 객체보다 오래 살아야 한다.
			 */
			class Viability
			{
			public:
				/** maxTokenBytes: 판정할 토큰 바이트 길이의 예상 상한. 넘는 토큰은 viable()로 따로 판정한다 */
				Viability(const Lattice& lat, const State& h, const std::string& buf, size_t maxTokenBytes);
				bool operator()(const std::string& tokenBytes);

			private:
				struct Target
				{
					std::u32string_view b;
					std::vector<int> row;
					int limit;
					bool noPure;
					bool span; /**< 원문 구간: 토큰 길이에 따라 b의 앞부분만 쓴다 */
					uint32_t pos;
					int slack;
				};
				const Lattice* lat;
				const State* h;
				std::string buf, nb0;
				size_t p0 = 0, maxLen = 0;
				bool core0 = false, built = false;
				std::u32string E0;
				std::vector<Target> targets;
				std::vector<int> d, e;

				void build();
			};
			/** relaxed: 이어 쓸 길이 없을 때 쓰던 형태를 끝내기 위해, 예산을 넘겨도 원문을 한 글자 이상 소비하는 자모 편집 경로를 허용한다(비용은 예산으로 자른다) */
			State advance(const State& h, const std::string& buf, bool relaxed = false) const;
			std::vector<std::string> continuations(const State& h, const std::string& buf) const;
			State after(const std::vector<std::string>& forms) const;
			/** 묶음 밖의 상태에서 형태 buf를 원문 복사로 마칠 때 소비하는 원문 구간들 [(시작, 끝)] */
			std::vector<std::pair<uint32_t, uint32_t>> copySpans(const State& h, const std::string& buf) const;

		private:
			struct Group
			{
				uint32_t start = 0, end = 0;
				std::vector<std::string> bytes;
				std::vector<std::u32string> jamo; /**< 형태별 자모(그래프 형태를 자모 편집해 쓰는 경로의 비용 계산용) */
			};
			std::u32string T;
			size_t n = 0;
			std::string Tn;                     /**< 공백을 뺀 원문 바이트 */
			std::vector<size_t> off;            /**< 글자 위치 → Tn 바이트 위치 */
			std::unordered_map<size_t, uint32_t> byte2pos;
			std::vector<uint32_t> eend;         /**< 위치 → 그 어절의 끝(공백 위치) */
			std::vector<std::u32string> J;      /**< 글자별 자모(한글 음절은 초·중·종성으로) */
			std::u32string Jn;                  /**< 공백을 뺀 원문의 자모 */
			std::vector<size_t> jOff;           /**< 글자 위치 → Jn 위치 */
			std::vector<Group> groups;
			std::vector<std::vector<uint32_t>> groupsAt; /**< 시작 위치 → groups 번호 */
			int budget;
			std::shared_ptr<const LatticeTokens> tokens;
			bool anyForm;

			static void put(State& h, Key key, int c);
			void settle(State& h, uint32_t p, int c, bool inserted = false) const;
			bool usable(const Key& key, uint32_t gi) const { return !(key.inserted && groups[gi].start == groups[gi].end); }
			bool known(const std::string& w) const { return anyForm || tokens->isKnown(w); }
			bool knownPrefix(const std::string& b) const { return anyForm || tokens->hasKnownPrefix(b); }
			std::u32string jamoSpan(uint32_t a, uint32_t b) const; /**< 원문 [a, b)의 자모(공백 제외) */
			uint32_t reach(uint32_t i, size_t k) const; /**< i에서 공백이 아닌 글자 k개를 지난 위치 */
			bool symbolAt(uint32_t i) const; /**< 원문 i번째부터 공백·공백류 글자를 건너뛴 첫 글자가 한글이 아닌 기호·문자인가 */
		};

		/**
		 * @brief 한 요청의 진행 상태(generative_msa gmsa.lattice._LatticeFollower와 같은 동작).
		 * 형태 토큰은 바이트를 모으고, 태그 토큰에서 형태를 마쳐 격자 상태를 옮긴다. 허용 토큰이 없으면 그 뒤로 제약을 푼다(dead).
		 */
		class LatticeFollower
		{
		public:
			LatticeFollower(std::shared_ptr<const Lattice> lat, std::shared_ptr<const LatticeTokens> tokens,
				std::vector<uint32_t> tags, uint32_t eos, const std::vector<std::string>& prefix,
				FormTagTable formTags);

			// prepare()가 만든 viability가 H를 가리키므로 옮기지 않는다
			LatticeFollower(LatticeFollower&&) = delete;
			LatticeFollower& operator=(LatticeFollower&&) = delete;

			void consume(uint32_t token);
			/** @return 허용 토큰. 이어 쓸 수 있는 토큰이 없으면, 쓰던 형태가 온전하면 예산을 넘겨 끝내는 태그를(numRelaxed), 아니면 종료 토큰만(isStuck). 처음부터 격자 밖이면 nullopt(제약 없음) */
			std::optional<std::vector<uint32_t>> allowed(const std::vector<uint32_t>& top);
			/** allowed()에서 top과 관계없는 판정(태그·종료 토큰, 원문·묶음에서 이어 쓸 토큰, 끊긴 글자를 마무리하는 토큰)을 미리 한다.
			 * 다음 consume() 전까지 allowed()는 top 토큰만 더 판정한다 */
			void prepare();
			bool isDead() const { return dead; }
			bool isStuck() const { return stuck; }
			size_t numRelaxed() const { return relaxedCount; }
			const std::string& buffer() const { return buf; }

		private:
			std::shared_ptr<const Lattice> lat;
			std::shared_ptr<const LatticeTokens> tokens;
			std::vector<uint32_t> tags;
			std::unordered_set<uint32_t> tagSet;
			uint32_t eos;
			/** 형태 → 원문 구간별 허용 태그. 형태가 그 구간을 그대로 옮긴 것일 때만 태그를 제한한다 */
			FormTagTable formTags;
			Lattice::State H;
			std::string buf;
			bool dead = false;
			bool stuck = false; /**< 이어 쓸 토큰이 없어 종료 토큰만 허용한 적이 있다 */
			bool relaxedNext = false; /**< 다음 태그 토큰에서 형태를 예산을 넘겨 끝낸다 */
			size_t relaxedCount = 0;
			int done = -1; /**< 지금 buf로 형태를 마칠 수 있는가(-1: 모름) */
			bool prepared = false;
			std::vector<uint32_t> prepOut;
			std::unordered_set<uint32_t> prepSeen;
			std::unique_ptr<Lattice::Viability> viability;

			std::vector<uint32_t> tagsFor(const std::string& buf) const;
		};
	}
}
