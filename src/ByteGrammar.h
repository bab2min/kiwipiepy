#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace kiwi
{
	namespace guide
	{
		struct GraphMorph
		{
			std::string form;
			std::vector<uint32_t> tagTokens; /**< 형태 뒤에 하나를 고를 태그 토큰(미등록어처럼 품사를 정하지 않은 형태소는 여럿) */
			std::vector<uint32_t> senseTokens; /**< 태그 뒤에 하나를 고를 수 있는 번호 토큰 */
			bool senseRequired = false; /**< false면 번호 자리를 건너뛸 수 있고, true면 senseTokens 중 하나를 반드시 고른다 */
		};

		struct GraphEdge
		{
			uint32_t from = 0, to = 0;
			std::vector<GraphMorph> morphs;
		};

		class GrammarMatcher;

		/**
		 * @brief 토크나이저 정보. 형태소 그래프마다 GrammarMatcher를 만든다.
		 */
		class ByteGrammar : public std::enable_shared_from_this<ByteGrammar>
		{
			friend class GrammarMatcher;
		public:
			/**
			 * 형태 자리에서는 바이트를 이었을 때 형태와 같아지는 모든 토큰열을 허용한다(BPE 결과 토큰열만이 아니다).
			 * @param tokenBytes 토큰 ID별 바이트열. 일반 토큰이 아니면(특수 토큰 등) 빈 문자열
			 * @param specialTokens 태그·번호 토큰 ID(일반 토큰과 겹치지 않아야 한다)
			 */
			ByteGrammar(std::vector<std::string> tokenBytes, const std::vector<uint32_t>& specialTokens, std::vector<uint32_t> eosTokens);

			size_t vocabSize() const { return vocabSize_; }

			std::unique_ptr<GrammarMatcher> makeMatcher(uint32_t numStates, const std::vector<GraphEdge>& edges) const;

		private:
			std::vector<std::string> tokenBytes;
			std::vector<uint32_t> eosTokens;
			std::vector<uint8_t> isSpecial;
			size_t vocabSize_ = 0;

			// 일반 토큰의 바이트 trie(허용 토큰 계산용)
			std::vector<std::vector<std::pair<uint8_t, uint32_t>>> vocabChildren;
			std::vector<int64_t> vocabToken;
		};

		class GrammarMatcher
		{
		public:
			GrammarMatcher(std::shared_ptr<const ByteGrammar> grammar, uint32_t numStates, const std::vector<GraphEdge>& edges);

			/** @return 허용되지 않는 토큰이면 false(상태는 그대로) */
			bool accept(uint32_t token);
			std::vector<uint32_t> allowedTokens() const;
			bool canTerminate() const;
			bool isTerminated() const { return terminated; }
			void reset();

			const ByteGrammar* getGrammar() const { return grammar.get(); }
			size_t numNodes() const { return nfa->children.size(); }
			size_t numDroppedEdges() const { return nfa->droppedEdges; }

		private:
			// 그래프로 만든 오토마톤은 복제한 matcher끼리 공유한다(진행 상태만 따로 가진다)
			struct Automaton
			{
				std::vector<std::vector<std::pair<uint32_t, uint32_t>>> children; // 키 순 정렬
				std::vector<std::vector<uint32_t>> eps;
				std::vector<uint8_t> finals;
				uint32_t start = 0;
				size_t droppedEdges = 0;
			};
			std::shared_ptr<const ByteGrammar> grammar;
			std::shared_ptr<const Automaton> nfa;
			std::vector<uint32_t> active;
			bool terminated = false;

			uint32_t child(uint32_t node, uint32_t key) const;
			void close(std::vector<uint32_t>& nodes) const;
			void collectFormTokens(uint32_t formNode, uint32_t vocabNode, std::vector<uint32_t>& ret) const;
		};
	}
}
