from threading import Thread

from kiwipiepy import Kiwi

NUM_THREADS = 16

def test_multithread():
    print("Testing multi-threaded tokenization...")
    kiwi = Kiwi()
    def worker(results):
        for _ in range(30000):
            results.append(kiwi.tokenize("안녕하세요. 반갑습니다!"))
    
    all_results = [[] for _ in range(NUM_THREADS)]
    threads = []
    for i in range(NUM_THREADS):
        thread = Thread(target=worker, args=(all_results[i],))
        threads.append(thread)
        
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
        
    def _make_comparable(results):
        return [' '.join(token.tagged_form for token in result) for result in results]

    ref = _make_comparable(all_results[0])
    for results in all_results:
        assert _make_comparable(results) == ref

def test_tokenize_with_adding():
    print("Testing tokenization with adding...")
    kiwi = Kiwi()
    def worker(results):
        for _ in range(3000):
            results.append(kiwi.tokenize("안녕하세요. 반갑습니다!"))
    
    all_results = [[] for _ in range(NUM_THREADS)]
    threads = []
    for i in range(NUM_THREADS):
        thread = Thread(target=worker, args=(all_results[i],))
        threads.append(thread)
        
    for thread in threads:
        thread.start()

    for i in range(25):
        kiwi.add_user_word(f"word{i:5}", "NNP")

    for thread in threads:
        thread.join()
