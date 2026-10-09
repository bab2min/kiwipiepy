import sys
import threading
import time
from threading import Thread

from kiwipiepy import Kiwi, MorphemeSet, TypoTransformer, basic_typos

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


SENTENCE = (
    "인장강도 시험은 STS304 시편을 대상으로 KS B 0802 기준에 따라 수행하며, "
    "측정값이 기준치를 벗어나면 검사책임자에게 보고한다. "
)
LONG_TEXT = SENTENCE * 1000  # long enough that its analysis runs without the GIL
SHORT_TEXT = "안녕하세요. 반갑습니다!"


class Heartbeat:
    """A thread that ticks every millisecond; the longest gap shows how long it was kept out."""

    def __init__(self):
        self.gaps = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        last = time.perf_counter()
        while not self._stop.is_set():
            time.sleep(0.001)
            now = time.perf_counter()
            self.gaps.append(now - last)
            last = now

    def __enter__(self):
        self._thread.start()
        time.sleep(0.01)
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def _blocked_share(fn):
    """How much of the time ``fn`` took the heartbeat spent waiting in one gap."""
    with Heartbeat() as beat:
        start = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - start
    return max(beat.gaps) / elapsed


def test_long_analysis_lets_other_threads_run():
    kiwi = Kiwi(num_workers=2)
    kiwi.tokenize(SHORT_TEXT)
    assert _blocked_share(lambda: kiwi.tokenize(LONG_TEXT)) < 0.5


def test_waiting_for_a_batch_lets_other_threads_run():
    kiwi = Kiwi(num_workers=2)
    kiwi.tokenize(SHORT_TEXT)
    assert _blocked_share(lambda: list(kiwi.tokenize([LONG_TEXT, LONG_TEXT]))) < 0.5


def _time_with_busy_thread(fn):
    """Seconds ``fn`` takes alone and next to a thread that keeps running Python bytecode."""
    start = time.perf_counter()
    fn()
    alone = time.perf_counter() - start

    stop = threading.Event()

    def busy():
        while not stop.is_set():
            for _ in range(1000):
                pass

    thread = threading.Thread(target=busy)
    thread.start()
    try:
        start = time.perf_counter()
        fn()
        contended = time.perf_counter() - start
    finally:
        stop.set()
        thread.join()
    return alone, contended


def test_short_analyses_keep_their_speed_next_to_a_busy_thread():
    # Giving up the GIL for a short analysis would make each call wait a whole switch
    # interval to get it back while another thread runs Python code.
    kiwi = Kiwi(num_workers=2)
    kiwi.tokenize(SHORT_TEXT)
    alone, contended = _time_with_busy_thread(lambda: [kiwi.tokenize(SHORT_TEXT) for _ in range(2000)])
    assert contended < alone * 5 + 0.5


def test_finished_batch_results_keep_their_speed_next_to_a_busy_thread():
    kiwi = Kiwi(num_workers=2)
    kiwi.tokenize(SHORT_TEXT)
    alone, contended = _time_with_busy_thread(lambda: list(kiwi.tokenize([SHORT_TEXT] * 20000)))
    assert contended < alone * 5 + 0.1


def test_analysis_while_its_inputs_change():
    # Analyses running without the GIL read the blocklist and the typo transformer while other
    # threads rebuild Kiwi, change both and re-initialise the typo transformer.
    kiwi = Kiwi(num_workers=2)
    blocklist = MorphemeSet(kiwi, ['고마움'])
    typos = basic_typos.copy()
    text = "고마움을 전합니다. " * 3000
    errors = []
    stop = threading.Event()

    def analyse():
        try:
            while not stop.is_set():
                assert kiwi.tokenize(text, blocklist=blocklist, typos=typos)[0].form == "고맙"
                for tokens in kiwi.tokenize([text[:3000]] * 4, blocklist=blocklist, typos=typos):
                    assert tokens[0].form == "고맙"
        except Exception as e:
            errors.append(e)

    workers = [threading.Thread(target=analyse) for _ in range(4)]
    for worker in workers:
        worker.start()
    try:
        for i in range(20):
            kiwi.add_user_word(f'신규용어{i}', 'NNP')
            blocklist._update(blocklist.set)
            typos |= TypoTransformer([])
            if getattr(sys, "_is_gil_enabled", lambda: True)():
                # Not on free-threaded builds: re-initialising an object writes back its old
                # header, lock included, under threads that hold that lock.
                typos.__init__([])
            time.sleep(0.05)
    finally:
        stop.set()
        for worker in workers:
            worker.join()
    assert not errors, errors[:3]


def test_error_from_the_input_reaches_the_caller():
    # The batch is torn down while its queued analyses still run; waiting for them must leave
    # the input's error in place.
    kiwi = Kiwi(num_workers=2)

    def texts():
        yield from [LONG_TEXT] * 8
        raise ValueError("input failed")

    try:
        list(kiwi.tokenize(texts()))
    except ValueError as e:
        assert str(e) == "input failed"
    else:
        assert False, "the input's error did not reach the caller"
