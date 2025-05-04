import re
import time
from difflib import SequenceMatcher

def load_dataset(path):
    buf = []
    for line in open(path, encoding='utf-8'):
        line = line.rstrip('\r\n')
        if not line:
            yield ''.join(buf), buf
            buf = []
        else:
            buf.append(line)
    if buf: yield ''.join(buf), buf

def evaluate_split(gold, pred):
    ws = re.compile(r'\s+')
    full_str = ''.join(ws.sub('', g) for g in gold)
    gold_spans = []
    pred_spans = []
    final_score = [0.0 for _ in gold]
    em = [0 for _ in gold]
    em_pred = [0 for _ in pred]
    pred_matches = [None for _ in pred]
    matches = []
    pred_ii = [[] for _ in pred]

    offset = 0
    for g in gold:
        l = len(ws.sub('', g))
        gold_spans.append((offset, offset + l))
        offset += l
    
    offset = 0
    for p in pred:
        p = ws.sub('', p)
        try:
            begin = full_str.index(p, offset)
            end = begin + len(p)
        except ValueError:
            diff = SequenceMatcher(a=full_str[offset:], b=p).get_opcodes()
            diff_start = next(i for i, (op, *_) in enumerate(diff) if op != 'delete')
            diff_end = -next(i for i, (op, *_) in enumerate(diff[::-1]) if op != 'delete') - 1
            begin = offset + diff[diff_start][1]
            end = offset + diff[diff_end][2]
        pred_spans.append((begin, end))
        offset = end
    
    start = 0
    for p, (pbegin, pend) in enumerate(pred_spans):
        matched = False
        for i in range(start, len(gold)):
            gbegin, gend = gold_spans[i]
            if pbegin == gbegin and pend == gend:
                final_score[i] = 1.0
                em[i] = 1
                em_pred[p] = 1
                pred_matches[p] = i
                break
            ibegin = max(pbegin, gbegin)
            iend = min(pend, gend)
            if ibegin < iend:
                r = 2*(iend - ibegin) / ((gend - gbegin) + (pend - pbegin))
                matches.append((r, i, p))
                pred_ii[p].append(len(matches) - 1)
                matched = True
            elif matched:
                break
            else:
                start += 1
    
    while True:
        try:
            s, i, p = max(matches)
        except: 
            break
        if final_score[i] < s:
            final_score[i] = s
            pred_matches[p] = i
        for n in pred_ii[p]:
            matches[n] = (0,)
    return final_score, em, em_pred, pred_matches

def run_evaluate(dataset, split_func, result_output=None, err_output=None):
    f1, norm_f1, em = [], [], []
    system_sents = 0
    elapsed = 0
    
    if result_output is not None:
        rout = open(result_output, 'w', encoding='utf-8')
    if err_output is not None:
        fout = open(err_output, 'w', encoding='utf-8')

    for text, gold in load_dataset(dataset):
        pcount = time.perf_counter()
        pred = split_func(text)
        elapsed += time.perf_counter() - pcount
        s, e, ep, pred_matches = evaluate_split(gold, pred)
        length_penalty = min(len(gold) / len(pred), 1)
        norm_s = [_s * length_penalty for _s in s]
        if result_output is not None:
            rout.write('\n'.join(pred) + '\n\n')
            rout.flush()
        if err_output is not None:
            for exact, sent, matched_gold in zip(ep, pred, pred_matches):
                if exact: continue
                r_score = None if matched_gold is None else s[matched_gold]
                m_gold = None if matched_gold is None else gold[matched_gold]
                fout.write(f"{r_score}\t{sent}\t{m_gold}\n")
            fout.write('\n')
            fout.flush()
        f1 += s
        norm_f1 += norm_s
        em += e
        system_sents += len(ep)
    
    if result_output is not None:
        rout.close()
    if err_output is not None:
        fout.close()

    print("[Sentence Split Benchmark] Dataset: " + dataset)
    print(
        f"Gold: {len(f1)} sents, "
        f"System: {system_sents} sents, "
        f"EM: {sum(em) / len(em):.5f}, "
        f"F1: {sum(f1) / len(f1):.5f}, "
        f"Normalized F1: {sum(norm_f1) / len(norm_f1):.5f}, "
        f"Latency: {elapsed*1000:.2f} msec"
    )
    print()
    return system_sents, em, f1, norm_f1

def baseline_splitter(text):
    sents = re.split(r'(?<=[.!?])\s', text)
    return sents

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('datasets', nargs='+')
    parser.add_argument('--write-result')
    parser.add_argument('--write-err')
    parser.add_argument('--model-path')
    parser.add_argument('--model-type', choices=['none', 'largest', 'knlm', 'sbg', 'cong', 'cong-global'])
    args = parser.parse_args()

    print('======== Baseline Splitter ========')
    all_system_sents = 0
    all_em = []
    all_f1 = []
    all_norm_f1 = []
    for dataset in args.datasets:
        system_sents, em, f1, norm_f1 = run_evaluate(dataset, baseline_splitter)
        all_system_sents += system_sents
        all_em += em
        all_f1 += f1
        all_norm_f1 += norm_f1
    
    print("[Overall]")
    print(
        f"Gold: {len(all_f1)} sents, "
        f"System: {all_system_sents} sents, "
        f"EM: {sum(all_em) / len(all_em):.5f}, "
        f"F1: {sum(all_f1) / len(all_f1):.5f}, "
        f"Normalized F1: {sum(all_norm_f1) / len(all_norm_f1):.5f}"
    )
    print()

    print('======== Kiwi.split_into_sents ========')
    all_system_sents = 0
    all_em = []
    all_f1 = []
    all_norm_f1 = []
    from kiwipiepy import Kiwi
    kiwi = Kiwi(model_path=args.model_path, model_type=args.model_type)
    kiwi.tokenize("foo-bar") # warm-up
    split = lambda text:[sent.text for sent in kiwi.split_into_sents(text, normalize_coda=True)]
    for dataset in args.datasets:
        system_sents, em, f1, norm_f1 = run_evaluate(dataset, split, args.write_result, args.write_err)
        all_system_sents += system_sents
        all_em += em
        all_f1 += f1
        all_norm_f1 += norm_f1
    
    print("[Overall]")
    print(
        f"Gold: {len(all_f1)} sents, "
        f"System: {all_system_sents} sents, "
        f"EM: {sum(all_em) / len(all_em):.5f}, "
        f"F1: {sum(all_f1) / len(all_f1):.5f}, "
        f"Normalized F1: {sum(all_norm_f1) / len(all_norm_f1):.5f}"
    )
