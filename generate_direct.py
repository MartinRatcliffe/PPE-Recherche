r"""
Direct-prompting synthetic data generation for Rest14 ATE
(reproduction of the "Raw Prompting" condition, WASSA 2026, Sec. 3.2 / Table 1).

One Ollama call per attempt, no verifier agent, no polarity judge.
Only deterministic, model-free controls are applied (format, presence of the
requested terms, duplicates, round-trip through the repo's ATE loader).

All attempts are appended to <out_dir>/attempts.jsonl and valid examples to
<out_dir>/accepted.jsonl as they are produced, so the run can be resumed and the
exact inputs (aspects, polarities, style references, seeds) can be replayed for
the agentic condition.

Usage (PowerShell):
  .\.venv\Scripts\python.exe generate_direct.py -n_target 20
  .\.venv\Scripts\python.exe generate_direct.py -n_target 500 -build -unload
"""
import argparse
import ast
import json
import os
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone

import pandas as pd
import requests

from InstructABSA.data_prep import DatasetLoader

TRAIN_CSV = os.path.join('Dataset', 'SemEval14', 'Train', 'Restaurants_Train.csv')
ALLOWED_POLARITIES = ('positive', 'negative', 'neutral')
# Characters the repo loader (DatasetLoader.reconstruct_strings) or the ATE metric
# (split on ', ') cannot round-trip.
FORBIDDEN_TERM_CHARS = set("'\"[]{}:,")

# Table 1, "Raw Prompting" column. Adaptations (documented in the report):
#  * the good/bad examples are shown in the Terms=/Polarity= output format required
#    by the same prompt (the paper shows them as "[terms] [polarities]\nsentence");
#  * the requested aspect terms and polarities are appended at the end, because
#    Table 1 refers to "the aspect mentioned in the prompt" without showing the slot.
PROMPT_TEMPLATE = """You are a critic who can generate comments on the specified aspect and sentiment.
We would like you to complete a sentence generation task. Please follow these requirements:
- You need to use the sentiment, the aspect mentioned in the prompt
- Domain: Restaurants
- Your response must include:
1. The sentence.
2. A line that starts with Terms= followed by the list of aspect terms used.
3. A line that starts with Polarity= followed by the matching polarity list.
- ALL aspect terms must appear as actual aspects in the sentence with intended polarities
- Generated sentence must have the writing style and grammar structure and length of this sentence: {sent}
- The sentence should not have aspect words not specified in the prompt
- DO NOT repeat the input text in the output
- PRINT ONLY THE ANSWER TEXT — no explaining, nothing else
- Make sure to use aspect words in the output
Good Examples:
The prices were too high for this type of restaurant
Terms=['prices']
Polarity=['negative']

The Gnocchi was perfectly cooked and delicious, but the cheesecake was dry and flavorless.
Terms=['Gnocchi', 'cheesecake']
Polarity=['positive', 'negative']
Bad Example:
The udon soup was rich and flavorful.
Terms=['soup']
Polarity=['positive']
(term incorrect)
Make sure your output exactly follows this format.

Aspect terms: {terms}
Polarities: {polarities}"""


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def norm_key(text):
    t = text.lower().replace('’', "'")
    t = re.sub(r'\s+', ' ', t).strip()
    return re.sub(r'[\s\.\!\?;,]+$', '', t)


def load_train():
    df = pd.read_csv(TRAIN_CSV)
    df['asp'] = df['aspectTerms'].apply(ast.literal_eval)
    pairs = []          # one entry per real (term, polarity) occurrence -> realistic label distribution
    k_counts = Counter()
    for lst in df['asp']:
        real = [d for d in lst if d['polarity'] != 'none']
        if real:
            k_counts[len(real)] += 1
        for d in real:
            term = d['term'].strip()
            if d['polarity'] in ALLOWED_POLARITIES and term and not (set(term) & FORBIDDEN_TERM_CHARS):
                pairs.append((term, d['polarity']))
    # 1 to 4 aspects (paper), with the empirical Rest14 distribution truncated to 1..4
    ks = [1, 2, 3, 4]
    k_weights = [k_counts[k] for k in ks]
    styles = list(zip(df['sentenceId'].tolist(), df['raw_text'].tolist()))
    train_keys = {norm_key(s) for s in df['raw_text']}
    return pairs, ks, k_weights, styles, train_keys


def sample_policy(attempt, seed, pairs, ks, k_weights, styles):
    rng = random.Random(seed * 1_000_003 + attempt)
    k = rng.choices(ks, weights=k_weights, k=1)[0]
    chosen, seen = [], set()
    while len(chosen) < k:
        term, pol = rng.choice(pairs)
        low = term.lower()
        # distinct terms, and no term nested in another (keeps the presence check unambiguous)
        if low in seen or any(low in s or s in low for s in seen):
            continue
        seen.add(low)
        chosen.append((term, pol))
    style_id, style_text = rng.choice(styles)
    return {
        'k': k,
        'terms': [t for t, _ in chosen],
        'polarities': [p for _, p in chosen],
        'style_sentence_id': int(style_id),
        'style_sentence': style_text,
        'call_seed': (seed * 1_000_003 + attempt) % (2**31 - 1),
    }


def build_prompt(policy):
    return PROMPT_TEMPLATE.format(sent=policy['style_sentence'],
                                  terms=repr(policy['terms']),
                                  polarities=repr(policy['polarities']))


def parse_list(raw):
    raw = raw.strip()
    try:
        val = ast.literal_eval(raw)
        if isinstance(val, (list, tuple)):
            return [str(v).strip() for v in val]
        if isinstance(val, str):
            return [val.strip()]
    except Exception:
        pass
    inner = raw.strip('[]')
    return [p.strip().strip('\'"').strip() for p in inner.split(',') if p.strip()]


def parse_response(text):
    """Return (sentence, terms, polarities) or raise ValueError(reason)."""
    text = text.replace('’', "'").replace('‘', "'").replace('“', '"').replace('”', '"').replace("\\'", "'")
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    t_idx = [i for i, l in enumerate(lines) if re.match(r'^\**\s*(?:\d\.\s*)?terms\s*=', l, re.I)]
    p_idx = [i for i, l in enumerate(lines) if re.match(r'^\**\s*(?:\d\.\s*)?polarit(?:y|ies)\s*=', l, re.I)]
    if len(t_idx) != 1:
        raise ValueError('format:terms_line')
    if len(p_idx) != 1:
        raise ValueError('format:polarity_line')
    sent_lines = [l for l in lines[:t_idx[0]]]
    sent_lines = [re.sub(r'^(?:\d\.\s*)?(?:sentence\s*:\s*)?', '', l, flags=re.I).strip() for l in sent_lines]
    sent_lines = [l for l in sent_lines if l]
    if len(sent_lines) != 1:
        raise ValueError('format:sentence_lines')
    if lines[t_idx[0] + 1:] and max(t_idx[0], p_idx[0]) < len(lines) - 1:
        raise ValueError('format:trailing_text')
    sentence = sent_lines[0].strip().strip('"').strip()
    if len(sentence.split()) < 2:
        raise ValueError('format:empty_sentence')
    terms = parse_list(lines[t_idx[0]].split('=', 1)[1])
    pols = [p.lower() for p in parse_list(lines[p_idx[0]].split('=', 1)[1])]
    return sentence, terms, pols


def find_span(sentence, term):
    m = re.search(r'(?<!\w)' + re.escape(term) + r'(?!\w)', sentence, flags=re.I)
    return m.group(0) if m else None


def loader_roundtrip(sentence, aspect_terms):
    """Feed the row through the repo's real ATE loader and return the label it produces."""
    row = pd.DataFrame({'raw_text': [sentence], 'aspectTerms': [str(aspect_terms)]})
    out = DatasetLoader().create_data_in_ate_format(row, 'term', 'raw_text', 'aspectTerms', '', '')
    return out['labels'].iloc[0]


def validate(raw, policy, dup_keys, train_keys):
    sentence, terms, pols = parse_response(raw)
    req_terms = [t.lower() for t in policy['terms']]
    if sorted(t.lower() for t in terms) != sorted(req_terms):
        raise ValueError('format:terms_not_requested')
    if len(pols) != len(terms):
        raise ValueError('format:polarity_length')
    declared = dict(zip([t.lower() for t in terms], pols))
    if any(declared[t.lower()] != p for t, p in zip(policy['terms'], policy['polarities'])):
        raise ValueError('format:polarity_not_requested')
    spans = []
    for term in policy['terms']:
        span = find_span(sentence, term)
        if span is None:
            raise ValueError('presence:term_missing')
        spans.append(span)
    key = norm_key(sentence)
    if key == norm_key(policy['style_sentence']):
        raise ValueError('duplicate:style_sentence')
    if key in train_keys:
        raise ValueError('duplicate:train_sentence')
    if key in dup_keys:
        raise ValueError('duplicate:synthetic')
    aspect_terms = [{'term': s, 'polarity': p} for s, p in zip(spans, policy['polarities'])]
    if loader_roundtrip(sentence, aspect_terms) != ', '.join(spans):
        raise ValueError('loader:roundtrip')
    return sentence, aspect_terms


def call_ollama(url, model, prompt, call_seed, num_predict, keep_alive):
    r = requests.post(f'{url}/api/generate', json={
        'model': model, 'prompt': prompt, 'stream': False, 'keep_alive': keep_alive,
        # decoding: model defaults from the Ollama Modelfile; only the seed (reproducibility)
        # and an output cap are set
        'options': {'seed': call_seed, 'num_predict': num_predict},
    }, timeout=600)
    r.raise_for_status()
    return r.json()


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def append_jsonl(path, rec):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def build_csvs(accepted, n_target, out_dir, synth_csv, mixed_csv):
    acc = sorted(accepted, key=lambda r: r['attempt'])[:n_target]
    if len(acc) < n_target:
        sys.exit(f'Only {len(acc)} accepted examples, need {n_target}.')
    synth = pd.DataFrame({
        'sentenceId': [r['id'] for r in acc],
        'raw_text': [r['raw_text'] for r in acc],
        'aspectTerms': [str(r['aspectTerms']) for r in acc],
        'aspectCategories': ['[]'] * len(acc),
    })
    real = pd.read_csv(TRAIN_CSV)
    mixed = pd.concat([real, synth], ignore_index=True)
    synth.to_csv(synth_csv, index=False)
    mixed.to_csv(mixed_csv, index=False)
    # check both files with the real ATE loader
    for path in (synth_csv, mixed_csv):
        df = pd.read_csv(path)
        out = DatasetLoader().create_data_in_ate_format(df, 'term', 'raw_text', 'aspectTerms', '', '')
        print(f'{path}: {len(df)} rows, {out["labels"].str.len().gt(0).sum()} non-empty ATE labels')
    print('Synthetic CSV:', synth_csv)
    print('Mixed CSV    :', mixed_csv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-n_target', type=int, default=500)
    ap.add_argument('-model', default='qwen2.5:7b')
    ap.add_argument('-seed', type=int, default=2026)
    ap.add_argument('-out_dir', default=os.path.join('Output', 'synthetic', 'rest14_direct_qwen2.5-7b'))
    ap.add_argument('-ollama_url', default='http://localhost:11434')
    ap.add_argument('-num_predict', type=int, default=200)
    ap.add_argument('-max_attempts', type=int, default=5000)
    ap.add_argument('-build', action='store_true', help='write the synthetic and mixed CSVs once n_target is reached')
    ap.add_argument('-synth_csv', default=os.path.join('Output', 'rest14_direct_synthetic_500.csv'))
    ap.add_argument('-mixed_csv', default=os.path.join('Output', 'rest14_direct_mixed_3541.csv'))
    ap.add_argument('-unload', action='store_true', help='unload the model from the GPU at the end')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    attempts_path = os.path.join(args.out_dir, 'attempts.jsonl')
    accepted_path = os.path.join(args.out_dir, 'accepted.jsonl')
    meta_path = os.path.join(args.out_dir, 'run_config.json')

    pairs, ks, k_weights, styles, train_keys = load_train()
    attempts = read_jsonl(attempts_path)
    accepted = read_jsonl(accepted_path)
    next_attempt = (max(a['attempt'] for a in attempts) + 1) if attempts else 0
    dup_keys = {norm_key(r['raw_text']) for r in accepted}
    print(f'Resume: {len(attempts)} attempts, {len(accepted)} accepted; next attempt #{next_attempt}')

    if not os.path.exists(meta_path):
        show = requests.post(f'{args.ollama_url}/api/show', json={'model': args.model}, timeout=60).json()
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump({'created': now(), 'model': args.model, 'seed': args.seed,
                       'num_predict': args.num_predict, 'modelfile_parameters': show.get('parameters'),
                       'model_details': show.get('details'), 'n_pairs_pool': len(pairs),
                       'k_values': ks, 'k_weights': k_weights, 'prompt_template': PROMPT_TEMPLATE},
                      f, ensure_ascii=False, indent=2)

    t0 = time.time()
    while len(accepted) < args.n_target and next_attempt < args.max_attempts:
        policy = sample_policy(next_attempt, args.seed, pairs, ks, k_weights, styles)
        prompt = build_prompt(policy)
        rec = {'attempt': next_attempt, 'time': now(), **policy}
        try:
            resp = call_ollama(args.ollama_url, args.model, prompt, policy['call_seed'], args.num_predict, '30m')
        except requests.RequestException as e:
            sys.exit(f'Ollama call failed at attempt {next_attempt}: {e} (nothing recorded; rerun to resume)')
        rec['raw_response'] = resp.get('response', '')
        try:
            sentence, aspect_terms = validate(rec['raw_response'], policy, dup_keys, train_keys)
            rec['status'] = 'accepted'
            rec['reason'] = None
        except ValueError as e:
            rec['status'] = 'rejected'
            rec['reason'] = str(e)
        append_jsonl(attempts_path, rec)
        if rec['status'] == 'accepted':
            acc = {'id': f'synth_direct_{len(accepted) + 1:04d}', 'attempt': next_attempt,
                   'raw_text': sentence, 'aspectTerms': aspect_terms,
                   'requested_terms': policy['terms'], 'requested_polarities': policy['polarities'],
                   'style_sentence_id': policy['style_sentence_id'], 'style_sentence': policy['style_sentence'],
                   'call_seed': policy['call_seed'], 'model': args.model, 'raw_response': rec['raw_response'],
                   'time': rec['time']}
            append_jsonl(accepted_path, acc)
            accepted.append(acc)
            dup_keys.add(norm_key(sentence))
        next_attempt += 1
        if next_attempt % 10 == 0:
            el = time.time() - t0
            print(f'attempt {next_attempt}: accepted {len(accepted)}/{args.n_target} ({el:.0f}s)', flush=True)

    attempts = read_jsonl(attempts_path)
    reasons = Counter(a['reason'] or 'accepted' for a in attempts)
    print(f'Total attempts: {len(attempts)}, accepted: {len(accepted)}')
    print('Outcome by reason:', dict(reasons.most_common()))

    if args.build and len(accepted) >= args.n_target:
        build_csvs(accepted, args.n_target, args.out_dir, args.synth_csv, args.mixed_csv)
    if args.unload:
        requests.post(f'{args.ollama_url}/api/generate', json={'model': args.model, 'keep_alive': 0}, timeout=120)
        print('Unload requested for', args.model)


if __name__ == '__main__':
    main()
