import argparse
import os
import random
import time
import heapq
import math
import sys
import shutil
from wordle import *
from wordle import _ABSENT, _PARTIAL, _PRESENT

WORD_LIST_PATH = os.path.join(os.path.dirname(__file__), 'valid-wordle-words.txt')
WORD_LIST = []

vowels = set("aeiouy")

FIRST_GUESS_POOL = []
UNIQUE5 = set()

def build_static_pools() -> None:
	global FIRST_GUESS_POOL, UNIQUE5
	UNIQUE5 = {w for w in WORD_LIST if len(set(w)) == 5}
	FIRST_GUESS_POOL = [w for w in UNIQUE5 if sum(ch in vowels for ch in w) >= 2]

def load_word_list():
	"""Load the word list from the specified file."""
	try:
		with open(WORD_LIST_PATH, 'r') as file:
			words = [line.strip().lower() for line in file if line.strip()]
		return words
	except FileNotFoundError:
		print(f"Error: The word list file '{WORD_LIST_PATH}' was not found.")
		return []
	except Exception as e:
		print(f"An error occurred while loading the word list: {e}")
		return []

def build_frequency_list(words: list):
	position_counts = [{} for _ in range(5)]
	global_counts = {}

	for word in words:
		for i, char in enumerate(word):
			position_counts[i][char] = position_counts[i].get(char, 0) + 1
			global_counts[char] = global_counts.get(char, 0) + 1

	return position_counts, global_counts

def _k(ch: str) -> int:
	return ord(ch) - 97

class WordIndex:
	def __init__(self, words: list[str]):
		self.words = [w.lower() for w in words]
		self.n = len(self.words)
		if self.n == 0:
			raise ValueError("Empty word list")
		for i, w in enumerate(self.words):
			if len(w) != 5 or not w.isalpha():
				raise ValueError(f"Invalid word at index {i}: {w!r}")

		self.all_mask = (1 << self.n) - 1
		self.pos = [[0] * 26 for _ in range(5)]
		self.has = [0] * 26
		self.counts: list[tuple[int, ...]] = [None] * self.n
		self.index_by_word: dict[str, int] = {}
		self.letters: list[tuple[int, int, int, int, int]] = [None] * self.n

		for i, w in enumerate(self.words):
			self.index_by_word[w] = i
			bit = 1 << i
			cnt = [0] * 26
			lt = [0] * 5
			for p, ch in enumerate(w):
				k = _k(ch)
				lt[p] = k
				self.pos[p][k] |= bit
				cnt[k] += 1
			for k in range(26):
				if cnt[k]:
					self.has[k] |= bit
			self.counts[i] = tuple(cnt)
			self.letters[i] = tuple(lt)

	def iter_indices(self, mask: int):
		while mask:
			lsb = mask & -mask
			yield lsb.bit_length() - 1
			mask ^= lsb

	def filter(
		self,
		present: dict[int, str] = None,
		partial: dict[str, set[int]] = None,
		excluded: set[str] = None,
		min_counts: dict[str, int] = None,
		max_counts: dict[str, int] = None
	) -> list[str]:
		present = present or {}
		partial = partial or {}
		excluded = excluded or set()
		min_counts = min_counts or {}
		max_counts = max_counts or {}

		mask = self.all_mask

		for p, ch in present.items():
			k = _k(ch.lower())
			mask &= self.pos[p][k]
			if mask == 0:
				return []

		for ch in excluded:
			k = _k(ch.lower())
			mask &= ~self.has[k]
			if mask == 0:
				return []

		for ch, bad_positions in partial.items():
			k = _k(ch.lower())
			mask &= self.has[k]
			if mask == 0:
				return []
			for p in bad_positions:
				mask &= ~self.pos[p][k]
				if mask == 0:
					return []

		if not min_counts and not max_counts:
			return [self.words[i] for i in self.iter_indices(mask)]

		need_min = { _k(c.lower()): v for c, v in min_counts.items() }
		need_max = { _k(c.lower()): v for c, v in max_counts.items() }

		out = []
		for i in self.iter_indices(mask):
			cnt = self.counts[i]
			ok = True
			for k, v in need_min.items():
				if cnt[k] < v:
					ok = False
					break
			if ok:
				for k, v in need_max.items():
					if cnt[k] > v:
						ok = False
						break
			if ok:
				out.append(self.words[i])
		return out

_FALLBACK_SEEDS = ["crane", "slate", "trace", "pleat", "stare", "panel", "flame", "cater", "prone", "shore"]

_WORD_INDEX = None

_POW3 = (1, 3, 9, 27, 81)

def _letters_tuple(s: str) -> tuple[int, int, int, int, int]:
	return tuple(ord(c) - 97 for c in s)

def _encode_feedback_fast(idx: WordIndex, guess_letters: tuple[int, int, int, int, int], t_idx: int) -> int:
	"""
	Fast Wordle feedback: use precomputed target letters & counts.
	Returns code in base-3 with 0=grey,1=yellow,2=green (LS trit = pos 0).
	"""
	t_letters = idx.letters[t_idx]
	counts = list(idx.counts[t_idx])

	states = [0, 0, 0, 0, 0]

	for i in range(5):
		gl = guess_letters[i]
		if gl == t_letters[i]:
			states[i] = 2
			counts[gl] -= 1

	for i in range(5):
		if states[i] == 0:
			gl = guess_letters[i]
			if counts[gl] > 0:
				states[i] = 1
				counts[gl] -= 1

	return states[0]*_POW3[0] + states[1]*_POW3[1] + states[2]*_POW3[2] + states[3]*_POW3[3] + states[4]*_POW3[4]

def _get_index():
	global _WORD_INDEX
	if _WORD_INDEX is None and WORD_LIST:
		_WORD_INDEX = WordIndex(WORD_LIST)
	return _WORD_INDEX

def is_first_guessable(word: str) -> bool:
	return (
		len(set(word)) == 5 and
		sum(1 for ch in word if ch in vowels) >= 2
	)

def is_second_guessable(word: str, first_guess: str) -> bool:
	chars = set(word)
	if len(chars) != 5:
		return False
	if not chars.isdisjoint(set(first_guess)):
		return False
	return sum(1 for ch in word if ch in vowels) >= 2

def _position_freq(words: list[str]) -> list[dict[str, int]]:
	pos = [dict() for _ in range(5)]
	for w in words:
		for i, ch in enumerate(w):
			pos[i][ch] = pos[i].get(ch, 0) + 1
	return pos

def _score_presence_position(word: str, pfreq: dict[str, int], posfreq: list[dict[str, int]]) -> int:
	seen = set(word)
	dup_pen = 5 - len(seen)
	return sum(pfreq.get(ch, 0) for ch in seen) + sum(posfreq[i].get(word[i], 0) for i in range(5)) - dup_pen * 200

def _best_second_guess_from_pool(first_guess: str, answers_after_first: list[str]) -> str:
	first_set = set(first_guess)
	cands = [w for w in WORD_LIST if len(set(w)) == 5 and set(w).isdisjoint(first_set) and sum(ch in vowels for ch in w) >= 2]
	if not cands:
		return random.choice(WORD_LIST or _FALLBACK_SEEDS)
	pfreq = _presence_freq(answers_after_first or WORD_LIST)
	posfreq = _position_freq(answers_after_first or WORD_LIST)
	return max(cands, key=lambda w: _score_presence_position(w, pfreq, posfreq))

def _encode_feedback(guess: str, target: str) -> int:
	"""
	Base-3 feedback code: 0=grey, 1=yellow, 2=green; least-significant trit is pos 0.
	Matches Wordle duplicate rules (greens first, then yellows by remaining counts).
	"""
	guess = guess.lower()
	target = target.lower()

	states = [0] * 5
	rem = {}
	for i in range(5):
		if guess[i] == target[i]:
			states[i] = 2
		else:
			ch = target[i]
			rem[ch] = rem.get(ch, 0) + 1

	for i in range(5):
		if states[i] == 0:
			ch = guess[i]
			if rem.get(ch, 0) > 0:
				states[i] = 1
				rem[ch] -= 1

	code = 0
	p = 1
	for s in states:
		code += s * p
		p *= 3
	return code

def _derive_constraints(prev_guesses: list[WordleResult]):
	present: dict[int, str] = {}
	partial: dict[str, set[int]] = {}
	min_counts: dict[str, int] = {}
	max_counts: dict[str, int] = {}
	black_pos_by_letter: dict[str, set[int]] = {}

	def _count_letters(s: str) -> dict[str, int]:
		d: dict[str, int] = {}
		for c in s:
			d[c] = d.get(c, 0) + 1
		return d

	for wr in prev_guesses:
		g = wr.guess.lower()
		r = wr.result

		total_counts = _count_letters(g)
		non_grey_counts: dict[str, int] = {}

		for i, (ch, st) in enumerate(zip(g, r)):
			if st == _PRESENT:
				present[i] = ch
				non_grey_counts[ch] = non_grey_counts.get(ch, 0) + 1
			elif st == _PARTIAL:
				partial.setdefault(ch, set()).add(i)
				non_grey_counts[ch] = non_grey_counts.get(ch, 0) + 1
			else:
				black_pos_by_letter.setdefault(ch, set()).add(i)

		for ch, m in non_grey_counts.items():
			if m > min_counts.get(ch, 0):
				min_counts[ch] = m

		for ch, total in total_counts.items():
			m = non_grey_counts.get(ch, 0)
			if total > m:
				if ch in max_counts:
					max_counts[ch] = min(max_counts[ch], m)
				else:
					max_counts[ch] = m

	known_present = {ch for ch, v in min_counts.items() if v >= 1}
	for ch in known_present:
		if ch in black_pos_by_letter:
			partial.setdefault(ch, set()).update(black_pos_by_letter[ch])

	excluded = {ch for ch, v in max_counts.items() if v == 0}

	return present, partial, excluded, min_counts, max_counts

def _filter_nonhard_pool(words: list[str],
		present: dict[int, str],
		partial: dict[str, set[int]],
		excluded: set[str]) -> list[str]:
	idx = _get_index()
	return idx.filter(present=present, partial=partial, excluded=excluded)

def _adaptive_top_k(n_answers: int) -> tuple[int, int]:
	if n_answers > 3000:
		return (80, 120)
	if n_answers > 1200:
		return (100, 150)
	if n_answers > 400:
		return (150, 180)
	return (200, 220)

def _presence_freq(words: list[str]) -> dict[str, int]:
	pf = {chr(a): 0 for a in range(97, 123)}
	for w in words:
		seen = set(w)
		for ch in seen:
			pf[ch] += 1
	return pf

def _score_by_presence(word: str, pfreq: dict[str, int]) -> int:
	seen = set(word)
	dup_pen = 5 - len(seen)
	return sum(pfreq.get(ch, 0) for ch in seen) - dup_pen * 100

def _best_joint(idx: WordIndex, hard_candidates: list[str], probe_candidates: list[str], answers_idx: list[int]) -> tuple[str, float, str, float]:
	"""
	Evaluate both hard & probing candidate sets in one pass.
	Returns: (best_hard, hard_exp, best_overall, overall_exp)
	"""
	N = len(answers_idx)
	if N == 0:
		any_pool = probe_candidates or hard_candidates
		g = random.choice(any_pool)
		return g, float('inf'), g, float('inf')

	union = []
	seen = set()
	for lst in (hard_candidates, probe_candidates):
		for w in lst:
			if w not in seen:
				seen.add(w)
				union.append(w)

	best_hard, best_hard_exp = None, float('inf')
	best_any, best_any_exp = None, float('inf')
	hard_set = set(hard_candidates)

	buckets = [0] * 243
	used_codes = []
	rem = [0] * 26
	rem_touched = []

	pow3 = (1, 3, 9, 27, 81)
	letters = idx.letters

	def encode_fast(guess_letters: tuple[int, int, int, int, int], t_idx: int) -> int:
		t_letters = letters[t_idx]
		s0 = s1 = s2 = s3 = s4 = 0
		if guess_letters[0] == t_letters[0]: s0 = 2
		else:
			k = t_letters[0]; rem[k] += 1; rem_touched.append(k)
		if guess_letters[1] == t_letters[1]: s1 = 2
		else:
			k = t_letters[1]; rem[k] += 1; rem_touched.append(k)
		if guess_letters[2] == t_letters[2]: s2 = 2
		else:
			k = t_letters[2]; rem[k] += 1; rem_touched.append(k)
		if guess_letters[3] == t_letters[3]: s3 = 2
		else:
			k = t_letters[3]; rem[k] += 1; rem_touched.append(k)
		if guess_letters[4] == t_letters[4]: s4 = 2
		else:
			k = t_letters[4]; rem[k] += 1; rem_touched.append(k)

		if s0 == 0:
			k = guess_letters[0]
			if rem[k] > 0: s0 = 1; rem[k] -= 1
		if s1 == 0:
			k = guess_letters[1]
			if rem[k] > 0: s1 = 1; rem[k] -= 1
		if s2 == 0:
			k = guess_letters[2]
			if rem[k] > 0: s2 = 1; rem[k] -= 1
		if s3 == 0:
			k = guess_letters[3]
			if rem[k] > 0: s3 = 1; rem[k] -= 1
		if s4 == 0:
			k = guess_letters[4]
			if rem[k] > 0: s4 = 1; rem[k] -= 1

		for k in rem_touched:
			rem[k] = 0
		rem_touched.clear()

		return s0*pow3[0] + s1*pow3[1] + s2*pow3[2] + s3*pow3[3] + s4*pow3[4]

	for g in union:
		g_idx = idx.index_by_word.get(g)
		guess_letters = letters[g_idx] if g_idx is not None else _letters_tuple(g)

		sum_sq = 0
		used_codes.clear()

		for j, t_idx in enumerate(answers_idx):
			code = encode_fast(guess_letters, t_idx)
			old = buckets[code]
			if old == 0:
				used_codes.append(code)
			new = old + 1
			buckets[code] = new
			sum_sq += new*new - old*old

			remain = N - (j + 1)
			r_avail = 243 - len(used_codes)
			if r_avail <= 0:
				lb_add = remain * remain
			else:
				q, r = divmod(remain, r_avail)
				lb_add = r * (q + 1) * (q + 1) + (r_avail - r) * (q * q)
			lb_exp = (sum_sq + lb_add) / N

			if lb_exp >= best_any_exp or (g in hard_set and lb_exp >= best_hard_exp):
				sum_sq = None
				break

		for code in used_codes:
			buckets[code] = 0

		if sum_sq is None:
			continue

		exp_size = sum_sq / N
		if exp_size < best_any_exp:
			best_any_exp = exp_size
			best_any = g
		if g in hard_set and exp_size < best_hard_exp:
			best_hard_exp = exp_size
			best_hard = g

	if best_hard is None:
		best_hard, best_hard_exp = best_any, best_any_exp
	return best_hard, best_hard_exp, best_any, best_any_exp


def make_guess(prev_guesses: list[WordleResult], hard_mode: bool = False, first_guess = None):
	if len(prev_guesses) == 0:
		if not first_guess:
			if FIRST_GUESS_POOL:
				first_guess = random.choice(FIRST_GUESS_POOL)
			else:
				first_guess = random.choice(WORD_LIST) if WORD_LIST else random.choice(_FALLBACK_SEEDS)
		return first_guess
	
	if len(prev_guesses) == 1:
		if not WORD_LIST:
			return random.choice(_FALLBACK_SEEDS)
		idx = _get_index()
		present, partial, excluded, min_counts, max_counts = _derive_constraints(prev_guesses)
		answers1 = idx.filter(present=present, partial=partial, excluded=excluded, min_counts=min_counts, max_counts=max_counts)
		first_guess = prev_guesses[0].guess if not first_guess else first_guess
		return _best_second_guess_from_pool(first_guess, answers1 or WORD_LIST)

	idx = _get_index()
	if idx is None:
		return random.choice(_FALLBACK_SEEDS)

	present, partial, excluded, min_counts, max_counts = _derive_constraints(prev_guesses)

	answers = idx.filter(
		present=present,
		partial=partial,
		excluded=excluded,
		min_counts=min_counts,
		max_counts=max_counts
	)

	if not answers:
		cands = [w for w in WORD_LIST if is_first_guessable(w)] or WORD_LIST or _FALLBACK_SEEDS
		return random.choice(cands)

	if len(answers) == 1:
		return answers[0]

	pfreq = _presence_freq(answers)

	hard_sorted = sorted(answers, key=lambda w: _score_by_presence(w, pfreq), reverse=True)

	probe_pool = _filter_nonhard_pool(WORD_LIST, present, partial, excluded)
	probe_pool = [w for w in probe_pool if len(set(w)) == 5]
	probe_sorted = sorted(probe_pool, key=lambda w: _score_by_presence(w, pfreq), reverse=True)

	top_hard, top_probe = _adaptive_top_k(len(answers))
	hard_candidates = heapq.nlargest(
		min(top_hard, len(answers)),
		answers,
		key=lambda w: _score_by_presence(w, pfreq)
	)

	probe_candidates = heapq.nlargest(
		min(top_probe, len(probe_pool)),
		probe_pool,
		key=lambda w: _score_by_presence(w, pfreq)
	)

	answers_idx = [idx.index_by_word[w] for w in answers]
	hard_guess, hard_exp, any_guess, any_exp = _best_joint(idx, hard_candidates, probe_candidates, answers_idx)

	if hard_mode:
		return hard_guess

	return any_guess if any_exp < hard_exp else hard_guess

def play(wordle: Wordle, target: str = None, hard_mode: bool = False, first_guess: str = None, print_guesses: bool = False):
	if not WORD_LIST:
		raise ValueError("WORD_LIST is empty, cannot play Wordle.")

	if target is None: target = random.choice(WORD_LIST)
	
	wordle.set_target(target)

	if print_guesses:
		print(f"Target word: {target.upper()}")

	prev_guesses = []

	starting_time = time.time_ns()
	while True:
		guess = make_guess(prev_guesses, hard_mode, first_guess)
		result = wordle.guess(guess)
		prev_guesses.append(result)

		if print_guesses:
			print(result)

		if result.result == [_PRESENT] * 5:
			elapsed_time = (time.time_ns() - starting_time) / 1000000
			return {
				"guesses": len(prev_guesses),
				"starting_guess": first_guess,
				"elapsed_time": elapsed_time,
				"success": len(prev_guesses) <= 6
			}

def _status_line(msg: str) -> None:
	cols = shutil.get_terminal_size(fallback=(80, 24)).columns
	sys.stdout.write("\r" + msg.ljust(cols))
	sys.stdout.flush()

def run_benchmark(game: Wordle, runs: int, print_guesses: bool = False) -> dict:
	total_guesses = 0
	total_runtime = 0.0
	fails = 0

	live = not print_guesses

	for i in range(runs):
		stats = play(game, print_guesses=print_guesses)
		if not stats["success"]:
			fails += 1
		total_guesses += stats["guesses"] if stats["success"] else 6
		total_runtime += stats["elapsed_time"]

		if live:
			sofar = i + 1
			avg_guesses_so_far = total_guesses / sofar
			avg_runtime_so_far = total_runtime / sofar
			_status_line(f"Run {sofar}/{runs}  |  avg guesses {avg_guesses_so_far:.2f}  |  avg runtime {avg_runtime_so_far:.2f} ms")

	if live:
		sys.stdout.write("\n")

	avg_guesses = total_guesses / runs
	avg_runtime = total_runtime / runs
	fail_pct = (fails / runs) * 100.0

	return {
		"average_guesses": avg_guesses,
		"average_runtime_ms": avg_runtime,
		"fails": fails,
		"fail_pct": fail_pct,
		"runs": runs,
	}

def init_words() -> None:
	"""Load globals WORD_LIST / FREQ_LIST / GLOB_FREQ once."""
	global WORD_LIST
	WORD_LIST = load_word_list()
	build_static_pools()
	if not WORD_LIST:
		print("Failed to load word list.")
		raise SystemExit(1)

def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Wordle solver benchmark")
	parser.add_argument("-n", "--runs", type=int, default=100, help="number of games to run")
	return parser.parse_args()

def main() -> None:
	args = parse_args()
	init_words()

	game = Wordle(WORD_LIST)
	print_guesses = args.runs < 5

	summary = run_benchmark(game, args.runs, print_guesses=print_guesses)

	print(
		f"Average guesses: {summary['average_guesses']:.2f}, "
		f"Average runtime: {summary['average_runtime_ms']:.2f} ms, "
		f"Fails: {summary['fails']} ({summary['fail_pct']:.2f}%)"
	)

if __name__ == "__main__":
	main()