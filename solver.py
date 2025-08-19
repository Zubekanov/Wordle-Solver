import os
import random

WORD_LIST_PATH = os.path.join(os.path.dirname(__file__), 'valid-wordle-words.txt')
WORD_LIST = []  # WORD_LIST is a list of valid Wordle words
FREQ_LIST = []  # FREQ_LIST is a list of frequency dictionaries for each letter position
GLOB_FREQ = {}  # GLOB_FREQ is a dictionary of global letter frequencies

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

		for i, w in enumerate(self.words):
			bit = 1 << i
			cnt = [0] * 26
			for p, ch in enumerate(w):
				k = _k(ch)
				self.pos[p][k] |= bit
				cnt[k] += 1
			for k in range(26):
				if cnt[k]:
					self.has[k] |= bit
			self.counts[i] = tuple(cnt)

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

if __name__ == "__main__":
	WORD_LIST = load_word_list()
	if WORD_LIST:
		FREQ_LIST, GLOB_FREQ = build_frequency_list(WORD_LIST)
		print("Word list and frequency data loaded successfully.")
		print(f"Total words loaded: {len(WORD_LIST)}")
		# print three most common letters in each position
		for i, counts in enumerate(FREQ_LIST):
			sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
			top_three = sorted_counts[:3]
			print(f"Top 3 letters in position {i + 1}: {top_three}")
	else:
		print("Failed to load the word list.")
