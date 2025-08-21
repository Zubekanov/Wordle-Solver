import random

_ABSENT = 0b00
_PARTIAL = 0b01
_PRESENT = 0b10

# ANSI styles
_RESET = "\x1b[0m"
_BOLD = "\x1b[1m"
_FG_BLACK = "\x1b[30m"
_FG_BRIGHT_BLACK = "\x1b[90m"
_BG_GREEN = "\x1b[42m"
_BG_YELLOW = "\x1b[43m"

class WordleResult:
	def __init__(self, guess: str, result: list[int]):
		self.guess = guess.lower()
		self.result = result

	def __str__(self):
		parts = []
		for ch, state in zip(self.guess.upper(), self.result):
			if state == _PRESENT:
				style = _BG_GREEN + _FG_BLACK + _BOLD
			elif state == _PARTIAL:
				style = _BG_YELLOW + _FG_BLACK + _BOLD
			else:
				style = _FG_BRIGHT_BLACK + _BOLD
			parts.append(f"{style}{ch}{_RESET}")
		return " ".join(parts)

class Wordle:
	def __init__(self, word_list: list[str], target: str = None):
		if not word_list or len(word_list) == 0:
			raise ValueError("word_list is absent.")
		self.word_list = word_list
		if target:
			if target not in word_list:
				raise ValueError(f"Target word '{target}' is not in the word list.")
			self.target = target
		else:
			self.target = self.randomise_target()

	def randomise_target(self):
		self.target = random.choice(self.word_list)
	
	def set_target(self, target: str):
		if target not in self.word_list:
			raise ValueError(f"Target word '{target}' is not in the word list.")
		self.target = target
		
	def guess(self, word: str) -> WordleResult:
		if len(word) != 5:
			raise ValueError("Word must be exactly 5 letters long.")
		guess = word
		target = self.target

		result = [_ABSENT] * 5
		remaining: dict[str, int] = {}

		for i in range(5):
			if guess[i] == target[i]:
				result[i] = _PRESENT
			else:
				ch = target[i]
				remaining[ch] = remaining.get(ch, 0) + 1

		for i in range(5):
			if result[i] != _PRESENT:
				ch = guess[i]
				if remaining.get(ch, 0) > 0:
					result[i] = _PARTIAL
					remaining[ch] -= 1
				else:
					result[i] = _ABSENT

		return WordleResult(guess, result)
