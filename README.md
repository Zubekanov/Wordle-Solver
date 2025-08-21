# Wordle Solver

Currently supports solving random Wordle problems

## Command-line options
- `-n <int>`, `--runs <int>`: number of games to run (default 100)
- `--plot`: save running-average plots and CSV to `plots/<timestamp>/`
- `--plot-dir <path>`: custom output directory (default `plots`)

# Benchmarks
Benchmarks obtained from running 10,000 Wordle games with randomly chosen solutions.
Included wordset has 14855 words, which are all 5 letters long. 
## Guesses to obtain solution
- Converges to 4.44 guesses.
  
|Running Average|Histogram|
|---------------|---------|
|<img width="768" height="576" alt="avg_guesses" src="https://github.com/user-attachments/assets/a29a6aed-bf62-4806-aa15-011a6e583d14" />|<img width="768" height="576" alt="hist_guesses" src="https://github.com/user-attachments/assets/0c084a3d-54e8-45b6-8509-999e5036ce6d" />|

## Runtime to obtain solution
- Converges to 47ms.

|Running Average|Histogram|
|---------------|---------|
|<img width="768" height="576" alt="avg_runtime_ms" src="https://github.com/user-attachments/assets/88f016cf-2cd5-4aa7-8586-9c5dd2a77d42" />|<img width="768" height="576" alt="hist_runtime_ms" src="https://github.com/user-attachments/assets/12ebf7a1-2796-448d-b555-88eca8bb006f" />|

## Failures to obtain solution (Guesses > 6)
- Converges to 8% failure rate.
- A “failure” in this report means the solver did not find the answer within the 6-guess limit (matching the game’s cap).

<p align="center">
  <img alt="fail_pct"
       src="https://github.com/user-attachments/assets/2cfd6a18-7651-4d70-8f2a-6afdb1a9e683"
       width="560">
</p>
