# ROI Timing Framework: simulation code and results

Companion code for the paper "ROI Timing Framework: Assessing Marketing Returns under Contractual Commitments and Delayed Revenue" (David Gaidukovych, 2026).

## Contents
- `rtf_simulation.py`: reproduces every table and figure in the paper and writes all quoted numbers to `results.json`.
- `results.json`: output of the script as used in the paper.
- `fig1_kernels.png` to `fig6_stopping.png`: figures.
- `RTF_article_EN.md`: manuscript source (Markdown with LaTeX math).
- `build_docx.py`: builds the Word manuscript from the Markdown source with pandoc.

## Run
Python 3.11, NumPy >= 1.26, SciPy >= 1.12, Matplotlib >= 3.8.

    pip install numpy scipy matplotlib
    python rtf_simulation.py

The random seed is fixed (20260915); a run takes about two minutes and regenerates `results.json` and the figures.

## License
MIT.
