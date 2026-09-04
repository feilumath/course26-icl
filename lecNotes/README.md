# Lecture Notes

This folder holds the LaTeX source for lecture notes, added as the semester
progresses.

## Organization

- One subfolder per lecture (or a small group of related lectures), named
  `lecXX_short-topic-name/`, e.g. `lec01_introduction/`.
- Each lecture folder contains at least a `lecXX.tex` file that
  `\input`s or `\usepackage`s the shared [`preamble.tex`](preamble.tex).
- Figures for a given lecture go in that lecture's own `figures/` subfolder.

```
lecNotes/
├── preamble.tex        # Shared macros, theorem environments, packages
├── template.tex         # Starting point for a new lecture's notes
└── lec01_introduction/
    ├── lec01.tex
    └── figures/
```

## Building

Each lecture note is a standalone document; compile it directly, e.g.

```
pdflatex lec01_introduction/lec01.tex
```

## Adding or editing notes

1. Copy [`template.tex`](template.tex) into a new `lecXX_topic/` folder and
   rename it, or edit an existing lecture file.
2. Use the shared macros in `preamble.tex` instead of redefining your own,
   so notation is consistent across lectures.
3. Open a pull request with your changes — see the repository's
   [`CONTRIBUTING.md`](../CONTRIBUTING.md).
