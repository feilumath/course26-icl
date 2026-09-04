# Contributing

This repository is a shared workspace for the course. Students can propose
edits to lecture notes and manage their own course projects here.

## Workflow

1. **Fork** this repository to your own GitHub account (or create a branch,
   if you have write access).
2. **Clone** your fork and create a topic branch for your change:
   ```
   git checkout -b my-change
   ```
3. Make your changes:
   - Lecture notes: edit files under `lecNotes/`. Keep the shared macros in
     `lecNotes/preamble.tex` consistent — don't redefine macros locally.
   - Projects: only add or edit files inside your own folder under
     `projects/`. See [`projects/README.md`](projects/README.md).
4. Commit your changes with a clear, descriptive commit message.
5. Push to your fork and open a **pull request** against this repository's
   default branch. Briefly describe what you changed and why.
6. The instructor (or a TA) will review and merge the pull request.

## Guidelines

- Keep pull requests focused: one lecture, one fix, or one project update
  per PR when possible.
- LaTeX sources should compile with `pdflatex`; please don't commit build
  artifacts (`.aux`, `.log`, `.out`, `.pdf`, etc. — these are covered by
  `.gitignore`).
- Be respectful of other students' contributions; see
  [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).
- If you find an error in someone else's project folder, open an issue or
  leave a PR review comment instead of editing their files directly.

## Reporting problems

If you find a typo, error, or broken link but don't want to fix it yourself,
please open a GitHub Issue describing the problem and where it is (file and
section).
