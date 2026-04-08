## Approach

- Think before acting.
- Read files before editing them. Do not edit blind.
- Prefer small edits over rewrites.
- Do not re-read files unless they may have changed.
- Test before declaring work done.
- Keep solutions and responses simple, direct, and concise.
- User instructions always override this file.

## Output

- Return code first. Explanation after, only if non-obvious.
- No inline prose. Use comments only when the logic is not obvious.
- No boilerplate unless explicitly requested.

## Code Rules

- Simplest working solution. No over-engineering.
- No abstractions for single-use operations.
- No speculative features or "you might also want..."
- No docstrings or type annotations on code not being changed.
- No error handling for scenarios that cannot happen.
- Three similar lines is better than a premature abstraction.
- Add new imports at the top of the file, not inside functions.
- Remove dead code and unused imports, variables, constants, and functions immediately.

## Review Rules

- State the bug. Show the fix. Stop.
- No suggestions beyond the scope of the review.
- No compliments or filler.

## Debugging Rules

- Read the relevant code before explaining the bug.
- State what you found, where, and the fix.
- If the cause is unclear, say so. Do not guess.

## Git Rules

- Merge to `main` with a single squashed commit only.

## Simple Formatting

- No em dashes, smart quotes, or decorative Unicode symbols.
- Plain hyphens and straight quotes only.
- Natural language characters (accented letters, CJK, etc.) are fine when the content requires them.
- Code output must be copy-paste safe.
