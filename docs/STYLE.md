# Documentation Style Guide

## Structure

Lead with the answer or the command. Don't open a page with "This page explains" or "In this section you will learn".

Use imperative mood in tutorials and how-tos: "Run", "Set", "Open". Use declarative sentences elsewhere.

## Sentence style

Plain declarative sentences. No rule-of-three adjectives. No "not X but Y" or "X isn't a bug, it's a feature" constructions. No dramatic one-line declaratives.

Cut intensifiers and fillers. The banned list includes: ``explicitly``, ``exactly``, ``fully``, ``simply``, ``seamlessly``, ``robust``, ``powerful``, ``rigorous``, ``comprehensive``, ``crucially``, ``importantly``.

Em dashes: at most one per page, and only where a comma or full stop won't do.

## Formatting

Bold only for UI or CLI element names on first use, not for every defined term.

Every number says where it came from (measured in which run or config, or computed from what).

Define each term once, in the glossary. Elsewhere, link to it.

## Placeholders

No placeholder text: no `<UPPER_CASE_TOKENS>`, no `[foo placeholder]`, no `TODO`, no `TBD`. If a value is not yet known, omit the field rather than print a placeholder.

## Numbers and measurements

State the source: "measured on a loopback pair with 200 paced datagrams" not "typically around 30 ms". If a value comes from a run, name the config it came from.

## Code blocks

All config snippets must be valid YAML that passes `cadet validate`. Use `literalinclude` pointing to files under `examples/` or `docs/snippets/` in preference to inline code blocks.

## Module docstrings

Keep the four pieces of information (purpose, contents, role in CADET, configuration) as a natural opening paragraph, not as four fixed headings.

## Home page

One or two plain sentences about what CADET does. No marketing.
