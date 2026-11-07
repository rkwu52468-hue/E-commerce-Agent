# Third-party notices

## Learn Claude Code

This project studies and adapts Agent mechanisms from:

- Repository: https://github.com/shareAI-lab/learn-claude-code
- Reference commit: `7226f555d48900076eba126acf37c9ed5f002b5d`
- Upstream license: MIT
- Copyright: Copyright (c) 2024 shareAI Lab

The full upstream license is preserved in [vendor/LICENSE.learn-claude-code](vendor/LICENSE.learn-claude-code).

[ vendor/snip.py ](vendor/snip.py) contains the recorded context-management methods from upstream `s08_context_compact/code.py`, with a project-specific minimal initializer. The source revision, hashes, retained methods and adaptation boundaries are documented in [vendor/SOURCE.json](vendor/SOURCE.json).

The project also adapts the educational mechanisms of the Agent loop, tools, hooks, planning, subagents, skills, memory, tasks, background execution, scheduling and teams into a simulated customer-service application. This is an AI-assisted learning and engineering project; it does not claim independent authorship of upstream course work.

## Runtime dependencies and model

Dependencies listed in `requirements*.txt` are installed separately rather than bundled in this repository. Each dependency retains its own license. The FastEmbed model is downloaded separately and remains subject to its model repository's terms and license. Model files and caches are not part of this source distribution.

## Repository-wide license

No separate repository-wide license has been selected for the remaining project code. The presence of an upstream MIT license does not apply that license to all files in this repository.
