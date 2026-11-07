---
name: gcp
description: Commit and push task changes when the user explicitly says "gcp" or requests a git commit and push.
---

# Git commit and push

Use this skill only after an explicit user request to commit and push.

1. Inspect the current branch, upstream, worktree status, and complete diffs for staged, modified, deleted, and untracked files.
2. Identify newly added production code modules. Do not count tests, documentation, configuration, or skill files as code modules. If exactly one new module is present, use the commit subject `add <module_name>`, omitting its file extension. If there are multiple candidate modules, no new module, or any unclear scope, ask the user before staging or committing.
3. By default, stage all changes for the requested task. Exclude unrelated pre-existing files, local IDE state, secrets, environment files, and generated artifacts unless the user explicitly asks to include them. Ask if it is unclear whether a file belongs to the task.
4. Run the smallest relevant checks, then inspect the staged diff. Do not commit if the checks fail or the staged scope is unclear.
5. Commit on the current branch using the agreed subject. Include `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>` unless the user explicitly asks not to.
6. Push the current branch to its configured upstream. Never force-push, amend, or rewrite history. If the destination is unclear or the push fails, stop and report the issue.
7. Report the commit hash and push result.
