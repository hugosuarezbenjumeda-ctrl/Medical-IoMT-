# Start Here: Project + Conversation Context

Purpose: This is the first file to read in new sessions. It keeps a durable record of current work, decisions, and a timestamped conversation timeline.

How to use:
- Update `Right Now` whenever focus changes.
- Append all major work updates to `Recent Changes`.
- Add one entry per conversation/session in `Conversation Timeline`.
- Keep entries brief and factual.

## Required Logging Rule (All Conversations)
- Every new conversation that performs analysis, file edits, commands, or decisions MUST log its actions in this file.
- At minimum, each conversation MUST append one `Conversation Timeline` entry with: timestamp, user intent, actions taken, outcome, and next step.
- If technical changes were made, the conversation MUST also update: `Right Now`, `Recent Changes`, and `Session Summary (Most Recent)`.
- This file should be treated as the single source of truth for project continuity across sessions.

## Current Goal
- Objective:
- Done when:

## Right Now
- Working on:
- Branch:
- Files in focus:
- Blockers:

## Recent Changes
- YYYY-MM-DD HH:MM (TZ) - Change:
  - Why:
  - Commands run:
  - Result:

## Decisions
- Decision:
  - Reason:
  - Alternatives considered:

## Next Steps
1. 
2. 
3. 

## Open Questions / Risks
- 
- 

## Environment / Runbook
- Start:
- Test:
- Build:
- Deploy:
- Env vars:

## References
- Ticket:
- PR:
- Docs:
- Commits:

## Conversation Timeline
- 2026-03-05 00:00 (Europe/Madrid) - Context file initialization
  - User intent: Create a top-level file that future conversations can inspect quickly and include a timeline log of conversations.
  - Actions taken: Created `000_START_HERE_CONTEXT.md` at `/home/capstone15` with project-context sections and timeline format.
  - Outcome: Persistent start-here context file now exists at directory root and is ready for ongoing updates.
  - Next session should start with: Review this file first, update `Right Now`, and append a new timeline entry.

## Session Summary (Most Recent)
- Date:
- Summary:
- Immediate next action:
