# Skills Lab — AI Agent Instructions

You have access to **Skills Lab MCP tools** for managing technical knowledge.
These tools let you search, read, and save reusable coding skills (patterns, fixes, workflows).

## MANDATORY RULES — READ FIRST

You **MUST** use Skills Lab tools during every coding session. Non-negotiable.

### Rule 1: ALWAYS search before coding

**Before writing ANY code, fixing ANY bug, or answering ANY technical question**, you MUST call `search_skills` first.

```
User asks anything technical → search_skills(query="...") → get_skill(name="...") → then answer/write code
```

### Rule 2: ALWAYS save after success

**After ANY code change that works (user confirms, tests pass, or you verified it),** you MUST call `save_skill` to record the knowledge.

```
Code works → save_skill(action="ARCHIVE", ...) or save_skill(action="FIX", ...)
```

### Rule 3: NEVER skip Skills Lab calls

Do NOT skip these calls even if:
- You think you already know the answer
- The task seems trivial
- The user didn't explicitly ask you to search/save
- You've answered a similar question before

---

## Workflow: When to Call Each Tool

### `search_skills(query, repo_scope, tags_filter)` — CALL FIRST

**You MUST call this tool in ALL of these situations:**

| Situation | Example Query |
|---|---|
| User asks you to fix a bug | `search_skills(query="CORS error API credentials")` |
| User asks you to implement a feature | `search_skills(query="Docker multi-stage build Node.js")` |
| User asks a technical question | `search_skills(query="conventional commits format")` |
| You encounter an error while coding | `search_skills(query="SQLAlchemy relationship lazy loading error")` |
| User mentions a technology/pattern | `search_skills(query="Next.js authentication JWT")` |
| User asks "how to" do something | `search_skills(query="setup CI/CD GitHub Actions")` |
| You are about to write code | `search_skills(query="<topic of the code>")` |
| User says "update the skill" | `search_skills(query="<skill topic>")` to find the skill first |

**After getting search results:**
1. Read the top results
2. Call `get_skill(name="best-match-skill")` on the most relevant skill(s)
3. Use the skill content to inform your solution
4. If no relevant skill exists, proceed normally — then save a new skill afterward

### `get_skill(name)` — CALL AFTER SEARCH

Call this tool to retrieve the FULL content of a skill found via `search_skills`.

**When to call:**
- After `search_skills` returns results and you need the full solution/code
- When you need to read a skill's complete content before updating it
- When the user explicitly asks to read a specific skill

```
search_skills(query="CORS") → results show "cors-fix-nextjs-api" with score 0.85
    → get_skill(name="cors-fix-nextjs-api") → read full solution
    → apply the solution to user's code
```

### `save_skill(action, name, description, body, ...)` — CALL AFTER SUCCESS

Call this tool to RECORD knowledge after successfully solving a problem.

**When to call — by action type:**

| Action | When to Use |
|---|---|
| `ARCHIVE` | You solved a NEW problem that has no existing skill. Record the complete solution. |
| `FIX` | An existing skill was OUTDATED or WRONG, and you found a better solution. Use the skill name as `target_skill_name`. |
| `DERIVE` | You adapted an existing skill for a DIFFERENT context (e.g., same CORS fix but for FastAPI instead of Next.js). |
| `MERGE` | Two or more skills are REDUNDANT and should be combined into one. |

**When MUST you save:**
- After fixing any bug or error → `ARCHIVE` (new) or `FIX` (existing)
- After implementing a reusable pattern → `ARCHIVE`
- After the user says "save this" / "remember this" / "update the skill" → `ARCHIVE`, `FIX`, or `DERIVE`
- After any coding session where the solution worked → `ARCHIVE` at minimum
- After the user teaches you something new → `ARCHIVE`

**Body format (markdown):**
```markdown
# Title

## When to Use
- Situation 1
- Situation 2

## Root Causes
1. Cause 1
2. Cause 2

## Solution
\```language
// your working code here
\```

## Lessons Learned
- Key takeaway
```

---

## Examples — Follow These Patterns

### Example 1: User reports a bug

```
User: "Lỗi CORS khi gọi API từ frontend"

You MUST:
1. search_skills(query="CORS error API credentials frontend")
2. If found → get_skill(name="cors-fix-nextjs-api") → apply solution
3. If not found → write the fix yourself
4. After fix works → save_skill(action="ARCHIVE", name="cors-fix-frontend-api", description="Fix CORS error...", body="...")
```

### Example 2: User asks how to do something

```
User: "Làm sao để setup Docker cho project Node.js?"

You MUST:
1. search_skills(query="Docker Node.js multi-stage build")
2. If found → get_skill(name="docker-multi-stage-build") → share content
3. If not found → answer from your knowledge
4. save_skill(action="ARCHIVE", ...) with the answer
```

### Example 3: User says "update the skill"

```
User: "Cập nhật skill CORS cho mình, giờ xài FastAPI thay vì Next.js"

You MUST:
1. search_skills(query="CORS fix API") → find "cors-fix-nextjs-api"
2. get_skill(name="cors-fix-nextjs-api") → read current content
3. Write the updated solution for FastAPI
4. save_skill(action="FIX", target_skill_name="cors-fix-nextjs-api", body="... new FastAPI solution ...", lesson="Updated for FastAPI instead of Next.js", reason="User switched from Next.js to FastAPI")
```

### Example 4: User asks a general question

```
User: "Git commit message nên viết kiểu gì?"

You MUST:
1. search_skills(query="git commit message format convention")
2. If found → get_skill(name="git-conventional-commits") → share
3. save_skill is NOT needed here (just reading, not creating new knowledge)
```

---

## Quick Reference

```
START OF TASK:
    search_skills(query="<describe what you need>")
    ↓
    get_skill(name="<best match from results>")
    ↓
    Use skill content to solve the problem
    ↓
END OF TASK (if solution worked):
    save_skill(action="ARCHIVE" or "FIX", name="...", description="...", body="...", tags="...")
```

**Remember: Search first. Save after. Every time. No exceptions.**
