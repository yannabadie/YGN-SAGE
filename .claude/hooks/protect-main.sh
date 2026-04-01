#!/bin/bash
INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty' 2>/dev/null)

if [ -z "$COMMAND" ]; then
    exit 0
fi

BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
if [[ "$BRANCH" == "main" || "$BRANCH" == "master" ]]; then
    if echo "$COMMAND" | grep -qE "^git (commit|push|reset|rebase)"; then
        echo "BLOCKED: Direct $COMMAND on protected branch '$BRANCH'. Use a feature branch." >&2
        exit 2
    fi
fi
exit 0
