# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

If you discover a security vulnerability in YGN-SAGE, please report it responsibly:

1. **Do not** open a public GitHub issue
2. Email: yann.abadie@protonmail.com
3. Include: description, reproduction steps, impact assessment

We aim to respond within 48 hours and provide a fix within 7 days for critical issues.

## Security Considerations

YGN-SAGE executes LLM-generated code in sandboxed environments. The sandbox uses:
- Tree-sitter AST validation (static analysis)
- Process isolation with restricted permissions
- Configurable timeout limits (default 30s)

API keys are loaded from environment variables, never stored in code or committed to git.
