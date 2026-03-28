# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.3.x   | Yes       |
| < 0.3   | No        |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do NOT** open a public issue
2. Email: [leandropatodo@gmail.com](mailto:leandropatodo@gmail.com)
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Impact assessment
   - Suggested fix (if any)

You will receive a response within 48 hours. Critical issues will be patched within 7 days.

## Security Measures

- All SQL queries use parameterized bindings (sqlx)
- No user input in shell commands
- Secrets via environment variables only
- UTF-8 safe string truncation
- GDPR Right to Erasure (`cuba_forget`)
- 0 active CVEs (audited 2026-03-28)
