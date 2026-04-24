import pytest

from sage.security.redaction import RedactionFilter, redact, redact_dict


def test_redact_openai_api_key():
    text = "token sk-proj-abcdefghijklmnopqrstuvwxyz1234567890ABCD done"

    redacted = redact(text)

    assert redacted == "token [REDACTED:openai_api_key] done"
    assert "sk-proj-" not in redacted


def test_redact_aws_access_key():
    text = "aws=AKIA1234567890ABCDEF"

    redacted = redact(text)

    assert redacted == "aws=[REDACTED:aws_access_key]"
    assert "AKIA1234567890ABCDEF" not in redacted


def test_redact_gcp_service_account_json_fragment():
    text = (
        'payload {"type": "service_account", "project_id": "demo", '
        '"private_key": "-----BEGIN PRIVATE KEY-----abc-----END PRIVATE KEY-----"}'
    )

    redacted = redact(text)

    assert redacted == "payload [REDACTED:gcp_service_account]"
    assert "PRIVATE KEY" not in redacted


def test_redact_bearer_token():
    text = "Authorization: Bearer abcdefghijklmnopqrstuvwxyz0123456789+/=="

    redacted = redact(text)

    assert redacted == "Authorization: [REDACTED:bearer_token]"
    assert "abcdefghijklmnopqrstuvwxyz" not in redacted


def test_redact_jwt_token():
    text = "jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signature123"

    redacted = redact(text)

    assert redacted == "jwt [REDACTED:jwt]"
    assert "eyJhbGci" not in redacted


def test_clean_text_is_unchanged():
    text = "normal execution log without credentials"

    assert redact(text) == text


def test_redaction_filter_can_be_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SAGE_REDACT_SECRETS", "0")
    text = "token sk-abcdefghijklmnopqrstuvwxyzABCDEF123456"

    assert RedactionFilter().redact(text) == text


def test_redact_dict_redacts_nested_values_without_mutating_original():
    data = {
        "outer": {
            "message": "key sk-abcdefghijklmnopqrstuvwxyzABCDEF123456",
            "items": ["Bearer abcdefghijklmnopqrstuvwxyz012345"],
        },
        "safe": 123,
    }

    redacted = redact_dict(data)

    assert redacted == {
        "outer": {
            "message": "key [REDACTED:openai_api_key]",
            "items": ["[REDACTED:bearer_token]"],
        },
        "safe": 123,
    }
    assert "sk-" in data["outer"]["message"]
