import time

from sage.agent_loop import AgentEvent
from sage.events.bus import EventBus


def test_emit_redacts_api_key_payload_before_subscriber_receives_it(monkeypatch):
    monkeypatch.setenv("SAGE_REDACT_SECRETS", "1")
    bus = EventBus()
    received = []
    bus.subscribe(received.append)

    bus.emit(
        AgentEvent(
            type="TEST",
            step=0,
            timestamp=time.time(),
            meta={"payload": "secret sk-abcdefghijklmnopqrstuvwxyzABCDEF123456"},
        )
    )

    assert received[0].meta == {"payload": "secret [REDACTED:openai_api_key]"}
    assert bus.query()[0].meta == {"payload": "secret [REDACTED:openai_api_key]"}
