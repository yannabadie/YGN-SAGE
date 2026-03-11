"""Test protocol package feature detection."""


def test_protocol_package_importable():
    import sage.protocols
    assert hasattr(sage.protocols, "HAS_MCP")
    assert hasattr(sage.protocols, "HAS_A2A")
    assert isinstance(sage.protocols.HAS_MCP, bool)
    assert isinstance(sage.protocols.HAS_A2A, bool)


def test_has_mcp_reflects_import():
    """HAS_MCP is True only if mcp is importable."""
    try:
        import mcp  # noqa: F401
        expected = True
    except ImportError:
        expected = False
    from sage.protocols import HAS_MCP
    assert HAS_MCP is expected


def test_has_a2a_reflects_import():
    """HAS_A2A is True only if a2a is importable."""
    try:
        import a2a  # noqa: F401
        expected = True
    except ImportError:
        expected = False
    from sage.protocols import HAS_A2A
    assert HAS_A2A is expected
