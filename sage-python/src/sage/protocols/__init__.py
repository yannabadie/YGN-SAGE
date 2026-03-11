"""Protocol support for MCP and A2A interoperability.

Both protocols are behind optional dependencies:
  pip install ygn-sage[mcp]        # MCP server
  pip install ygn-sage[a2a]        # A2A agent
  pip install ygn-sage[protocols]  # Both
"""

try:
    import mcp  # noqa: F401
    HAS_MCP = True
except ImportError:
    HAS_MCP = False

try:
    import a2a  # noqa: F401
    HAS_A2A = True
except ImportError:
    HAS_A2A = False
