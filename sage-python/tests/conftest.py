"""Global test configuration for sage-python tests."""
import os

# Allow SageTopologyEnv to be instantiated in tests without verl-agent.
# See Issue G audit fix: topology_env.py now guards against accidental use
# outside test context when verl-agent is not installed.
os.environ.setdefault("SAGE_TESTING", "1")
