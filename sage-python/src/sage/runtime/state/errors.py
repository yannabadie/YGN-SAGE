class StateConflict(RuntimeError):
    """Reducer detected an irreconcilable conflict in a delta application.

    Only raised when raise_on_conflict=True; default is non-raising
    StateApplyResult(applied=False, conflicts=...).
    """
