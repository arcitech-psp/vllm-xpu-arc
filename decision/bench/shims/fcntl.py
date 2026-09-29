# Windows stand-in for POSIX fcntl, used only by jevbench/budget.py's ledger locking.
# Runs here are strictly serial (one process, one request at a time), so a no-op lock is safe.
LOCK_SH, LOCK_EX, LOCK_UN, LOCK_NB = 1, 2, 8, 4
def flock(fd, op):
    return None
