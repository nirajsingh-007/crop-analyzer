import sys
import types

fcntl = types.ModuleType("fcntl")

# Define dummy constants
fcntl.F_GETFD = 1
fcntl.F_SETFD = 2
fcntl.F_SETLKW = 3
fcntl.F_SETLK = 4

# Define dummy functions
def noop(*args, **kwargs):
    return 0

fcntl.flock = noop
fcntl.ioctl = noop
fcntl.fcntl = noop

# Register fake module
sys.modules["fcntl"] = fcntl
