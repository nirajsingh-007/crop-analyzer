import sys
import types

termios = types.ModuleType("termios")

# Dummy constants expected by blessings or Earth Engine
termios.TCSANOW = 0
termios.TCSADRAIN = 1
termios.TCSAFLUSH = 2
termios.TIOCGWINSZ = 1074295912  # Arbitrary value, only to avoid ImportError

# Define no-op (empty) placeholder functions
def noop(*args, **kwargs):
    return 0

termios.tcgetattr = noop
termios.tcsetattr = noop
termios.tcflush = noop
termios.tcgetwinsize = noop

# Register this fake module so Python can import it
sys.modules["termios"] = termios
