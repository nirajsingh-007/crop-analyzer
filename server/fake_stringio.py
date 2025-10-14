import sys
import io

# Register the Python 3 io.StringIO as StringIO
sys.modules["StringIO"] = io
