__version__ = "0.0.0"

import sys

if 'torch' in sys.modules and 't10' not in sys.modules:
    assert False, "Import t10n first, then torch to avoid openmp version conflicts"