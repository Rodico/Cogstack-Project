import sys
from types import ModuleType

# Keep the pkg_resources mock so MedCAT can load smoothly
if 'pkg_resources' not in sys.modules:
    mock_pkg = ModuleType('pkg_resources')
    def dummy_get(*args, **kwargs):
        class MockDist:
            version = "1.7.0"
        return MockDist()
    mock_pkg.get_distribution = dummy_get
    mock_pkg.DistributionNotFound = Exception
    sys.modules['pkg_resources'] = mock_pkg

import os
from medcat.cdb import CDB
from medcat.vocab import Vocab
from medcat.config import Config

export_dir = r"C:\Users\mtmic\cogstack-fde-task\models\v1_export"
os.makedirs(export_dir, exist_ok=True)

print("Building clean legacy structures...")
config = Config()
cdb = CDB(config=config)
vocab = Vocab()

print("Saving using MedCAT's native serialization protocols...")
# Use MedCAT's built-in save methods instead of manual dill.dump!
cdb.save(os.path.join(export_dir, "cdb.dat"))
vocab.save(os.path.join(export_dir, "vocab.dat"))

print("Done! Perfectly structured native legacy files are ready.")