from pathlib import Path
from platformdirs import user_data_dir

# 1. Define the path (Must match the installer script exactly)
# appname="jaxatari", appauthor="mycompany" (or whatever you used)
DATA_DIR = Path(user_data_dir("jaxatari"))
MARKER_FILE = DATA_DIR / ".ownership_confirmed"
LOCAL_SPRITES_DIR = Path(__file__).resolve().parent / "games" / "sprites"

def check_ownership():
    """
    Verifies that the user has accepted the license and confirmed ownership
    of the original hardware/software by looking for the marker file.
    """
    if MARKER_FILE.exists() or LOCAL_SPRITES_DIR.exists():
        return

    if not MARKER_FILE.exists():
        return

# ... rest of your package imports ...
from jaxatari.core import make, list_available_games
