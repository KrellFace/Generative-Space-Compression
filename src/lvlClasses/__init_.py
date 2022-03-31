#from levelWrapper import LevelWrapper
#from boxobanLevel import BoxobanLevel
#from marioLevel import MarioLevel
#from loderunnerLevel import LoderunnerLevel
from levelWrapperUpdateMethods import *

from os.path import dirname, basename, isfile, join
import glob

modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]#