import warnings

import torch.cuda
from torch.utils.tensorboard import SummaryWriter

from evaluator import *
from evaluator import PatchEvaluator
from patch import *
from patch_config import *
from utils.frequency_tools import *
from utils.transforms import CMYK2RGB
from utils.utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
warnings.filterwarnings('ignore')

# set random seed
torch.manual_seed(2233)
torch.cuda.manual_seed(2233)
np.random.seed(2233)
