import os
import copy
import json
import time
import torch
import argparse
import soundfile as sf
from tqdm import tqdm
from diffusers import DDPMScheduler
from audioldm_eval import EvaluationHelper
from models import build_pretrained_models, AudioDiffusion
from transformers import AutoProcessor, ClapModel
import torchaudio
from tango import Tango
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import IPython
from IPython.display import Audio, display
import torch
import librosa, librosa.display
from tqdm import tqdm
import json


tango = Tango("declare-lab/tango-full-ft-audiocaps")


text = "A drumstick is continuously hitting a hard metal surface in a large room"

# attention_weights = None
attention_weights = torch.from_numpy(np.array([1 for i in range(17)])).cuda()

wav = tango.generate(prompt=text, steps=100, disable_progress=False, attention_weights=attention_weights).astype(np.float16)