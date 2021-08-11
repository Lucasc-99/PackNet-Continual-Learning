from packnet.packnet import PackNet
from packnet.packnet_method import PackNetMethod
from sequoia import Method
from sequoia.settings.sl import TaskIncrementalSLSetting
from sequoia.methods.stable_baselines3_methods.a2c import A2CMethod
from pytorch_lightning import Trainer
import torch

agent = A2CMethod()


