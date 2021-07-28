from packnet.packnet_method import PackNetMethod
from packnet.nets import MnistClassifier
from sequoia.settings.sl import TaskIncrementalSLSetting


setting = TaskIncrementalSLSetting(
    dataset="mnist",
    increment=2
)

m = MnistClassifier(input_channels=3)
my_method = PackNetMethod(model=m, prune_instructions=.5, epoch_split=(5, 2))
results = setting.apply(my_method)
# results.make_plots()
