from packnet.packnet_method import PackNetMethod
from packnet.nets import SequoiaClassifier
from sequoia.settings.sl import TaskIncrementalSLSetting

setting = TaskIncrementalSLSetting(
    dataset="mnist",
    increment=1
)

m = SequoiaClassifier(input_channels=3)
my_method = PackNetMethod(model=m, prune_instructions=.7, epoch_split=(3, 1))
results = setting.apply(my_method)
# results.make_plots()
