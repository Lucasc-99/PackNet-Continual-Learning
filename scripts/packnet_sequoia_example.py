from packnet.packnet_method import PackNetMethod
from packnet.nets import SequoiaClassifier, SmallerSequoiaClassifier, MnistClassifier
from sequoia.settings.sl import TaskIncrementalSLSetting

setting = TaskIncrementalSLSetting(
    dataset="mnist"
)

m = SmallerSequoiaClassifier(input_channels=3)
m2 = MnistClassifier(input_channels=3)


my_method = PackNetMethod(model=m, prune_instructions=.7, epoch_split=(1, 1))
results = setting.apply(my_method)
# results.make_plots()
