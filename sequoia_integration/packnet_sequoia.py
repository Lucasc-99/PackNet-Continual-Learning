"""
Wrapper for PackNet integration into the Sequoia Research Tree Library
"""


from src.packnet import PackNet
from src.nets import MnistClassifier
from sequoia import Method, TaskIncrementalSetting


class PackNetMethod(Method, target_setting=TaskIncrementalSetting):
    ...  # Your code here.

    def fit(self, train_env, valid_env):
        # Train your model however you want here.



        self.trainer.fit(
            self.model,
            train_dataloader=train_env,
            val_dataloaders=valid_env,
        )

    def get_actions(self,
                    observations,
                    observation_space):
        # Return an "Action" (prediction) for the given observations.
        # Each Setting has its own Observations, Actions and Rewards types,
        # which are based on those of their parents.
        return self.model.predict(observations.x)

    def on_task_switch(self, task_id):
        # This method gets called if task boundaries are known in the current
        # setting. Furthermore, if task labels are available, task_id will be
        # the index of the new task. If not, task_id will be None.
        # For example, you could do something like this:
        self.model.current_output_head = self.model.output_heads[task_id]