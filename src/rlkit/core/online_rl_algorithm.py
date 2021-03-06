import abc

import gtimer as gt
from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import (
    PathCollector,
    StepCollector,
)


class OnlineRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: StepCollector,
            evaluation_data_collector: PathCollector,
            replay_buffer: ReplayBuffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            # num_trains_per_train_loop,
            # num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
        )
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        # self.num_trains_per_train_loop = num_trains_per_train_loop
        # self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training

        # assert self.num_trains_per_train_loop >= self.num_expl_steps_per_train_loop, \
            # 'Online training presumes num_trains_per_train_loop >= num_expl_steps_per_train_loop'

    def _train(self):
        self.training_mode(False)
        # print("Exploring")
        if self.min_num_steps_before_training > 0:
            self.expl_data_collector.collect_new_steps(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
                isPPO=False
            )
            init_expl_paths = self.expl_data_collector.get_epoch_paths()
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

            gt.stamp('initial exploration', unique=True)
        print("Done initial Exploring")

        # num_trains_per_expl_step = self.num_trains_per_train_loop // self.num_expl_steps_per_train_loop
        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):  
            # test policy
            print("Testing policy for %d steps" % self.num_eval_steps_per_epoch)
            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )
            gt.stamp('evaluation sampling')

            # for _ in range(self.num_train_loops_per_epoch):
            # for _ in range(self.num_expl_steps_per_train_loop):
            print("Exploring %d steps with %d steps per path" % (self.num_expl_steps_per_train_loop, self.max_path_length))
            self.expl_data_collector.collect_new_steps(
                self.max_path_length,
                self.num_expl_steps_per_train_loop,  # num steps
                discard_incomplete_paths=False,
                isPPO=False
            )
            gt.stamp('exploration sampling', unique=False)

            self.training_mode(True)
            print("Training for %d steps" % (self.num_expl_steps_per_train_loop))

            for _ in range(self.num_expl_steps_per_train_loop):
                train_data = self.replay_buffer.random_batch(
                    self.batch_size)
                self.trainer.train(train_data)
            gt.stamp('training', unique=False)
            self.training_mode(False)
            # print("Training done")

            new_expl_paths = self.expl_data_collector.get_epoch_paths()
            self.replay_buffer.add_paths(new_expl_paths)
            gt.stamp('data storing', unique=False)
        
            self._end_epoch(epoch)
