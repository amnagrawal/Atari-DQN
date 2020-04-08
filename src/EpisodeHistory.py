from keras.callbacks import Callback


class EpisodeLogHistory(Callback):
    def on_train_begin(self, logs={}):
        self.episode_reward = []
        self.nb_episode_steps = []
        self.nb_steps = []

    def on_epoch_end(self, batch, logs={}):
        self.episode_reward.append(logs['episode_reward'])
        self.nb_episode_steps.append(logs['nb_episode_steps'])
        self.nb_steps.append(logs['nb_steps'])