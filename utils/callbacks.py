class CallbackLog:
    def __init__(self, filename):
        self._fp = open(filename, 'w', buffering=1)

    def on_episode_end(self, model, rewards, loss):
        self._fp.write('{},{},{}\n'
                        .format(model.t, rewards[-1], loss))