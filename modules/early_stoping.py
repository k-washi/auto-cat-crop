class EarlyStopping():
    def __init__(self, n_epoch_stop) -> None:
        self._min_val_loss = 10 ** 10
        self._acc = 0.
        self._epoch = 0
        self._epochs_no_impove = 0
        self._early_stop = False
        self._n_epoch_stop = n_epoch_stop
        self._model_bath = ""

    def reset(self):
        self._epochs_no_impove = 0

    def epoch(self):
        return self._epoch

    def acc(self):
        return self._acc

    def loss(self):
        return self._min_val_loss

    def model_path(self):
        return self._model_bath

    def best_model_param(self, epoch, model_path="", acc=0):
        self._acc = acc
        self._epoch = epoch
        self._model_bath = model_path

    def check_loss(self, val_loss):
        if val_loss < self._min_val_loss:
            self.reset()
            self._min_val_loss = val_loss

            return False, True  # no early stop and best model

        self._epochs_no_impove += 1
        if self._n_epoch_stop == self._epochs_no_impove:
            return True, False  # early stop and no best model

        return False, False  # no early stop and no best model
