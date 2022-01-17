from torch.utils.tensorboard import SummaryWriter


class LanguageModelingLogger(SummaryWriter):
    def __init__(self, log_dir):
        super().__init__(log_dir)

    def log_train(self, loss, lr, iter):
        self.add_scalar("train/loss", loss, lr, iter)

    def log_eval(self, loss, epoch):
        self.add_scalar("eval/perplexity", loss, epoch)
