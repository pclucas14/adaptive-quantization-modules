"""
Progress bar
"""
from datetime import datetime

class Progress():
  """ Pretty print progress for neural net training """
  def __init__(self, batches, best_val=float('inf'), test_val=0, epoch=0,
        progress_bar=True, bar_length=30, track_best=True,
        custom_increment=False):
    """

    Args
        custom_increment: If True, must provide batch increment in `print_trian`
    """
    self.progress_bar = progress_bar # boolean
    self.bar_length = bar_length
    self.t1 = datetime.now()
    self.train_start_time = self.t1
    self.batches = batches
    self.current_batch = 0
    self.last_eval = '' # save last eval to add after train
    self.last_train = ''
    self.track_best = track_best
    self.best_val = best_val
    self.test_val = test_val
    self.epoch = epoch
    self.pause_bar = False
    self.bar_char = "#"
    self.custom_increment = custom_increment
    if custom_increment == True:
        self.pause_bar = True

  def epoch_start(self):
    print()
    self.t1 = datetime.now()
    self.epoch += 1
    self.current_batch = 0 # reset batch

  def train_end(self):
    print()

  def print_train(self, **kwargs):
    """
    Args:
        increment (int): used with custom increment
    """
    if self.custom_increment == True:
        msg = "You need to provide `increment=int` to `print_train"
        assert("increment" in kwargs), msg
        self.current_batch += kwargs["increment"]
    elif self.pause_bar == False:
      self.current_batch += 1
    t2 = datetime.now()
    epoch_time = (t2 - self.t1).total_seconds()
    total_time = (t2 - self.train_start_time).total_seconds()/60
    values = ""
    for k,v in kwargs.items():
        if k=="increment":
            continue
        values += '| {}: {:>3.4f}'.format(k,v)
    self.last_train='{:2.0f}: sec: {:>3.0f} | t-min: {:>5.1f} '.format(
        self.epoch, epoch_time, total_time) + values
    print(self.last_train, end='')
    self.print_bar()
    print(self.last_eval, end='\r')

  def print_cust(self, msg):
    """ Print anything, append previous """
    print(msg, end='')

  def test_best_val(self, te_acc):
    """ Test set result at the best validation checkpoint """
    self.test_val = te_acc

  def print_end_epoch(self, msg=""):
    """ Print after training , then new line """
    print(self.last_train, end='')
    print(msg, end='\r')

  def print_eval(self, value):
    # Print last training info
    print(self.last_train, end='')
    self.last_eval = '| last val: {:>3.4f} '.format(value)

    # If tracking eval, update best
    extra = ''
    if self.track_best == True:
      if value < self.best_val:
        self.best_val = value
      self.last_eval += '| best val: {:>3.4f}'.format(self.best_val)
    print(self.last_eval, end='\r')

  def print_bar(self):
    bars_full = int(self.current_batch/self.batches*self.bar_length)
    bars_empty = self.bar_length - bars_full
    progress ="| [{}{}] ".format(self.bar_char*bars_full, '-'*bars_empty)
    self.last_train += progress
    print(progress, end='')
