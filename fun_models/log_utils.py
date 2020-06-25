
class _BaseLogger(object):
  def log(*args, **kwargs):
    raise NotImplementedError('Please inheric from `LoggerBase` and implement the `log` method')

  def __call__(self, running_loss, running_metric, batch_idx,
               current_num_samples, is_train_mode):
    return self.log(running_loss, running_metric, batch_idx,
                    current_num_samples, is_train_mode)

class BatchPrintLogger(_BaseLogger):
  def __init__(self, prefix='\t-', postfix='\n', loss_name='loss',
               metric_name='accuracy', print_every=1):
    self.print_every = print_every
    self.prefix = prefix
    self.postfix = postfix
    self.loss_name = loss_name
    self.metric_name = metric_name

  def log(self, running_loss, running_metric, batch_idx, current_num_samples,
          is_train_mode):
    if (batch_idx + 1) % self.print_every != 0:
      return

    if is_train_mode:
      mode = 'Train'
    else:
      mode = 'Valid'
    print_str = f'{self.prefix} {mode} '
    print_str += f'- {self.loss_name}: {running_loss / current_num_samples:.2e} '
    print_str += f'- {self.metric_name}: {running_metric / current_num_samples:.2f}'
    print_str += f'{self.postfix}'

    print(print_str, end='')

