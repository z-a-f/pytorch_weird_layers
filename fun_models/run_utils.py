
import torch

from metric_utils import number_of_correct_metric

def run_epoch(model, loader, loss_criterion, metric_fn=None, optimizer=None,
              device=None, logger_fn=None):
  '''Runs a single training/inference epoch.

  Args:
    model: Model to run
    loader: DataLoader instance
    loss_criterion: Loss function to be used
    metric_fn: Metric function to be used.
               The result is accumulated and averaged before returning.
               Default: Number of correctly classified
    optimizer: Optimizer instance. If not set, this function is run in inference
    device: Device to run on
    logger_fn: Logging function. Receives following arguments:
               1. Running loss
               2. Running metric
               3. Batch number
               4. Number of samples processed so far
               5. Training flag (True if training, False if inference)
  Returns:
    - Loss, normalized to the number of samples
    - Metric, normalized to the number of samples
    - Logger results -- if the `logger_fn` returns non-None, contains the list
      of every return value from it.
  '''
  train = (optimizer is not None)
  # print("Training Mode: ", train)
  running_loss = 0.0
  running_metric = 0.0
  running_logger = []
  sample_num = 0

  if metric_fn is None:
    metric_fn = number_of_correct_metric

  if device is None:
    device = next(model.parameters()).device
  else:
    model = model.to(device)

  # Disable grad if not in training mode
  prev_grad_mode = torch.is_grad_enabled()
  torch._C.set_grad_enabled(train)
  model.train(train)

  for idx, data in enumerate(loader, 0):
    x, y = data
    x = x.to(device)
    y = y.to(device)

    y_hat = model(x)
    loss = loss_criterion(y_hat, y)

    if train:
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    running_metric += metric_fn(y_hat, y)
    running_loss += loss.item()
    sample_num += len(x)

    if logger_fn is not None:
      log = logger_fn(running_loss, running_metric, idx, sample_num, train)
      if log is not None:
        running_logger.append(log)

  if logger_fn is not None:
      log = logger_fn(running_loss, running_metric, -1, sample_num, train)
      if log is not None:
        running_logger.append(log)

  # print("Training Mode:", train, "")

  loss = running_loss / sample_num
  metric = running_metric / sample_num

  # Reset the grad mode to the previous state
  torch._C.set_grad_enabled(prev_grad_mode)

  return loss, metric, running_logger
