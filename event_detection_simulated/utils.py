import logging 

from torch import nn


def setup_logger(
    name, log_file, formatter,
    level=logging.INFO
):
    handler = logging.FileHandler(log_file, mode='w')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def count_parameters(model: nn.Module) -> int:
    """Count the number of parameters in a PyTorch model

    Args:
        model (nn.Module): Model to count parameters of

    Returns:
        int: Number of parameters
    """
    return sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )