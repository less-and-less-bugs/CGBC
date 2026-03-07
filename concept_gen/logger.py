import logging

def setup_logger(log_file, level=logging.INFO):
    """Function to set up a shared logger."""
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # File handler
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Creating a root logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Avoid adding handlers multiple times
    if not logger.hasHandlers():
        logger.addHandler(handler)
        logger.addHandler(console_handler)

    return logger