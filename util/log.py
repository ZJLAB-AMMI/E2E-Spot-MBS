import logging


def set_logger(log_path, log_level=None):
    if log_level is not None:
        loglevel = eval('logging.' + log_level)
    else:
        loglevel = logging.INFO

    logger = logging.getLogger()
    logger.setLevel(loglevel)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

    logging.info('writting logs to file {}'.format(log_path))
