import logging


def get_logger(name):
    #default = "__app__"
    logger = logging.getLogger(name)
    formatter = logging.Formatter('%(levelname)-8s: %(asctime)s | %(filename)-12s - %(funcName)-12s : %(lineno)-4s -- %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.setLevel(logging.DEBUG)

    return logger
