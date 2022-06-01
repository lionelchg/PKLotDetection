import logging

# Formatters
formats = {'small': logging.Formatter('%(asctime)s - %(message)s',
                            datefmt='%m-%d %H:%M'),
           'long': logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            datefmt='%m-%d %H:%M')}

def create_log(logname, logdir, logformat='small', console=False):
    """ Create logging handlers to file and console if specified """
    logger = logging.getLogger(logname)
    logger.setLevel(logging.DEBUG)

    # set formatter
    formatter = formats[logformat]

    # create file handler which logs even debug messages
    fh = logging.FileHandler(logdir / 'train.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if console:
        # create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        logger.addHandler(ch)

    return logger