import logging.config
from src.util import FileProvider as FP

logging.basicConfig(filename='test.log', level=logging.DEBUG)
logging.config.fileConfig(FP.get_log_config_file())

from src.main import main

if __name__ == '__main__':
    main()
