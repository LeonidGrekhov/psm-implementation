build_path = "build"
config_path = "config"
data_archive_dir = 'data_arch'


def get_log_config_file():
    return f'{config_path}/logging.conf'


def get_app_config_file_name():
    return f'{config_path}/app_config.json'


def temp_dir(name: str):
    return f'temp_{name}'
