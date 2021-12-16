from pytorch_lightning import loggers as pl_loggers

from pathlib import Path
import re


def get_tensorboard_logger(args) -> pl_loggers.LightningLoggerBase:
    """
    Logs directory is generated from `args` parameters.
    Training logs are saved in '/logs/{args.project}/{args.model}/{args.version}/'

    If args.version is None, logs are saved in '/logs/{args.project}/{args.model}/version_{i}/'
    """

    version = str(args.version) if args.version is not None else None
    version = args.model + '-' + version
    log_dir = Path(args.log_dir) / args.project / version
    while log_dir.exists():
        try:
            version = update_version(version)
        except AssertionError:
            raise ValueError('log_dir alrady exists and fail to update version.'
                             f'Confirm `args.version`: {args.version}.')
        log_dir = log_dir.parent / version

    return pl_loggers.TensorBoardLogger(save_dir=args.log_dir, name=args.project,
                                        version=version)


def update_version(version: str) -> str:
    """
    バージョン文字列を受け取り、末尾にバージョン数字がある場合は+1して返す
    末尾にバージョン文字列が無い場合、'_ver2'を付与して返す

    >>> update_version('v1')
    'v1_ver2'
    >>> update_version('v_1')
    'v_2'
    >>> update_version('x_ver2')
    'x_ver3'
    >>> update_version('x_version_3')
    'x_version_4'
    """

    ver_ptn = r'(_)((?:ver(?:sion)?)?)(_?)([0-9]+)$'
    match = re.search(ver_ptn, version)
    if match:
        # 末尾の数字のみを+1したversionを作成
        ver_num = int(match.groups()[-1]) + 1
        new_version = version[:match.start()] + ''.join(match.groups()[:-1]) + str(ver_num)
    else:
        new_version = version + '_ver2'

    assert new_version != version
    return new_version


if __name__ == '__main__':
    import doctest
    doctest.testmod()
