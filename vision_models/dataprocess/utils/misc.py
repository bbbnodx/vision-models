from pathlib import Path
import json

from typing import Union, Dict, Any


Pathlike = Union[str, Path]


def load_dataset_info(json_path: Pathlike) -> Dict[str, Any]:
    """dataset_info.jsonを読み込んで辞書オブジェクトを返すヘルパー関数"""

    json_path = Path(json_path).resolve()
    with open(json_path, 'r', encoding='utf-8') as f:
        json_dict = json.load(f)

    # root_pathのデフォルト値をJSONファイルの親ディレクトリにする
    data_info = json_dict["dataset_information"]
    root_path = data_info.get("root_path", '')
    if not (isinstance(root_path, str) and root_path):
        data_info["root_path"] = str(json_path.parent)

    assert Path(data_info["root_path"]).is_dir(), f"`root_path` isn't a directory. Confirm {json_path}."

    return json_dict
