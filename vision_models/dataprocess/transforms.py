from torch import Tensor
import torchvision.transforms as T

import warnings


def get_transforms(datatype: str, json_dict):
    datatype = datatype.strip().lower()
    params = json_dict["transform_parameters"]

    # Resize
    img_size = json_dict["dataset_information"]["img_size"]
    resize = T.Resize(img_size)

    # Scaling to [0, 1] and normalize
    norm_params = params.get("normalize", {})
    if not norm_params:
        warnings.warn("No parameters at `normalize`. So Confirm configuration in JSON.")
    mean = norm_params.get("mean", [0.0])
    std = norm_params.get("std", [1.0])
    scaling_normalize = ScalingNormalize(mean=mean, std=std)

    # # Random Erasing
    # erase = params.get("random_erase", {})
    # erase_params = {
    #     "p": erase.get("p_erase", 0.0),
    #     "scale": erase.get("scale_range", (0, 0)),
    #     "ratio": erase.get("aspect_ratio", (1, 1)),
    #     "value": erase.get("fill_value", 0),
    # }

    # Transform pipeline
    if datatype == 'train':
        transforms = T.Compose([
            resize,
            T.RandomAffine(**params["random_affine"]),
            scaling_normalize,
        ])

    elif datatype in ('val', 'test'):
        transforms = T.Compose([
            resize,
            scaling_normalize,
        ])
    else:
        raise ValueError(f"`datatype` should be 'train', 'val' or 'test', but '{datatype}'.")

    return transforms


class ScalingNormalize(T.Normalize):
    """
    入力画像がuint型[0, 255]の場合、[0.0, 1.0]にスケーリングしてからMean subtractionをかけるラッパークラス
        Normalize(x_{c, i, j}) = (x_{c, i, j} - mean(x_c)) / std(x_c)
    """
    def forward(self, img: Tensor):
        img = img.float() / 255. if not img.dtype.is_floating_point else img
        return super().forward(img)
