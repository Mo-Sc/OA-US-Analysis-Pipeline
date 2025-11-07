import numpy as np


def calculate_mso2(hb_scan, hbo2_scan, nan_invalid=False, clip=True):
    """
    Calculate the mso2 value from the scan as mso2 = HbO2 / (Hb + HbO2)
    if nan_invalid is set to True, all values outside the range [0, 1] are set to nan
    if clip is set to True, all values outside the range [0, 1] are clipped to the range [0, 1]
    """

    thb = hb_scan + hbo2_scan
    # thb[thb == 0] = np.nan  # to avoid div by 0

    mso2 = hbo2_scan / thb

    if nan_invalid:
        mso2[mso2 > 1] = np.nan
        mso2[mso2 < 0] = np.nan
    elif clip:
        mso2[mso2 > 1] = 1
        mso2[mso2 < 0] = 0
        mso2[np.isnan(mso2)] = 0

    return mso2


def calculate_ratio(scan1, scan2, epsilon=1e-6, clip=[-1, 1], zero_nan=False):
    """
    Calculate the ratio of two scans as scan1 / scan2
    if zero_nan is set to True, all values where scans are nan are set to 0
    if clip is set, all values outside the range are clipped to the range
    """
    ratio = scan1 / (scan2 + epsilon)

    if zero_nan:
        ratio[np.isnan(ratio)] = 0

    if clip is not None:
        ratio[ratio < clip[0]] = clip[0]
        ratio[ratio > clip[1]] = clip[1]

    return ratio


def infer_channel_names_from_shape(shape):
    """
    Infer OA channel names based on common dataset shapes.
    Just for legacy support; it's better to provide channel names explicitly in metadata.
    """

    from v2.data.constants import MSOT_2_DEFAULT_WAVS, MSOT_3_DEFAULT_WAVS

    if shape[0] == 2:
        oa_channels = ["Hb", "HbO2"]
    elif shape[0] == 3:
        oa_channels = ["700", "850", "980"]
    elif shape[0] == 13:
        oa_channels = MSOT_2_DEFAULT_WAVS
    elif shape[0] == 41:
        oa_channels = MSOT_3_DEFAULT_WAVS

    else:
        raise ValueError(
            f"Cannot infer channel names for shape {shape}. Please provide channel names in metadata."
        )
    return oa_channels
