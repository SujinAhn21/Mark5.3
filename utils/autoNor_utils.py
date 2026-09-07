# autoNor_utils.py
# getnerate_dataset_index.py에서 자동정규화 용으로 쓰임.


def normalize_label(label: str) -> str:
    """
    주어진 라벨 문자열을 mark 버전에 따라 표준화된 라벨로 정규화.
    예: 'thumping_noise' -> 'thumping', 'other_noise_1' -> 'others'
    """
    label = label.lower().strip()

    label_map = {
        "thumping": ["thumping_noise", "thumping"],
        "others": [
            "other_noise_1", "other_noise_2", "other_noise_3", "other_noise_4", "others"
        ],
        "water": ["water_sound", "water"],
        "construction": ["construction_sound", "construction"],
        "daily_human": ["daily_human_noise", "daily_human"]
    }

    for normalized, aliases in label_map.items():
        if label in aliases:
            return normalized

    print(f"[normalize_label Warning] Unknown label encountered: '{label}' -> fallback to original.")
    return label


