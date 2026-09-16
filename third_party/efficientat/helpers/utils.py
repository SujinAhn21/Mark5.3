# [사본 축약 2026-09-05] 원본 helpers/utils.py 에서 NAME_TO_WIDTH 만 남겼다.
# 원본은 import 시점에 metadata/class_labels_indices.csv 를 현재 폴더 기준 상대경로로 열어(원본 38행)
# 저장소 루트 밖에서 import 하면 FileNotFoundError 로 죽는다. models/mn/model.py 가 이 파일에서 쓰는 것은
# NAME_TO_WIDTH 뿐이므로 그 함수만 원본 그대로 옮겼다(1~33행, 수정 없음).

def NAME_TO_WIDTH(name):
    mn_map = {
        'mn01': 0.1,
        'mn02': 0.2,
        'mn04': 0.4,
        'mn05': 0.5,
        'mn06': 0.6,
        'mn08': 0.8,
        'mn10': 1.0,
        'mn12': 1.2,
        'mn14': 1.4,
        'mn16': 1.6,
        'mn20': 2.0,
        'mn30': 3.0,
        'mn40': 4.0,
    }

    dymn_map = {
        'dymn04': 0.4,
        'dymn10': 1.0,
        'dymn20': 2.0
    }

    try:
        if name.startswith('dymn'):
            w = dymn_map[name[:6]]
        else:
            w = mn_map[name[:4]]
    except:
        w = 1.0

    return w

