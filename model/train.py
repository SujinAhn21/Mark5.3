# train.py
import argparse
import os
import sys

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
UTILS_DIR = os.path.join(PROJECT_ROOT, "utils")
if UTILS_DIR not in sys.path:
    sys.path.append(UTILS_DIR)

# 유틸 및 학습 모듈 import
from seed_utils import set_seed
from train_mark5 import train_mark5

def main():
    parser = argparse.ArgumentParser(description="Train Teacher or Student model.")
    parser.add_argument(
        '--mode',
        type=str,
        choices=['student'],
        required=True,
        help="학습 모드 선택 (현재 student만 지원)"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="전역 랜덤 시드 값"
    )
    parser.add_argument(
        '--mark_version',
        type=str,
        default="mark5.0",
        help="모델 및 데이터셋 버전 (예: mark5.0, mark3_semisupervised_poc)"
    )
    args = parser.parse_args()

    set_seed(args.seed)

    if args.mode == "student":
        print(f"[INFO] ViLD-image Student 모델 ({args.mark_version}) 학습을 시작합니다. (Seed: {args.seed})")
        train_mark5(seed_value=args.seed, mark_version=args.mark_version)

if __name__ == "__main__":
    main()
