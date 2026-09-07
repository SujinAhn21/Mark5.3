# fix_audio_length.py (리팩토링 버전)
# [변경 2026-07-11] 파일명 fix_audio_length_to_240000.py -> fix_audio_length.py (mark4와 통일).
# "N초로 강제 통일(자르기+패딩)" -> "최소 길이만 보장(패딩만, 자르지 않음)"으로 로직 변경.
# 이유: 파서(vild_parser_common.py)가 세그먼트 단위(1초 창, 0.5초 stride)로 salient_topk 5개를 뽑도록
# 이미 자체 정규화하고 있어서, 원본 클립 길이를 사전에 통일할 구조적 이유가 없음. 예전 방식(15초로
# 자르기)은 15초 넘는 클립의 뒷부분을 통째로 버려 타겟 소리가 잘려나가는 실제 버그였음(FSD50K 실측:
# 15초 초과 클립 17.6%). 반대로 너무 짧으면 파서가 세그먼트를 5개 못 채워 마지막 세그먼트를 복제하는
# 열화 폴백이 걸림 -> 최소 길이 보장만 하고 자르기는 없앰.

import os
import glob
import librosa
import soundfile as sf
import numpy as np
import argparse
from tqdm import tqdm

TARGET_SR = 16000
# segment_length=101프레임, segment_hop=50프레임, max_segments=5 기준: 서로 다른 5개 세그먼트를 뽑으려면
# 최소 101+4*50=301프레임 필요 -> hop_length=160/sample_rate=16000 -> 300*160=48,000샘플(정확히 3.0초).
TARGET_MIN_SAMPLES = 48000  # 3.0초 * 16000Hz (최소 보장 길이, 강제 통일 아님)

def process_audio_file(input_path, output_path):
    """오디오 파일을 로드하고, 최소 길이 미만이면 무음 패딩으로 채웁니다(자르기 없음)."""
    try:
        # librosa로 로드하면 자동으로 resample됨
        wav, sr = librosa.load(input_path, sr=TARGET_SR)

        if len(wav) < TARGET_MIN_SAMPLES:
            # 짧으면 최소 길이까지만 0으로 채우기 (패딩). 길어도 자르지 않음(뒷부분 소리 소실 방지).
            padding = np.zeros(TARGET_MIN_SAMPLES - len(wav), dtype=np.float32)
            wav = np.concatenate([wav, padding])

        # 출력 폴더가 없으면 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, wav, TARGET_SR, 'PCM_16')
        return True
    except Exception as e:
        print(f"[ERROR] 파일 처리 실패: {input_path} -> {e}")
        return False

def main(input_dir, output_dir):
    print("="*50)
    print("오디오 길이 정규화 시작...")
    print(f"입력 폴더: {input_dir}")
    print(f"출력 폴더: {output_dir}")
    print("="*50)
    
    if not os.path.isdir(input_dir):
        print(f"[ERROR] 입력 폴더를 찾을 수 없습니다: {input_dir}")
        return

    # 입력 폴더 내의 모든 .wav 파일을 재귀적으로 찾음
    audio_files = glob.glob(os.path.join(input_dir, "**", "*.wav"), recursive=True)
    
    if not audio_files:
        print("[WARNING] 처리할 .wav 파일이 없습니다.")
        return

    success_count = 0
    for file_path in tqdm(audio_files, desc="오디오 파일 처리 중"):
        # 입력 폴더 경로를 제외한 상대 경로를 구함
        relative_path = os.path.relpath(file_path, input_dir)
        # 출력 경로를 만듦 (하위 폴더 구조 유지)
        output_file_path = os.path.join(output_dir, relative_path)
        
        if process_audio_file(file_path, output_file_path):
            success_count += 1
            
    print("\n--- 결과 요약 ---")
    print(f"총 {len(audio_files)}개 파일 스캔 완료.")
    print(f"성공적으로 처리된 파일: {success_count}개")
    print(f"처리된 데이터 저장 위치: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="오디오 파일이 최소 3초(48,000 샘플) 미만이면 무음 패딩으로 채웁니다(자르기 없음).")
    parser.add_argument("--input_dir", type=str, required=True, help="처리할 원본 오디오 파일이 있는 폴더")
    parser.add_argument("--output_dir", type=str, required=True, help="처리된 파일을 저장할 폴더")
    args = parser.parse_args()
    
    main(args.input_dir, args.output_dir)