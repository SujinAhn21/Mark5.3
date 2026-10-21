# 출처
- 패키지: prodigyopt 1.1.2 (PyPI, MIT) — https://github.com/konstmish/prodigy
- 내려받은 파일: prodigyopt-1.1.2-py3-none-any.whl (sha256 4f8f79f71d37e4a501527fc30ecc847731369dc7cbe12e6a178157924e30ce03, 2026-10-22)
- 가져온 파일: prodigyopt/__init__.py · prodigyopt/prodigy.py · LICENSE — 수정 없음(원본 그대로)
  - md5: __init__.py e66f44c9bade2d4460d772b3fdca5442 · prodigy.py a2502bb51d305e09b444fd0697a90ea0 · LICENSE ed044d5ca4d2a3ffadbbec3404fa4950
  - 패키지 안의 prodigy_reference.py(참고 구현)는 쓰지 않아 넣지 않았다.
- 논문: Mishchenko & Defazio, "Prodigy: An Expeditiously Adaptive Parameter-Free Learner", ICML 2024 (arXiv 2306.06101)
- 코랩에 pip 로 설치된 prodigyopt 가 있어도 이 폴더가 먼저 잡히도록 학습 코드가 이 폴더를 sys.path 맨 앞에 넣고,
  불러온 모듈 파일 경로가 이 폴더인지 확인한다.

## 우리 학습에서 쓰는 설정 (2026-10-22 결정)
- Prodigy(params, lr=1.0, weight_decay=1e-4, decouple=False) — 나머지는 기본값(d0=1e-6, d_coef=1.0, bias correction 끔)
  - lr 은 학습률이 아니라 Prodigy 가 추정한 보폭 d 에 곱하는 배수다. 모든 파라미터 그룹이 같은 값이어야 한다(prodigy.py:151).
  - decouple=False: 지금까지의 Adam 과 같은 결합형 L2 weight decay(1e-4)를 유지한다(기본값은 AdamW 식 분리형).
- ReduceLROnPlateau(factor 0.5, patience 3)는 그대로 둔다. Prodigy 의 d 는 줄지 않으므로(prodigy.py:230-233)
  val 손실이 정체되면 스케줄러가 lr 배수를 절반으로 줄인다.
