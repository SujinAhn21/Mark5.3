"""Clotho 2.1(Zenodo 4783391) 압축 3개를 받아 샤워 21개만 꺼낸다(MD5 대조 뒤 압축 삭제).
사용: python clotho_pick.py [<외부 자료 폴더>/clotho]   (기본 ./clotho) · 필요: pip install py7zr
"""
import sys, os, hashlib, urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pylib"))   # [10-25] 따로 깐 py7zr 가 있으면 그것, 없으면 일반 설치본
import py7zr
X = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "clotho")   # [10-25] 옛 판은 C드라이브 경로를 코드에 고정했다
os.makedirs(X + "/audio", exist_ok=True)
WANT = {
 "evaluation": ["ShowerAndSoap.wav", "shower taking.wav", "Turning on Shower 2_1-2.wav", "Shower Running 01.wav", "Shower and walk - front.wav", "RBH_Household_shower 03.wav"],
 "validation": ["Douche aan.wav"],
 "development": ["showering shower different intensities water bathroom WC.wav", "Shower turning on and off with drain noise.wav", "Whistling in the shower.wav",
                 "Turning on Shower_1-2.wav", "Hotel room shower.wav", "Getting Into Shower and Turning On.wav", "Shower, A.wav", "Bathroom shower.wav",
                 "JM_HOME&amp;OFFICE_Shower 01 - Taking a shower.wav", "Shower 2.wav", "Shower 1.wav", "untitled_shower 2.wav", "Shower Running 02.wav", "Shower faucet.wav"],
}
MD5 = {"evaluation": "4569624ccadf96223f19cb59fe4f849f", "validation": "7dba730be08bada48bd15dc4e668df59", "development": "c8b05bc7acdb13895bb3c6a29608667e"}
for split in ("validation", "evaluation", "development"):
    arc = f"{X}/clotho_audio_{split}.7z"
    os.system(f'curl -sL -C - -o "{arc}" "https://zenodo.org/records/4783391/files/clotho_audio_{split}.7z?download=1"')
    h = hashlib.md5()
    with open(arc, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    assert h.hexdigest() == MD5[split], (split, h.hexdigest())
    with py7zr.SevenZipFile(arc) as z:
        names = z.getnames()
        tg = [n for n in names if os.path.basename(n) in WANT[split]]
        miss = set(WANT[split]) - {os.path.basename(n) for n in tg}
        print(split, "압축 안 파일", len(names), "· 고름", len(tg), "· 못 찾음", miss, flush=True)
        z.extract(path=X + "/audio/" + split, targets=tg)
    os.remove(arc)
print("[CLOTHO_DONE]")
