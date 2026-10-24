"""BeDAViN 분할 zip(raw_file.z01~z03 + .zip)에서 shower/toilet 원본만 Range 로 꺼낸다."""
import sys, os, struct, zlib, csv, io, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fsd_range as F
F.BASE = "https://zenodo.org/records/14486235/files/{}?download=1"
F.PART = 2147483648
F.NPART = {"raw": 3}
F.LAST_SIZE = {"raw": 310243864}
F.part_name = lambda split, disk: "raw_file." + ("zip" if disk == 3 else f"z{disk + 1:02d}")
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
meta = urllib.request.urlopen(F.BASE.format("metadata.csv")).read().decode()
want = {r["audio_id"]: r["category"] for r in csv.DictReader(io.StringIO(meta)) if r["category"] in ("shower", "toilet")}
print(want)
cd = F.central_dir("raw")
names = [n for n in cd if os.path.splitext(n)[0] in want]
print(len(cd), "항목 · 대상", names)
for n in names:
    meth, crc, csz, usz, dsk, loff = cd[n]
    g = dsk * F.PART + loff
    h = F.read_global("raw", g, 30); nl, el = struct.unpack("<HH", h[26:30])
    data = F.read_global("raw", g + 30 + nl + el, csz)
    raw = data if meth == 0 else zlib.decompress(data, -15)
    assert len(raw) == usz and (zlib.crc32(raw) & 0xFFFFFFFF) == crc
    open(os.path.join(out, f"{want[os.path.splitext(n)[0]]}__{n}"), "wb").write(raw)
print("[완료]")
