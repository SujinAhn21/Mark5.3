"""DCASE2018 Task5 eval(Zenodo 1964758) 의 일반 zip 23개에서 필요한 wav 만 Range 요청으로 꺼낸다.

  python3 dcase_t5_range.py index <out_json>               # 23개 zip 의 중앙 목록 → 파일 위치표
  python3 dcase_t5_range.py fetch <index_json> <list_txt> <out_dir>   # 목록의 파일만 받기(받은 파일은 건너뜀)
"""
import json, os, struct, sys, time, urllib.request, zlib

BASE = "https://zenodo.org/records/1964758/files/DCASE2018-task5-eval.audio.{}.zip?download=1"
N = 23


def get(url, start, end, tries=10):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            data = urllib.request.urlopen(req, timeout=120).read()
            if len(data) == end - start + 1:
                return data
        except Exception as e:  # 끊김은 다시 시도
            err = e
        time.sleep(min(60, 5 * (k + 1)))
    raise RuntimeError(f"Range 실패 {url} {start}-{end}")


def size_of(url):
    req = urllib.request.Request(url, method="HEAD")
    return int(urllib.request.urlopen(req, timeout=60).headers["Content-Length"])


def central_dir(i):
    url = BASE.format(i)
    total = size_of(url)
    tail = get(url, total - 65557, total - 1)
    p = tail.rfind(b"PK\x05\x06")
    cd_size, cd_off = struct.unpack("<II", tail[p + 12:p + 20])
    if cd_off == 0xFFFFFFFF:  # zip64
        q = tail.rfind(b"PK\x06\x06")
        cd_size, cd_off = struct.unpack("<QQ", tail[q + 40:q + 56])
    cd = get(url, cd_off, cd_off + cd_size - 1)
    out, j = {}, 0
    while j < len(cd) and cd[j:j + 4] == b"PK\x01\x02":
        meth, = struct.unpack("<H", cd[j + 10:j + 12])
        crc, csz, usz = struct.unpack("<III", cd[j + 16:j + 28])
        nl, el, cl = struct.unpack("<HHH", cd[j + 28:j + 34])
        loff, = struct.unpack("<I", cd[j + 42:j + 46])
        name = cd[j + 46:j + 46 + nl].decode()
        extra = cd[j + 46 + nl:j + 46 + nl + el]
        if 0xFFFFFFFF in (csz, usz, loff):  # zip64 extra
            k = 0
            while k < len(extra):
                hid, hl = struct.unpack("<HH", extra[k:k + 4])
                if hid == 1:
                    vals = list(struct.unpack("<" + "Q" * (hl // 8), extra[k + 4:k + 4 + hl]))
                    if usz == 0xFFFFFFFF: usz = vals.pop(0)
                    if csz == 0xFFFFFFFF: csz = vals.pop(0)
                    if loff == 0xFFFFFFFF: loff = vals.pop(0)
                k += 4 + hl
        out[name] = [i, meth, crc, csz, usz, loff]
        j += 46 + nl + el + cl
    return out


def main():
    if sys.argv[1] == "index":
        idx = {}
        for i in range(1, N + 1):
            cd = central_dir(i)
            idx.update(cd)
            print(f"zip {i}: {len(cd)}개", flush=True)
        json.dump(idx, open(sys.argv[2], "w"))
        print(f"[완료] {len(idx)}개")
    elif sys.argv[1] == "fetch":
        idx = json.load(open(sys.argv[2]))
        names = [l.strip() for l in open(sys.argv[3]) if l.strip()]
        out = sys.argv[4]; os.makedirs(out, exist_ok=True)
        fail = 0
        for n_, name in enumerate(names, 1):
            dst = os.path.join(out, os.path.basename(name))
            if os.path.isfile(dst):
                continue
            try:
                i, meth, crc, csz, usz, loff = idx[name]
                url = BASE.format(i)
                h = get(url, loff, loff + 29)
                nl, el = struct.unpack("<HH", h[26:30])
                data = get(url, loff + 30 + nl + el, loff + 30 + nl + el + csz - 1)
                raw = data if meth == 0 else zlib.decompress(data, -15)
                assert len(raw) == usz and (zlib.crc32(raw) & 0xFFFFFFFF) == crc, "CRC 불일치"
                open(dst, "wb").write(raw)
            except Exception as e:
                fail += 1
                print(f"[실패] {name}: {e}", flush=True)
            if n_ % 50 == 0:
                print(f"{n_}/{len(names)}", flush=True)
        print(f"[끝] 실패 {fail}개")
        sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
