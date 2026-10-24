"""FSD50K 분할 zip 에서 필요한 wav 만 HTTP Range 로 꺼낸다 (전체 30GB 를 받지 않기 위해).
사용: python fsd_range.py <ids.txt: split,id 줄> <out_dir>
"""
import sys, os, struct, zlib, json, urllib.request, time

BASE = "https://zenodo.org/records/4060432/files/{}?download=1"
PART = 3221225472
NPART = {"dev": 5, "eval": 1}          # .z01.. 개수, 마지막 디스크는 .zip
LAST_SIZE = {"dev": 2306663327, "eval": 3037675767}


def get(name, a, b, tries=10):
    for t in range(tries):
        try:
            req = urllib.request.Request(BASE.format(name), headers={"Range": f"bytes={a}-{b}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                assert r.status == 206, r.status
                d = r.read()
            assert len(d) == b - a + 1, (len(d), b - a + 1)
            return d
        except Exception as e:
            print("  retry", name, a, e, flush=True); time.sleep(min(60, 5 * (t + 1)))
    raise RuntimeError(name)


def part_name(split, disk):
    return f"FSD50K.{split}_audio." + ("zip" if disk == NPART[split] else f"z{disk + 1:02d}")


def read_global(split, g, n):
    """디스크들을 이어 붙인 가상 주소 g 부터 n 바이트."""
    out = b""
    while n > 0:
        disk, off = divmod(g, PART)
        size = LAST_SIZE[split] if disk == NPART[split] else PART
        k = min(n, size - off)
        out += get(part_name(split, disk), off, off + k - 1)
        g += k; n -= k
    return out


def central_dir(split):
    name, size = part_name(split, NPART[split]), LAST_SIZE[split]
    tail = get(name, size - 65536 - 22, size - 1)
    i = tail.rfind(b"PK\x05\x06")
    eocd = tail[i:]
    cd_disk = struct.unpack("<H", eocd[6:8])[0]
    n_tot, cd_size, cd_off = struct.unpack("<HII", eocd[10:20])
    j = tail.rfind(b"PK\x06\x07")                      # zip64 locator
    if j >= 0:
        z64_disk, z64_off = struct.unpack("<IQ", tail[j + 4:j + 16])
        z = read_global(split, z64_disk * PART + z64_off, 56)
        assert z[:4] == b"PK\x06\x06"
        cd_disk = struct.unpack("<I", z[20:24])[0]
        n_tot, cd_size, cd_off = struct.unpack("<QQQ", z[32:56])
    cd = read_global(split, cd_disk * PART + cd_off, cd_size)
    ents, p = {}, 0
    while p < len(cd):
        assert cd[p:p + 4] == b"PK\x01\x02"
        meth, crc, csz, usz, nl, el, cl, dsk = struct.unpack("<H4xIIIHHHH", cd[p + 10:p + 36])
        loff = struct.unpack("<I", cd[p + 42:p + 46])[0]
        fn = cd[p + 46:p + 46 + nl].decode()
        ex = cd[p + 46 + nl:p + 46 + nl + el]
        q = 0
        while q < len(ex):
            hid, hl = struct.unpack("<HH", ex[q:q + 4])
            if hid == 1:
                vals = ex[q + 4:q + 4 + hl]; r = 0
                if usz == 0xFFFFFFFF: usz = struct.unpack("<Q", vals[r:r + 8])[0]; r += 8
                if csz == 0xFFFFFFFF: csz = struct.unpack("<Q", vals[r:r + 8])[0]; r += 8
                if loff == 0xFFFFFFFF: loff = struct.unpack("<Q", vals[r:r + 8])[0]; r += 8
                if dsk == 0xFFFF: dsk = struct.unpack("<I", vals[r:r + 4])[0]
            q += 4 + hl
        ents[os.path.basename(fn)] = (meth, crc, csz, usz, dsk, loff)
        p += 46 + nl + el + cl
    assert len(ents) >= n_tot - 2, (len(ents), n_tot)
    return ents


def main():
    want = [l.strip().split(",") for l in open(sys.argv[1]) if l.strip()]
    out = sys.argv[2]; os.makedirs(out, exist_ok=True)
    cds = {}
    ok = bad = 0
    for split, fid in want:
        dst = os.path.join(out, f"{fid}.wav")
        if os.path.exists(dst):
            ok += 1; continue
        if split not in cds:
            cds[split] = central_dir(split); print(f"[CD] {split} {len(cds[split])} 항목", flush=True)
        e = cds[split].get(f"{fid}.wav")
        if e is None:
            print("  없음", split, fid); bad += 1; continue
        meth, crc, csz, usz, dsk, loff = e
        g = dsk * PART + loff
        try:   # [10-23] 한 파일이 실패해도 멈추지 않고 다음 파일로 넘어간다(실패 수는 끝에 보고, 종료 코드 1)
            h = read_global(split, g, 30)
            nl, el = struct.unpack("<HH", h[26:30])
            data = read_global(split, g + 30 + nl + el, csz)
            raw = data if meth == 0 else zlib.decompress(data, -15)
            assert len(raw) == usz and (zlib.crc32(raw) & 0xFFFFFFFF) == crc, fid
        except Exception as ex:
            print("  실패", split, fid, repr(ex)[:120], flush=True); bad += 1; continue
        open(dst + ".part", "wb").write(raw); os.replace(dst + ".part", dst)
        ok += 1
        if ok % 25 == 0: print(f"  {ok}/{len(want)}", flush=True)
    print(f"[완료] 받음 {ok} · 실패 {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
