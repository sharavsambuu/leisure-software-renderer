#!/usr/bin/env python3
"""t0_parity — Tier0 cross-backend equivalence gate (P1.3).

Compares a *_sw.png against its *_vk.png twin: exact-match percentage
(0.00% differing = gate PASS) plus a tolerance count (pixels within 1 LSB).
Also prints an ASCII difference map for localization (lessons doc S7:
difference maps beat percentages).

Stdlib only (zlib + struct): no PIL, runs anywhere.
Usage: t0_parity.py <sw.png> <vk.png> [--diffmap] [--tol 1]
Exit: 0 exact parity, 2 tolerance-only, 1 significant drift.
"""
import struct
import sys
import zlib


def read_png(path):
    data = open(path, "rb").read()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", f"not a PNG: {path}"
    pos, w, h, ctype, depth, idat = 8, None, None, None, None, b""
    while pos < len(data):
        (ln,) = struct.unpack(">I", data[pos:pos + 4])
        typ = data[pos + 4:pos + 8]
        chunk = data[pos + 8:pos + 8 + ln]
        if typ == b"IHDR":
            w, h, depth, ctype, comp, filt, inter = struct.unpack(">IIBBBBB", chunk)
            assert depth == 8 and inter == 0, f"unsupported PNG format: {path}"
            assert ctype in (2, 6), f"need RGB/RGBA8, got ctype={ctype}: {path}"
        elif typ == b"IDAT":
            idat += chunk
        elif typ == b"IEND":
            break
        pos += 12 + ln
    raw = zlib.decompress(idat)
    ch = 3 if ctype == 2 else 4
    stride = w * ch
    px = bytearray(w * h * ch)
    prev = bytearray(stride)
    p = 0
    for y in range(h):
        f = raw[p]
        p += 1
        line = bytearray(raw[p:p + stride])
        p += stride
        if f == 1:
            for i in range(ch, stride):
                line[i] = (line[i] + line[i - ch]) & 0xFF
        elif f == 2:
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif f == 3:
            for i in range(stride):
                a = line[i - ch] if i >= ch else 0
                line[i] = (line[i] + ((a + prev[i]) >> 1)) & 0xFF
        elif f == 4:
            for i in range(stride):
                a = line[i - ch] if i >= ch else 0
                b = prev[i]
                c = prev[i - ch] if i >= ch else 0
                q = a + b - c
                pa, pb, pc = abs(q - a), abs(q - b), abs(q - c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (line[i] + pr) & 0xFF
        elif f != 0:
            raise ValueError(f"bad filter {f}")
        px[y * stride:(y + 1) * stride] = line
        prev = line
    if ctype == 2:  # expand RGB -> RGBA opaque for uniform compare
        out = bytearray(w * h * 4)
        for i in range(w * h):
            out[4 * i:4 * i + 3] = px[3 * i:3 * i + 3]
            out[4 * i + 3] = 255
        px = out
    return w, h, px


def main():
    argv_ = sys.argv[1:]
    diffmap = "--diffmap" in argv_
    tol = 1
    args = []
    i = 0
    # Both documented forms must work: "--tol 1" and "--tol=1".
    while i < len(argv_):
        a = argv_[i]
        if a == "--tol" and i + 1 < len(argv_):
            tol = int(argv_[i + 1])
            i += 2
            continue
        if a.startswith("--tol="):
            tol = int(a.split("=", 1)[1])
            i += 1
            continue
        if a == "--diffmap":
            i += 1
            continue
        args.append(a)
        i += 1
    if len(args) != 2:
        print(__doc__)
        return 3
    sw_p, vk_p = args
    w1, h1, a = read_png(sw_p)
    w2, h2, b = read_png(vk_p)
    assert (w1, h1) == (w2, h2), f"size mismatch {w1}x{h1} vs {w2}x{h2}"
    n = w1 * h1
    exact = sum(1 for i in range(n) if a[4 * i:4 * i + 4] != b[4 * i:4 * i + 4])
    maxd = 0
    for i in range(n):
        for k in range(4):
            d = abs(a[4 * i + k] - b[4 * i + k])
            if d > maxd:
                maxd = d
    within = sum(
        1 for i in range(n)
        if a[4 * i:4 * i + 4] != b[4 * i:4 * i + 4]
        and all(abs(a[4 * i + k] - b[4 * i + k]) <= tol for k in range(4))
    )
    pct = 100.0 * exact / n
    print(f"parity {sw_p} vs {vk_p}: {w1}x{h1} n={n}")
    print(f"differ_exact={exact} ({pct:.2f}%) within_{tol}lsb={within} max_abs_drift={maxd}")
    if diffmap:
        cols, rows = 80, 24
        print("diffmap (.=same #=differ):")
        for r in range(rows):
            line = ""
            for c in range(cols):
                x0, y0 = c * w1 // cols, r * h1 // rows
                x1, y1 = (c + 1) * w1 // cols, (r + 1) * h1 // rows
                hit = False
                for yy in range(y0, y1):
                    for xx in range(x0, x1):
                        i = yy * w1 + xx
                        if a[4 * i:4 * i + 4] != b[4 * i:4 * i + 4]:
                            hit = True
                            break
                    if hit:
                        break
                line += "#" if hit else "."
            print(line)
    if exact == 0:
        print("GATE: PASS (exact)")
        return 0
    if exact == within:
        print("GATE: TOLERANCE-ONLY")
        return 2
    print("GATE: FAIL (significant drift)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
