#!/usr/bin/env python3
import argparse, re, struct, wave
from pathlib import Path

COEF = [(0, 0), (60, 0), (115, -52), (98, -55), (122, -60)]

def decode_psx_adpcm(data):
    pcm, prev1, prev2 = [], 0, 0
    for pos in range(0, len(data) - 15, 16):
        frame = data[pos:pos+16]
        shift = frame[0] & 0x0F
        filt = min((frame[0] >> 4) & 0x0F, 4)
        c1, c2 = COEF[filt]
        for b in frame[2:16]:
            for n in (b & 0x0F, b >> 4):
                n = n if n < 8 else n - 16
                sample = (n << 12) >> shift
                sample += (c1 * prev1 + c2 * prev2 + 32) >> 6
                sample = max(-32768, min(32767, sample))
                pcm.append(sample)
                prev2, prev1 = prev1, sample
    return pcm

def safe_name(name, index):
    name = name or f"audio_{index:03d}"
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    return name or f"audio_{index:03d}"

def get_name(blob, base, form_size, stream_index, total):
    ftxt = base + 0x20 + form_size
    if blob[ftxt:ftxt+4] != b"FTXT":
        return None
    chunk = ftxt + 0x10
    if struct.unpack_from("<i", blob, chunk)[0] != total:
        return None
    rel = struct.unpack_from("<I", blob, chunk + 4 + stream_index*4)[0]
    no = chunk + rel
    end = blob.find(b"\0", no, no + 1024)
    return blob[no:end if end >= 0 else no+1024].decode("utf-8", "replace")

def extract(input_file, output_dir):
    blob = Path(input_file).read_bytes()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    positions, p = [], 0
    while True:
        p = blob.find(b"RXWS", p)
        if p < 0: break
        if p + 0x10 <= len(blob):
            size = struct.unpack_from("<I", blob, p+4)[0] + 0x10
            if size >= 0x30 and p + size <= len(blob):
                positions.append(p)
        p += 4

    count = 0
    for base in positions:
        total = struct.unpack_from("<i", blob, base+0x20)[0]
        form_size = struct.unpack_from("<I", blob, base+0x14)[0]

        # Find BODY chunk.
        p = base + 0x10
        while p + 8 <= len(blob):
            cid = blob[p:p+4]
            csize = struct.unpack_from("<I", blob, p+4)[0]
            if cid == b"BODY":
                body = p + 0x10
                break
            p += 0x10 + csize
        else:
            raise RuntimeError("BODY não encontrado")

        for si in range(total):
            ho = base + 0x24 + 0x1C*si
            typ = blob[ho]
            channels = blob[ho+9]
            rate = struct.unpack_from("<H", blob, ho+10)[0]
            off = struct.unpack_from("<I", blob, ho+16)[0]
            size = struct.unpack_from("<i", blob, ho+20)[0]
            raw = blob[body+off:body+off+size]

            if typ == 0:
                if channels != 1:
                    raise RuntimeError("PS-ADPCM multicanal não implementado neste script")
                samples = decode_psx_adpcm(raw)
                frames = struct.pack("<%dh" % len(samples), *samples)
            elif typ == 1:
                frames = raw
            else:
                raise RuntimeError(f"Tipo RXWS 0x{typ:02X} não suportado")

            name = safe_name(get_name(blob, base, form_size, si, total), count+1)
            out = output_dir / f"{count+1:03d}_{name}.wav"
            with wave.open(str(out), "wb") as w:
                w.setnchannels(1 if typ == 0 else channels)
                w.setsampwidth(2)
                w.setframerate(rate)
                w.writeframes(frames)
            count += 1

    print(f"Extraídos {count} áudio(s) para: {output_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Extrai áudios de arquivos RXWS/.XWS/.BIN da Sony.")
    ap.add_argument("arquivo", help="arquivo RXWS/.bin")
    ap.add_argument("-o", "--saida", default="audio_extraido", help="pasta de saída")
    args = ap.parse_args()
    extract(args.arquivo, args.saida)
