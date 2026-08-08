#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recompila arquivos WAV para o formato RXWS usado pelo karvoice.bin.

Uso:
    python recompilar_karvoice.py karvoice.bin pasta_dos_wavs -o karvoice_novo.bin

A pasta pode conter:
    bt001_010.wav
    bt001_020.wav
    ...

O programa usa os nomes FTXT existentes no BIN para fazer o vínculo.
Se não encontrar um WAV pelo nome, usa os WAVs em ordem alfabética.

Observação:
- O formato original deste arquivo usa Sony/PlayStation ADPCM (PS-ADPCM).
- A recompilação não precisa manter o mesmo tamanho comprimido do áudio.
- O programa recalcula BODY, tamanho RXWS e padding de 0x1000 bytes.
"""

import argparse
import math
import struct
import wave
from pathlib import Path

# Coeficientes PS-ADPCM correspondentes ao decoder usado na extração.
COEFS = (
    (0, 0),
    (60, 0),
    (115, -52),
    (98, -55),
    (122, -60),
)


def clamp16(x):
    return max(-32768, min(32767, int(x)))


def decode_predict(prev1, prev2, filt):
    c1, c2 = COEFS[filt]
    return (c1 * prev1 + c2 * prev2 + 32) >> 6


def encode_frame(samples, prev1, prev2):
    """Codifica 28 samples em um frame PS-ADPCM de 16 bytes."""
    original_len = len(samples)
    if original_len < 28:
        samples = samples + [0] * (28 - original_len)

    best = None

    # Testa todos os filtros e shifts possíveis.
    # A métrica é o erro quadrático do áudio reconstruído.
    for filt in range(5):
        for shift in range(13):
            p1, p2 = prev1, prev2
            nibbles = []
            error = 0
            ok = True

            for s in samples:
                pred = decode_predict(p1, p2, filt)

                # Inverte aproximadamente:
                # decoded = (nibble << 12) >> shift + predictor
                scale = 1 << (12 - shift)
                q = int(round((s - pred) / scale))

                if q < -8:
                    q = -8
                elif q > 7:
                    q = 7

                recon = clamp16(pred + (q << 12 >> shift))
                error += (s - recon) * (s - recon)
                nibbles.append(q)

                p2, p1 = p1, recon

            if best is None or error < best[0]:
                best = (error, filt, shift, nibbles)

    _, filt, shift, nibbles = best

    # Reconstrói novamente para obter o estado do predictor
    # usado no próximo frame.
    p1, p2 = prev1, prev2
    for q in nibbles:
        recon = clamp16(decode_predict(p1, p2, filt) +
                        ((q << 12) >> shift))
        p2, p1 = p1, recon

    header = ((filt & 0x0F) << 4) | (shift & 0x0F)
    out = bytearray(16)
    out[0] = header
    out[1] = 0
    for i in range(14):
        lo = nibbles[i * 2] & 0x0F
        hi = nibbles[i * 2 + 1] & 0x0F
        out[2 + i] = lo | (hi << 4)

    return bytes(out), p1, p2


def encode_psx_adpcm(samples):
    """Converte PCM16 mono para PS-ADPCM."""
    out = bytearray()
    prev1 = prev2 = 0

    for pos in range(0, len(samples), 28):
        frame, prev1, prev2 = encode_frame(
            samples[pos:pos + 28], prev1, prev2
        )
        out.extend(frame)

    # Áudio vazio: um frame silencioso.
    if not out:
        frame, _, _ = encode_frame([0] * 28, 0, 0)
        out.extend(frame)

    return bytes(out)


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        channels = w.getnchannels()
        sample_width = w.getsampwidth()
        rate = w.getframerate()
        frames = w.getnframes()
        raw = w.readframes(frames)

    if channels != 1:
        raise ValueError(
            f"{path.name}: o formato original é mono; "
            f"este WAV tem {channels} canais."
        )
    if sample_width != 2:
        raise ValueError(
            f"{path.name}: o WAV precisa ser PCM 16-bit."
        )

    samples = list(struct.unpack("<%dh" % frames, raw))
    return rate, samples


def find_rxws(blob):
    positions = []
    p = 0

    while True:
        p = blob.find(b"RXWS", p)
        if p < 0:
            break

        if p + 0x10 <= len(blob):
            declared = struct.unpack_from("<I", blob, p + 4)[0]
            # O tamanho declarado exclui os 16 bytes iniciais do RXWS.
            size = declared + 0x10
            if size >= 0x80 and p + size <= len(blob):
                positions.append(p)

        p += 4

    return positions


def get_stream_name(blob, base):
    form_size = struct.unpack_from("<I", blob, base + 0x14)[0]
    ftxt = base + 0x20 + form_size

    if blob[ftxt:ftxt + 4] != b"FTXT":
        return None

    chunk_data = ftxt + 0x10
    total = struct.unpack_from("<i", blob, chunk_data)[0]

    if total < 1:
        return None

    rel = struct.unpack_from("<I", blob, chunk_data + 4)[0]
    name_pos = chunk_data + rel

    if name_pos < 0 or name_pos >= len(blob):
        return None

    end = blob.find(b"\0", name_pos, name_pos + 1024)
    if end < 0:
        end = name_pos + 1024

    return blob[name_pos:end].decode("utf-8", "replace")


def natural_key(path):
    import re
    return [
        int(x) if x.isdigit() else x.lower()
        for x in re.split(r"(\d+)", path.name)
    ]


def build_block(original, base, wav_path):
    # Este formato do arquivo possui exatamente 1 stream por RXWS.
    total = struct.unpack_from("<i", original, base + 0x20)[0]
    if total != 1:
        raise RuntimeError(
            f"RXWS em 0x{base:X} possui {total} streams; "
            "este recompilador espera 1 stream por bloco."
        )

    # Tipo original: 0 = PS-ADPCM.
    stream_header = base + 0x24
    typ = original[stream_header]
    channels = original[stream_header + 9]
    original_rate = struct.unpack_from("<H", original, stream_header + 10)[0]

    if typ != 0:
        raise RuntimeError(
            f"RXWS em 0x{base:X}: tipo de áudio 0x{typ:02X} não é PS-ADPCM."
        )
    if channels != 1:
        raise RuntimeError(
            f"RXWS em 0x{base:X}: canais={channels}; esperado mono."
        )

    rate, samples = read_wav(wav_path)

    if rate != original_rate:
        raise ValueError(
            f"{wav_path.name}: taxa {rate} Hz, mas o BIN espera "
            f"{original_rate} Hz."
        )

    encoded = encode_psx_adpcm(samples)

    # BODY começa em base+0x70 neste formato.
    body_header = base + 0x70
    if original[body_header:body_header + 4] != b"BODY":
        raise RuntimeError(
            f"BODY não encontrado no RXWS em 0x{base:X}."
        )

    body_prefix = bytearray(original[base:body_header + 0x10])

    # Atualiza tamanho do BODY.
    struct.pack_into("<I", body_prefix, body_header - base + 4, len(encoded))

    # Atualiza offset e byte_count do stream.
    rel_header = stream_header - base
    struct.pack_into("<I", body_prefix, rel_header + 16, 0)
    struct.pack_into("<i", body_prefix, rel_header + 20, len(encoded))

    new_block = bytearray(body_prefix)
    new_block.extend(encoded)

    # O RXWS usa tamanho sem os 16 bytes do cabeçalho RXWS.
    struct.pack_into("<I", new_block, 4, len(new_block) - 0x10)

    return bytes(new_block)


def main():
    parser = argparse.ArgumentParser(
        description="Recompila WAVs para um arquivo RXWS/.BIN."
    )
    parser.add_argument(
        "original",
        help="BIN original usado como modelo"
    )
    parser.add_argument(
        "wav_dir",
        help="pasta contendo os WAVs"
    )
    parser.add_argument(
        "-o", "--output",
        default="karvoice_recompilado.bin",
        help="arquivo BIN de saída"
    )
    args = parser.parse_args()

    original_path = Path(args.original)
    wav_dir = Path(args.wav_dir)
    output_path = Path(args.output)

    if not original_path.is_file():
        raise SystemExit(f"Arquivo não encontrado: {original_path}")
    if not wav_dir.is_dir():
        raise SystemExit(f"Pasta não encontrada: {wav_dir}")

    original = original_path.read_bytes()
    positions = find_rxws(original)

    if not positions:
        raise SystemExit("Nenhum RXWS encontrado no arquivo original.")

    wavs = sorted(
        [x for x in wav_dir.iterdir()
         if x.is_file() and x.suffix.lower() == ".wav"],
        key=natural_key
    )

    if len(wavs) < len(positions):
        raise SystemExit(
            f"Faltam WAVs: o BIN possui {len(positions)} áudios, "
            f"mas a pasta possui {len(wavs)}."
        )

    # Mapeia primeiro pelos nomes FTXT.
    by_stem = {x.stem.lower(): x for x in wavs}
    used = set()
    mapping = []

    for index, base in enumerate(positions):
        name = get_stream_name(original, base)
        candidate = by_stem.get(name.lower()) if name else None

        if candidate is None:
            # Fallback: usa o WAV na mesma posição.
            candidate = wavs[index]

        mapping.append((base, candidate, name))
        used.add(candidate)

    result = bytearray()
    cursor = 0
    changed = 0

    for index, (base, wav_path, name) in enumerate(mapping, 1):
        # Copia tudo entre o bloco anterior e este.
        result.extend(original[cursor:base])

        old_declared = struct.unpack_from("<I", original, base + 4)[0]
        old_block_size = old_declared + 0x10

        new_block = build_block(original, base, wav_path)
        result.extend(new_block)

        print(
            f"[{index:03d}/{len(mapping):03d}] "
            f"{name or wav_path.stem}: "
            f"{len(original[base:base + old_block_size])} -> "
            f"{len(new_block)} bytes"
        )

        # RXWSs originais são alinhados em 0x1000.
        new_padded_end = (base + len(new_block) + 0xFFF) & ~0xFFF
        result.extend(b"\0" * (new_padded_end - (base + len(new_block))))

        cursor = base + old_block_size
        changed += 1

    # Copia qualquer dado restante após o último RXWS.
    result.extend(original[cursor:])

    output_path.write_bytes(result)

    print()
    print(f"Concluído: {changed} áudio(s) recompilado(s).")
    print(f"Saída: {output_path}")
    print(f"Tamanho: {len(result):,} bytes")


if __name__ == "__main__":
    main()
