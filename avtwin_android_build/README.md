# AV-Twin Android Responder

Temporary Android responder used to reproduce the AV-Twin two-way acoustic handshake.

## v0.2
- 48 kHz mono PCM16 recording
- C1: 11-19 kHz, 0.2 s
- C2: 300 Hz-9 kHz, 0.2 s
- AudioTrack uses MODE_STREAM for broader tablet compatibility, including Xiaomi Pad 7S Pro / HyperOS devices where MODE_STATIC may fail to initialize.
- Detect C1 -> record t2 -> play C2 -> self-detect t3 -> report t2/t3 to Linux via UDP.
