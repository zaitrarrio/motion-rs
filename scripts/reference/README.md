# Reference fixtures

Each `*_fixtures.py` runs the upstream PyTorch code (clones of
nv-tlabs/kimodo and NVlabs/motionbricks, `torch` CPU) on fixed inputs and
writes safetensors fixtures under `crates/<crate>/tests/fixtures/`. The tiny
random-weight fixtures are committed and run in CI; the released-weight
fixtures need the Hugging Face checkpoints (`MOTION_KIMODO_DIR`,
`MOTION_BRICKS_DIR`) and back the `#[ignore]`d tests.

Run them with `python3 -I` from the repository root; each script documents
its arguments at the top.
