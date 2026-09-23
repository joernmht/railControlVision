## Checklist

- [ ] `requirements.txt` changed -> `make lock` was run and the regenerated `requirements.lock` is committed.
- [ ] Schema models changed -> `make schema` was run and `schema/v0.json` plus `docs/schema.md` are updated.
- [ ] `make check` is green on Python 3.11 and 3.12.
- [ ] Any new stub raises `NotImplementedError` with the documented message (it is discovered by `tests/unit/test_stubs.py`).
- [ ] No fabricated results or metrics: nothing pretends to be implemented.
