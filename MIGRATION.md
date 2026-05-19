# Migration Notes

## Keep

- MemShare CTDG end-to-end training path.
- FlareDTDG STGraphLoader training path.
- Task adapters independent from graph mode.
- C++ native temporal sampler bindings.
- Random and future boundary-aware negative sampling.
- STGraphLoader decay sampling.
- Distributed feature fetch, mailbox, hot memory cache, model computation, and communication abstraction.

## Do Not Carry Forward

- Three parallel runtime hierarchies for `ctdg`, `dtdg`, and `chunk`.
- Placeholder pipeline adapters that return empty chunks.
- Node-task target materialization in the CTDG edge-prediction hot path.
- Multiple model registries and duplicated task heads.
- Cache/state classes that only contain `NotImplementedError` unless they define a stable interface.

## Current Bridge Strategy

The first pass keeps old implementations behind two bridge classes:

- `ctdg.runtime.backend.MemShareCTDGBackend`
- `dtdg.runtime.backend.FlareDTDGBackend`

Everything else should talk to the common `DataBackend` and `TaskAdapter` protocols.
