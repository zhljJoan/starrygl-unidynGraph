# ATC-StarryglLib

This is the slim migration target for the StarryUniGraph runtime work.

The retained scope is intentionally small:

- CTDG end-to-end path compatible with the existing MemShare flow.
- DTDG path compatible with the existing Flare/STGraphLoader flow.
- Pluggable task adapters.
- Native temporal sampling and negative sampling interfaces.
- STGraphLoader decay sampling hooks.
- Shared feature, mailbox, memory cache, model, and communication interfaces.

The old repository remains the source of truth while this package is being filled in. Legacy bridge modules are isolated under the CTDG and DTDG packages so they can be removed one by one.

## Layout

```text
src/atc_starrygl_lib/
  core/        session, registry, shared batch/artifact types
  ctdg/        MemShare-compatible CTDG path
  dtdg/        Flare/STGraphLoader-compatible DTDG path
  tasks/       task adapters
  sampling/    negative, native, boundary sampling
  features/    feature store and remote fetch contracts
  memory/      mailbox and hot/decay cache contracts
  comm/        communication abstraction
  models/      model contracts and shared heads
```

## Migration Rule

New code should depend on `core`, `tasks`, `sampling`, `features`, `memory`, `comm`, and `models`. CTDG and DTDG may wrap old StarryUniGraph code during migration, but legacy imports should stay at the edge.

## Native Build

The C++ BTS/MemShare sampler source is under `csrc/ctdg_sampler/bts_sampler`.

Build the native modules with:

```bash
bash scripts/build_native.sh
```

The build writes `libstarrygl_sampler.so` and `adaptive_split_cpp.so` into `src/atc_starrygl_lib/lib`, where `atc_starrygl_lib.lib.loader` imports them.
