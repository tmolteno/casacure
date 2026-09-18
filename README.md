# casacure

A rust replacement for casacore, designed to be pip installable on any machine

This will be a rust crate, with a python interface. The needed functionality is that sufficient to stop dask-ms depending on casacore which is a building nightmare on non-amd64 architectures.

Initially casacure will be a rust-crate that we keep updated with the casacore master repository (we will never modify this code). There will also be a python-casacure. Initially muliprocessing will be left out, but eventually rayon may be used to speed up an compute intensive tasks.

## TODO 

* Create  a rust crate framework called casacure.
* Check the file CASACORE_TO_CASA_RS.md for an outline of the needed functionality
* Create lots of unit-tests to guarantee compatabillity
* Set up a testing comparison framework against the system installed casacore library.
* Create github issues for each of the major functionality areas that need implementation, and track progress there.
* Create a master document called ARE_WE_CURED.md which tracks overall progress (passing tests, percentage coverage)
* Make it easy for others to contribute by creating a TODO.md document that keeps track of needed next steps.
* When performing steps, do them into small subtasks, and add them to TODO.md before starting.
* Remove each task from TODO.md when completed and add an entry to CHANGELOG.md
* Then push changes to github before doing the next TODO.md item.

## Linting

Rust work is linted with clippy and rustfmt as we go — CI runs these as gates
(`.github/workflows/ci.yml`), so run them locally before committing:

```sh
cargo fmt
cargo clippy --workspace --all-targets -- -D warnings
```

`clippy` is run with `-D warnings` (warnings are errors); `cargo fmt` keeps
the formatting canonical. Everything merged must pass both.


