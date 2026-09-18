# casacure

A rust replacement for casacore, designed to be pip installable on any machine

This will be a rust crate, with a python interface. The needed functionality is that sufficient to stop dask-ms depending on casacore which is a building nightmare on non-amd64 architectures.

Initially casacore will be a submodule, that we keep updated with the casacore master repository (we will never modify this code)

## TODO 

* Create  a rust crate framework called casacure.
* Check the file CASACORE_TO_CASA_RS.md for an outline of the needed functionality
* Create lots of unit-tests to guarantee compatabillity
* Set up a testing comparison framework against the system installed casacore library.
* Create github issues for each of the major functionality issues, and track progress there.
* Create a master document called ARE_WE_CURED.md which tracks overall progress (passing tests, percentage coverage)
* Make it easy for others to contribute by creating a TODO.md document that keeps track of needed next steps.

