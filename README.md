# Generate complete light trigger events for Module 0

This is a script for creating an event-built light hdf5 file that can be associated with charge data.

Takes as input the timestamp corrected ROOT format created by [https://github.com/peter-madigan/ADCViewer64-Module0].

## Install

Install via `pip`::

	pip install -e .

## Run or get help

After installing, the script should exist in your path, so

      light_event_builder.py --help