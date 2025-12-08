Template for a dual python/typescript library (called kotogram) that can be published to PyPi and npm respectively

** Example logic **
There is a simple component that serves as an example:

interface Codec
  public encode(string) : string
  public decode(string) : string
  # defined in Codec.py and Codec.ts respectively

class ReversingCodec : Codec
  implementation of codec. Both encode and decode reverse the input string.
  # implemented in ReversingCodec.py and ReversingCodec.ts respectively.

** Github workflows **
python_canary.yml -- build and test python code
typescript_canary.yml -- build and test typescript code
python_publish.yml -- publish python library to PyPi
typescript_publish.yml -- publish typescript to npm

** Publish flow **
./version.txt -- initially contains "0.0.1". The single source of truth for the current version.
./publish.sh -- bumps the version number, creates a git tag and pushes it. This triggers python_publish.yml and typescript_publish.yml actions.

** Inspiration **
These projects are temporarily available to provide inspiration
.tmp-inspiration/srsdb -- a python library that builds, tests, and publishes to PyPi.
.tmp-inspiration/kana-control -- a typescript library that builds, tests, and publishes to PyPi. This library is a Lit control, but the template we're creating isn't Lit, it's just a pure typescript library.
These '.tmp-inspiration' projects will be deleted once this template is functional, so we shouldn't modify or directly depend on them.
