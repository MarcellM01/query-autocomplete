# Third-Party Licenses

`query-autocomplete` is licensed under the MIT License.

Some dependencies may be licensed under different terms.

## `marisa-trie`

- Project: `marisa-trie`
- PyPI: https://pypi.org/project/marisa-trie/
- Source: https://github.com/pytries/marisa-trie

At the time of writing, PyPI lists this SPDX license expression for `marisa-trie`:

`MIT AND (BSD-2-Clause OR LGPL-2.1-or-later)`

The PyPI description states that:
- the Python wrapper is licensed under MIT
- the bundled `marisa-trie` C++ library is dual-licensed under BSD-2-Clause or LGPL-2.1-or-later

`query-autocomplete` uses `marisa-trie` as an ordinary Python dependency and does not modify or vendor its source code. For the bundled C++ core, the BSD-2-Clause option is the simpler permissive path for an MIT-licensed project.

### Python wrapper notice

```text
Copyright (c) marisa-trie authors and contributors, 2012-2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished
to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR
A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

### Bundled MARISA C++ core notice

The upstream MARISA C++ core is dual-licensed under BSD-2-Clause OR LGPL-2.1-or-later. This project relies on the BSD-2-Clause option.

```text
Copyright (c) 2010-2025, Susumu Yata
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

This summary is provided for convenience and does not constitute legal advice. Please review the upstream project directly if you need a formal license assessment.
