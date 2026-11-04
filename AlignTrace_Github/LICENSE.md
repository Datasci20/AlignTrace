# MIT License

Copyright (c) 2026 Yaoyu Yang, Minghao Mou, Dongzhuyuan Lu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Scope

The MIT license above applies to AlignTrace's original source code,
documentation, and included pretrained model weights. For this grant, those
model weights are included in the term "Software". Third-party components
retain their respective licenses, as described below. The companion real and
synthetic datasets are separate distributions; this license does not assign
or change their reuse terms.

## Third-party notices

The `partition_into` and `insertion_into` routines in `aligntrace/beam.py`
were adapted from Numba's quicksort implementation.

Copyright (c) 2012, Anaconda, Inc. All rights reserved.

These routines retain their BSD two-clause license. Its complete copyright
notice, conditions, and disclaimer are reproduced in
[`licenses/NUMBA_LICENSE.txt`](licenses/NUMBA_LICENSE.txt).
Numba's accompanying third-party notices are preserved in
[`licenses/NUMBA_THIRD_PARTY_LICENSES.txt`](licenses/NUMBA_THIRD_PARTY_LICENSES.txt).
These third-party terms are not replaced by AlignTrace's MIT license.

NumPy, Numba, PyTorch, and RapidFuzz are external dependencies, installed
separately and governed by their respective licenses. Their package binaries
and Python environments are not redistributed in this repository.
