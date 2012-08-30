# Bit Compressed Vector

This project provides a simple implementation of a bit compressed vector. This
means, that in contrast ot a bit vector which only captures a series of 0s and
1s this library provides a vector that can store arbitrary numbers. However,
the width of value has to be specified.

The goal is to achieve a good compression ratio by keeping up with the
sequential scan speed of a std::vector.

This work is largely based on the algorithms presented by Wilhalm et al "SIMD-
Scan: Ultra Fast in-Memory Table Scan using on-Chip Vector Processing Units"
published in PVLDB 2(1): 385-394 (2009).

![Bit Dependency](grundprinzip.github.com/bitcompressedvector/images/initial.png)


## Usage

The BCV is intended to be a drop-in replacement of ``std::vector``, however,
currently  it is only of fixed size and does not support any kind of iterator
interface.  The access methods to the vector are:

  1. Index-based subscript
  1. Index-based ``get``/``set()``
  1. Multi-get based

The index-based access allows array subscript operator access, however this is
proxy access around the get() / set() methods. Especially the ``[]`` as lvalue
might be more expensive than a simple ``set()``.

The multi-get method allows to extract multiple values at once. Here we
differentiate between to versions of the ``mget()`` the first amget will
extract one cache line of compressed values and write them out to a external
storage array. The second version ``mget_fixed()`` will only extract one cache
line of uncompressed values and write them to the external storage. It is
important to mention that ``mget_fixed()`` will not perform any range checks
on the data, so make sure you extract the right amount of data.

## Adding to your Project

To increase the performance of the bit-compressed vector some parts of the bit
mask lookups are generated so you have to run

	make release

before continuing. Now you can copy everything from pkg/bcv to your project
and use it as is.


## Performance Numbers

Currently the performance of the vector is comparable to the ``std::vector<T>``
for sequential scans but allowing to save a significant amount of memory

For a vector with 100M elements the sequential scan speed is on a Intel Xeon
7560 and 5 bits stored for 32 bit integers a scan aggregating all values takes:

  * get time ``0.268859``s
  * mget time ``0.095577``s
  * vector time ``0.133813``s

The memory consumption for the vector is ~ 400MB and for the bit compressed
vector ~ 60MB.



## Licence 

Copyright (c) 2012, Martin Grund

All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

  1. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
  2. All advertising materials mentioning features or use of this software must display the following acknowledgement: “This product includes software developed by the University of California, Berkeley and its contributors.”
  3. Neither the name of the University nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
