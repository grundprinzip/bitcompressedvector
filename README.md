h1. Bit Compressed Vector

This project provides a simple implementation of a bit compressed vector. This means, that in contrast ot a bit vector which only captures a series of 0s and 1s this library provides a vector that can store arbitrary numbers. However, the width of value has to be specified.

The goal is to achieve a good compression ratio by keeping up with the sequential scan speed of a std::vector.
