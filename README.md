# Fast Bag of Words

**This is a modified version of [the original Fast Bag of Words](https://github.com/rmsalinas/fbow) by [@rmsalinas](https://github.com/rmsalinas).**

FBoW (Fast Bag of Words) is an extremely optimized version of the [DBoW2](https://github.com/dorian3d/DBoW2)/[DBoW3](https://github.com/rmsalinas/DBow3) libraries.  
The library is highly optimized to speed up the Bag of Words creation using AVX,SSE and MMX instructions.  
In loading a vocabulary, FBoW is about 80x faster than DBoW2.  
In transforming an image into Bag of Words using on machines with AVX instructions, it is about 6.4x faster.  

## Build

```bash
$ git clone https://github.com/stella-cv/FBoW.git
$ cd FBoW && mkdir build && cd build
$ cmake .. -DBUILD_TESTS=ON -DBUILD_UTILS=ON
$ make
```

## License

This software is distributed under MIT License.  
See the [LICENSE](./LICENSE).
