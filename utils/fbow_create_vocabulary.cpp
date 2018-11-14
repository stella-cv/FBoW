/**

The MIT License

Copyright (c) 2017 Rafael Muñoz-Salinas

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

*/

#include "cmd_line_parser.h"
#include "vocabulary_creator.h"

#include <iostream>
#include <fstream>
#include <vector>

#include <opencv2/core/core.hpp>

std::vector<cv::Mat> readFeaturesFromFile(const std::string& filename, std::string& desc_name) {
    std::vector<cv::Mat> features;
    std::ifstream ifile(filename, std::ios::binary);
    if (!ifile.is_open()) {
        std::cerr << "could not open the input file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    char _desc_name[20];
    ifile.read(_desc_name, 20);
    desc_name = _desc_name;

    uint32_t size;
    ifile.read((char*) &size, sizeof(size));
    features.resize(size);
    for (size_t i = 0; i < size; i++) {
        uint32_t cols, rows, type;
        ifile.read((char*)&cols, sizeof(cols));
        ifile.read((char*)&rows, sizeof(rows));
        ifile.read((char*)&type, sizeof(type));
        features[i].create(rows, cols, type);
        ifile.read((char*)features[i].ptr<uchar>(0), features[i].total() * features[i].elemSize());
    }
    return features;
}

int main(int argc, char** argv) {
    try {
        CmdLineParser cml(argc, argv);
        if (cml["-h"] || argc < 3) {
            std::cerr << "Usage: FEATURE_INPUT OUTPUT_VOCABULARY [-k K] [-l L] [-t NUM_THREADS] [--max-iters NUM_ITER] [-v]" << std::endl;
            std::cerr << std::endl;
            std::cerr << "Second step is creating the vocabulary of K^L from the set of features." << std::endl;
            std::cerr << "By default, we employ a random selection center without running a single iteration of the k means." << std::endl;
            std::cerr << "As indicated by the authors of the FLANN library in their paper, the result is not very different from using k-means, but speed is much better." << std::endl;
            std::cerr << std::endl;
            return EXIT_FAILURE;
        }

        std::string desc_name;
        auto features = readFeaturesFromFile(argv[1], desc_name);

        std::cout << "descriptor name: " << desc_name << std::endl;

        fbow::VocabularyCreator::Params params;
        params.k = stoi(cml("-k", "10"));
        params.L = stoi(cml("-l", "6"));
        params.nthreads = stoi(cml("-t", "4"));
        params.maxIters = std::stoi(cml("--max-iters", "0"));
        params.verbose = cml["-v"];

        srand(0);
        fbow::VocabularyCreator vocab_creator;
        fbow::Vocabulary vocab;

        std::cout << "creating a " << params.k << "^" << params.L << " vocabulary ..." << std::endl;

        auto t_start = std::chrono::high_resolution_clock::now();
        vocab_creator.create(vocab, features, desc_name, params);
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "time: " << (double)(std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count()) << "ms" << std::endl;
        std::cout << "number of blocks: " << vocab.size() << std::endl;
        std::cout << "saving the vocabulary: " << argv[2] << std::endl;

        vocab.saveToFile(argv[2]);
    } catch (std::exception& ex) {
        std::cerr << ex.what() << std::endl;
    }

    return EXIT_SUCCESS;
}
