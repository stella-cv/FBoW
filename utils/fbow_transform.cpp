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

#include "fbow.h"
#include "cmd_line_parser.h"

#include <iostream>
#include <chrono>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#ifdef USE_CONTRIB
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#endif

cv::Mat loadFeatures(const std::string& path_to_image, std::string descriptor = "orb") {
    cv::Ptr<cv::Feature2D> feat_detector;
    if (descriptor == "orb") feat_detector = cv::ORB::create(2000);
    else if (descriptor == "brisk") feat_detector = cv::BRISK::create();
    else if (descriptor == "akaze") feat_detector = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, 1e-4);
#ifdef USE_CONTRIB
    else if (descriptor == "surf") feat_detector = cv::xfeatures2d::SURF::create(15, 4, 2);
#endif
    else throw std::runtime_error("invalid descriptor");

    assert(!descriptor.empty());

    std::cout << "extracting features ..." << std::endl;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    std::cout << "reading image: " << path_to_image << std::endl;
    cv::Mat image = cv::imread(path_to_image, 0);
    if (image.empty()) {
        std::cerr << "could not open image: " << path_to_image << std::endl;
        exit(EXIT_FAILURE);
    }
    feat_detector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
    std::cout << "extracted features: total = " << keypoints.size() << std::endl;
    std::cout << "done detecting features" << std::endl;

    return descriptors;
}

int main(int argc, char** argv) {
    CmdLineParser cml(argc, argv);
    try {
        if (argc < 3 || cml["-h"]) {
            std::cerr << "Usage: VOCABULARY IMAGE" << std::endl;
            std::cerr << std::endl;
            std::cerr << "Loads a vocabulary and am image." << std::endl;
            std::cerr << "Extracts image features and then compute the BoW of the image." << std::endl;
            std::cerr << std::endl;
            return EXIT_FAILURE;
        }
        fbow::Vocabulary vocab;
        vocab.readFromFile(argv[1]);

        std::string desc_name = vocab.getDescName();
        std::cout << "vocabulary descriptor: " << desc_name << std::endl;
        auto features = loadFeatures(std::string(argv[2]), desc_name);
        std::cout << "size: " << features.rows << " " << features.cols << std::endl;

        fbow::BoWVector bow_vec;
        auto t_start = std::chrono::high_resolution_clock::now();
        bow_vec = vocab.transform(features);
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "time: " << (double)(std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count()) << "ms" << std::endl;
        std::cout << std::endl;

        for (const auto& v : bow_vec) {
            std::cout << v.first << "(" << (float)v.second << ")" << " ";
        }
        std::cout << std::endl;
    } catch (std::exception& ex) {
        std::cerr << ex.what() << std::endl;
    }

    return EXIT_SUCCESS;
}
