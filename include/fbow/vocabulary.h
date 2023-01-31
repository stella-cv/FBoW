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

#ifndef FBOW_VOCABULARY_H_
#define FBOW_VOCABULARY_H_

#include "fbow_exports.h"
#include "type.h"
#include "bow_vector.h"
#include "bow_feat_vector.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <map>
#include <memory>
#include <bitset>
#if !defined(__ANDROID__) && !defined(__arm64__) && !defined(__arm__) && !defined(__aarch64__)
#if defined(USE_AVX)
#include <immintrin.h>
#endif
#endif
#include "cpu.h"

namespace fbow {

/**Main class to represent a vocabulary of visual words
 */
class FBOW_API Vocabulary {
    friend class VocabularyCreator;

public:
    ~Vocabulary();

    //transform the features stored as rows in the returned BagOfWords
    BoWVector transform(const cv::Mat& features);
    void transform(const cv::Mat& features, int level, BoWVector& result, BoWFeatVector& result2);

    //loads/saves from a file
    void readFromFile(const std::string& filepath);
    void saveToFile(const std::string& filepath);
    ///save/load to binary streams
    void toStream(std::ostream& str) const;
    void fromStream(std::istream& str);
    //returns the descriptor type (CV_8UC1, CV_32FC1  )
    uint32_t getDescType() const { return _params._desc_type; }
    //returns desc size in bytes or 0 if not set
    uint32_t getDescSize() const { return _params._desc_size; }
    //returns the descriptor name
    std::string getDescName() const { return _params._desc_name_; }
    //returns the branching factor (number of children per node)
    uint32_t getK() const { return _params._m_k; }
    //indicates whether this object is valid
    bool isValid() const { return _data != 0; }
    //total number of blocks
    size_t size() const { return _params._nblocks; }
    //removes all data
    void clear();
    //returns a hash value idinfying the vocabulary
    uint64_t hash() const;

private:
    void setParams(int aligment, int k, int desc_type, int desc_size, int nblocks, std::string desc_name);
    struct params {
        char _desc_name_[50];                 //descriptor name. May be empty
        uint32_t _aligment = 0, _nblocks = 0; //memory aligment and total number of blocks
        uint64_t _desc_size_bytes_wp = 0;     //size of the descriptor(includes padding)
        uint64_t _block_size_bytes_wp = 0;    //size of a block   (includes padding)
        uint64_t _feature_off_start = 0;      //within a block, where the features start
        uint64_t _child_off_start = 0;        //within a block,where the children offset part starts
        uint64_t _total_size = 0;
        int32_t _desc_type = 0, _desc_size = 0; //original descriptor types and sizes (without padding)
        uint32_t _m_k = 0;                      //number of children per node
    };
    params _params;
    char* _data = nullptr; //pointer to data

    //structure represeting a information about node in a block
    struct block_node_info {
        uint32_t id_or_childblock; //if id ,msb is 1.
        float weight;
        inline bool isleaf() const { return (id_or_childblock & 0x80000000); }

        //if not leaf, returns the block where the children are
        //if leaf, returns the index of the feature it represents. In case of bagofwords it must be a invalid value
        inline uint32_t getId() const { return (id_or_childblock & 0x7FFFFFFF); }

        //sets as leaf, and sets the index of the feature it represents and its weight
        inline void setLeaf(uint32_t id, float Weight) {
            assert(!(id & 0x80000000)); //check msb is zero
            id_or_childblock = id;
            id_or_childblock |= 0x80000000; //set the msb to one to distinguish from non leaf
            //now,set the weight too
            weight = Weight;
        }
        //sets as non leaf and sets the id of the block where the chilren are
        inline void setNonLeaf(uint32_t id) {
            //ensure the msb is 0
            assert(!(id & 0x80000000)); //32 bits 100000000...0.check msb is not set
            id_or_childblock = id;
        }
    };

    //a block represent all the child nodes of a parent, with its features and also information about where the child of these are in the data structure
    //a block structure is as follow: N|isLeaf|BlockParentId|p|F0...FN|C0W0 ... CNWN..
    //N :16 bits : number of nodes in this block. Must be <=branching factor k. If N<k, then the block has empty spaces since block size is fixed
    //isLeaf:16 bit inicating if all nodes in this block are leaf or not
    //BlockParentId:31: id of the parent
    //p :possible offset so that Fi is aligned
    //Fi feature of the node i. it is aligned and padding added to the end so that F(i+1) is also aligned
    //CiWi are the so called block_node_info (see structure up)
    //Ci : either if the node is leaf (msb is set to 1) or not. If not leaf, the remaining 31 bits is the block where its children are. Else, it is the index of the feature that it represent
    //Wi: float value empkoyed to know the weight of a leaf node (employed in cases of bagofwords)
    struct Block {
        Block(char* bsptr, uint64_t ds, uint64_t ds_wp, uint64_t fo, uint64_t co)
            : _blockstart(bsptr), _desc_size_bytes(ds), _desc_size_bytes_wp(ds_wp), _feature_off_start(fo), _child_off_start(co) {}
        Block(uint64_t ds, uint64_t ds_wp, uint64_t fo, uint64_t co)
            : _desc_size_bytes(ds), _desc_size_bytes_wp(ds_wp), _feature_off_start(fo), _child_off_start(co) {}

        inline uint16_t getN() const { return (*((uint16_t*)(_blockstart))); }
        inline void setN(uint16_t n) { *((uint16_t*)(_blockstart)) = n; }

        inline bool isLeaf() const { return *((uint16_t*)(_blockstart) + 1); }
        inline void setLeaf(bool /*v*/) const { *((uint16_t*)(_blockstart) + 1) = 1; }

        inline void setParentId(uint32_t pid) { *(((uint32_t*)(_blockstart)) + 1) = pid; }
        inline uint32_t getParentId() { return *(((uint32_t*)(_blockstart)) + 1); }

        inline block_node_info* getBlockNodeInfo(int i) { return (block_node_info*)(_blockstart + _child_off_start + i * sizeof(block_node_info)); }
        inline void setFeature(int i, const cv::Mat& feature) { memcpy(_blockstart + _feature_off_start + i * _desc_size_bytes_wp, feature.ptr<char>(0), feature.elemSize1() * feature.cols); }
        inline void getFeature(int i, cv::Mat feature) { memcpy(feature.ptr<char>(0), _blockstart + _feature_off_start + i * _desc_size_bytes, _desc_size_bytes); }
        template<typename T>
        inline T* getFeature(int i) { return (T*)(_blockstart + _feature_off_start + i * _desc_size_bytes_wp); }
        char* _blockstart;
        uint64_t _desc_size_bytes = 0;    //size of the descriptor(without padding)
        uint64_t _desc_size_bytes_wp = 0; //size of the descriptor(includding padding)
        uint64_t _feature_off_start = 0;
        uint64_t _child_off_start = 0; //into the block,where the children offset part starts
    };

    //returns a block structure pointing at block b
    inline Block getBlock(uint32_t b) {
        assert(_data != 0);
        assert(b < _params._nblocks);
        return Block(_data + b * _params._block_size_bytes_wp, _params._desc_size, _params._desc_size_bytes_wp, _params._feature_off_start, _params._child_off_start);
    }
    //given a block already create with getBlock, moves it to point to block b
    inline void setBlock(uint32_t b, Block& block) { block._blockstart = _data + b * _params._block_size_bytes_wp; }

    //information about the cpu so that mmx,sse or avx extensions can be employed
    std::shared_ptr<cpu> cpu_info;

    template<typename Computer>
    BoWVector _transform(const cv::Mat& features) {
        Computer comp;
        comp.setParams(_params._desc_size, _params._desc_size_bytes_wp);
        using DType = typename Computer::DType; //distance type
        using TData = typename Computer::TData; //data type

        BoWVector result;
        std::pair<DType, uint32_t> best_dist_idx(std::numeric_limits<uint32_t>::max(), 0); //minimum distance found
        block_node_info* bn_info;
        for (int cur_feature = 0; cur_feature < features.rows; cur_feature++) {
            comp.startwithfeature(features.ptr<TData>(cur_feature));
            //ensure feature is in a
            Block c_block = getBlock(0);
            //copy to another structure and add padding with zeros
            do {
                //given the current block, finds the node with minimum distance
                best_dist_idx.first = std::numeric_limits<uint32_t>::max();
                for (int cur_node = 0; cur_node < c_block.getN(); cur_node++) {
                    DType d = comp.computeDist(c_block.getFeature<TData>(cur_node));
                    if (d < best_dist_idx.first)
                        best_dist_idx = std::make_pair(d, cur_node);
                }
                bn_info = c_block.getBlockNodeInfo(best_dist_idx.second);
                //if the node is leaf get word id and weight,else go to its children
                if (bn_info->isleaf()) { //if the node is leaf get word id and weight
                    result[bn_info->getId()] += bn_info->weight;
                }
                else
                    setBlock(bn_info->getId(), c_block); //go to its children
            } while (!bn_info->isleaf() && bn_info->getId() != 0);
        }
        return result;
    }
    template<typename Computer>
    void _transform2(const cv::Mat& features, uint32_t storeLevel, BoWVector& r1, BoWFeatVector& r2) {
        Computer comp;
        comp.setParams(_params._desc_size, _params._desc_size_bytes_wp);
        using DType = typename Computer::DType; //distance type
        using TData = typename Computer::TData; //data type

        r1.clear();
        r2.clear();
        std::pair<DType, uint32_t> best_dist_idx(std::numeric_limits<uint32_t>::max(), 0); //minimum distance found
        block_node_info* bn_info;
        int nbits = ceil(log2(_params._m_k));
        for (int cur_feature = 0; cur_feature < features.rows; cur_feature++) {
            comp.startwithfeature(features.ptr<TData>(cur_feature));
            //ensure feature is in a
            Block c_block = getBlock(0);
            uint32_t level = 0;   //current level of recursion
            uint32_t curNode = 0; //id of the current node of the tree
            //copy to another structure and add padding with zeros
            do {
                //given the current block, finds the node with minimum distance
                best_dist_idx.first = std::numeric_limits<uint32_t>::max();
                for (int cur_node = 0; cur_node < c_block.getN(); cur_node++) {
                    DType d = comp.computeDist(c_block.getFeature<TData>(cur_node));
                    if (d < best_dist_idx.first)
                        best_dist_idx = std::make_pair(d, cur_node);
                }
                if (level == storeLevel) //if reached level,save
                    r2[curNode].push_back(cur_feature);

                bn_info = c_block.getBlockNodeInfo(best_dist_idx.second);
                //if the node is leaf get weight,else go to its children
                if (bn_info->isleaf()) {
                    r1[bn_info->getId()] += bn_info->weight;
                    if (level < storeLevel) //store level not reached, save now
                        r2[curNode].push_back(cur_feature);
                    break;
                }
                else
                    setBlock(bn_info->getId(), c_block); //go to its children
                curNode = curNode << nbits;
                curNode |= best_dist_idx.second;
                level++;
            } while (!bn_info->isleaf() && bn_info->getId() != 0);
        }
    }
};

} // namespace fbow

#endif // FBOW_VOCABULARY_H_