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

#include "fbow/fbow.h"
#include <fstream>
#include <cstring>
#include <limits>
#include <cstdint>
#include <algorithm>

namespace fbow{

inline void* AlignedAlloc(int __alignment, int size) {
    assert(__alignment < 256);

    unsigned char* ptr = (unsigned char*)malloc(size + __alignment);

    if (!ptr)
        return 0;

    // align the pointer

    size_t lptr = (size_t)ptr;
    int off = lptr % __alignment;
    if (off == 0)
        off = __alignment;

    ptr = ptr + off;                 //move to next aligned address
    *(ptr - 1) = (unsigned char)off; //save in prev, the offset  to properly remove it
    return ptr;
}
inline void AlignedFree(void* ptr) {
    unsigned char* uptr = (unsigned char*)ptr;
    unsigned char off = *(uptr - 1);
    uptr -= off;
    std::free(uptr);
}

////////////////////////////////////////////////////////////
//base class for computing distances between feature vectors
template<typename register_type, typename distType, int aligment>
class Lx {
public:
    typedef distType DType;
    typedef register_type TData;

protected:
    int _nwords, _aligment, _desc_size;
    int _block_desc_size_bytes_wp;
    register_type* feature = 0;

public:
    virtual ~Lx() {
        if (feature != 0)
            AlignedFree(feature);
    }
    void setParams(int desc_size, int block_desc_size_bytes_wp) {
        assert(block_desc_size_bytes_wp % aligment == 0);
        _desc_size = desc_size;
        _block_desc_size_bytes_wp = block_desc_size_bytes_wp;
        assert(_block_desc_size_bytes_wp % sizeof(register_type) == 0);
        _nwords = _block_desc_size_bytes_wp / sizeof(register_type); //number of aligned words
        feature = static_cast<register_type*>(AlignedAlloc(aligment, _nwords * sizeof(register_type)));
        memset(feature, 0, _nwords * sizeof(register_type));
    }
    inline void startwithfeature(const register_type* feat_ptr) { memcpy(feature, feat_ptr, _desc_size); }
    virtual distType computeDist(register_type* fptr) = 0;
};

struct L2_generic : public Lx<float, float, 4> {
    virtual ~L2_generic() {}
    inline float computeDist(float* fptr) {
        float d = 0;
        for (int f = 0; f < _nwords; f++)
            d += (feature[f] - fptr[f]) * (feature[f] - fptr[f]);
        return d;
    }
};
#if defined(__ANDROID__) || defined(__arm64__) || defined(__aarch64__)
//fake elements to allow compilation
struct L2_avx_generic : public Lx<uint64_t, float, 32> {
    inline float computeDist(uint64_t* ptr) { return std::numeric_limits<float>::max(); }
};
struct L2_se3_generic : public Lx<uint64_t, float, 32> {
    inline float computeDist(uint64_t* ptr) { return std::numeric_limits<float>::max(); }
};
struct L2_sse3_16w : public Lx<uint64_t, float, 32> {
    inline float computeDist(uint64_t* ptr) { return std::numeric_limits<float>::max(); }
};
struct L2_avx_8w : public Lx<uint64_t, float, 32> {
    inline float computeDist(uint64_t* ptr) { return std::numeric_limits<float>::max(); }
};

#else
struct L2_avx_generic : public Lx<__m256, float, 32> {
    virtual ~L2_avx_generic() {}
    inline float computeDist(__m256* ptr) {
        __m256 sum = _mm256_setzero_ps(), sub_mult;
        //substract, multiply and accumulate
        for (int i = 0; i < _nwords; i++) {
            sub_mult = _mm256_sub_ps(feature[i], ptr[i]);
            sub_mult = _mm256_mul_ps(sub_mult, sub_mult);
            sum = _mm256_add_ps(sum, sub_mult);
        }
        sum = _mm256_hadd_ps(sum, sum);
        sum = _mm256_hadd_ps(sum, sum);
        float* sum_ptr = (float*)&sum;
        return sum_ptr[0] + sum_ptr[4];
    }
};
struct L2_se3_generic : public Lx<__m128, float, 16> {
    inline float computeDist(__m128* ptr) {
        __m128 sum = _mm_setzero_ps(), sub_mult;
        //substract, multiply and accumulate
        for (int i = 0; i < _nwords; i++) {
            sub_mult = _mm_sub_ps(feature[i], ptr[i]);
            sub_mult = _mm_mul_ps(sub_mult, sub_mult);
            sum = _mm_add_ps(sum, sub_mult);
        }
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        float* sum_ptr = (float*)&sum;
        return sum_ptr[0];
    }
};
struct L2_sse3_16w : public Lx<__m128, float, 16> {
    inline float computeDist(__m128* ptr) {
        __m128 sum = _mm_setzero_ps(), sub_mult;
        //substract, multiply and accumulate
        for (int i = 0; i < 16; i++) {
            sub_mult = _mm_sub_ps(feature[i], ptr[i]);
            sub_mult = _mm_mul_ps(sub_mult, sub_mult);
            sum = _mm_add_ps(sum, sub_mult);
        }
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        float* sum_ptr = (float*)&sum;
        return sum_ptr[0];
    }
};
//specific for surf in avx
struct L2_avx_8w : public Lx<__m256, float, 32> {
    inline float computeDist(__m256* ptr) {
        __m256 sum = _mm256_setzero_ps(), sub_mult;
        //substract, multiply and accumulate

        for (int i = 0; i < 8; i++) {
            sub_mult = _mm256_sub_ps(feature[i], ptr[i]);
            sub_mult = _mm256_mul_ps(sub_mult, sub_mult);
            sum = _mm256_add_ps(sum, sub_mult);
        }

        sum = _mm256_hadd_ps(sum, sum);
        sum = _mm256_hadd_ps(sum, sum);
        float* sum_ptr = (float*)&sum;
        return sum_ptr[0] + sum_ptr[4];
    }
};

#endif

//generic hamming distance calculator
struct L1_x64 : public Lx<uint64_t, uint64_t, 8> {
    inline uint64_t computeDist(uint64_t* feat_ptr) {
        uint64_t result = 0;
        for (int i = 0; i < _nwords; ++i)
            result += std::bitset<64>(feat_ptr[i] ^ feature[i]).count();
        return result;
    }
};

struct L1_x32 : public Lx<uint32_t, uint32_t, 8> {
    inline uint32_t computeDist(uint32_t* feat_ptr) {
        uint32_t result = 0;
        for (int i = 0; i < _nwords; ++i)
            result += std::bitset<32>(feat_ptr[i] ^ feature[i]).count();
        return result;
    }
};

//for orb
struct L1_32bytes : public Lx<uint64_t, uint64_t, 8> {
    inline uint64_t computeDist(uint64_t* feat_ptr) {
        return uint64_popcnt(feat_ptr[0] ^ feature[0]) + uint64_popcnt(feat_ptr[1] ^ feature[1]) + uint64_popcnt(feat_ptr[2] ^ feature[2]) + uint64_popcnt(feat_ptr[3] ^ feature[3]);
    }
    inline uint64_t uint64_popcnt(uint64_t n) {
        return std::bitset<64>(n).count();
    }
};
//for akaze
struct L1_61bytes : public Lx<uint64_t, uint64_t, 8> {
    inline uint64_t computeDist(uint64_t* feat_ptr) {
        return uint64_popcnt(feat_ptr[0] ^ feature[0]) + uint64_popcnt(feat_ptr[1] ^ feature[1]) + uint64_popcnt(feat_ptr[2] ^ feature[2]) + uint64_popcnt(feat_ptr[3] ^ feature[3]) + uint64_popcnt(feat_ptr[4] ^ feature[4]) + uint64_popcnt(feat_ptr[5] ^ feature[5]) + uint64_popcnt(feat_ptr[6] ^ feature[6]) + uint64_popcnt(feat_ptr[7] ^ feature[7]);
    }
    inline uint64_t uint64_popcnt(uint64_t n) {
        return std::bitset<64>(n).count();
    }
};


Vocabulary::~Vocabulary(){
    if (_data!=nullptr) AlignedFree( _data);
}


void Vocabulary::setParams(int aligment, int k, int desc_type, int desc_size, int nblocks, std::string desc_name) {
    auto ns= desc_name.size()<static_cast<size_t>(49)?desc_name.size():128;
    desc_name.resize(ns);

    std::strcpy(_params._desc_name_,desc_name.c_str());
    _params._aligment=aligment;
    _params._m_k= k;
    _params._desc_type=desc_type;
    _params._desc_size=desc_size;
    _params._nblocks=nblocks;


    uint64_t _desc_size_bytes_al=0;
    uint64_t _block_size_bytes_al=0;

    //consider possible aligment of each descriptor adding offsets at the end
    _params._desc_size_bytes_wp=_params._desc_size;
    _desc_size_bytes_al= _params._desc_size_bytes_wp/ _params._aligment;
    if( _params._desc_size_bytes_wp% _params._aligment!=0)   _desc_size_bytes_al++;
    _params._desc_size_bytes_wp= _desc_size_bytes_al* _params._aligment;


    int foffnbytes_alg=sizeof(uint64_t)/_params._aligment;
    if(sizeof(uint64_t)%_params._aligment!=0) foffnbytes_alg++;
    _params._feature_off_start=foffnbytes_alg*_params._aligment;
    _params._child_off_start=_params._feature_off_start+_params._m_k*_params._desc_size_bytes_wp ;//where do children information start from the start of the block

    //block: nvalid|f0 f1 .. fn|ni0 ni1 ..nin
    _params._block_size_bytes_wp=_params._feature_off_start+  _params._m_k * ( _params._desc_size_bytes_wp + sizeof(Vocabulary::block_node_info));
    _block_size_bytes_al=_params._block_size_bytes_wp/_params._aligment;
    if (_params._block_size_bytes_wp%_params._aligment!=0) _block_size_bytes_al++;
    _params._block_size_bytes_wp= _block_size_bytes_al*_params._aligment;

    //give memory
    _params._total_size=_params._block_size_bytes_wp*_params._nblocks;
    _data=(char*)AlignedAlloc(_params._aligment,_params._total_size);
    memset( _data,0,_params._total_size);

}

void Vocabulary::transform(const cv::Mat &features, int level,BoWVector &result,BoWFeatVector&result2){
    if (features.rows==0) return;
    if (features.type()!=_params._desc_type) throw std::runtime_error("Vocabulary::transform features are of different type than vocabulary");
    if (features.cols *  features.elemSize() !=size_t(_params._desc_size)) throw std::runtime_error("Vocabulary::transform features are of different size than the vocabulary ones");

    //get host info to decide the version to execute
    if (!cpu_info){
        cpu_info=std::make_shared<cpu>();
        cpu_info->detect_host();
    }
     //decide the version to employ according to the type of features, aligment and cpu capabilities
    if (_params._desc_type==CV_8UC1){
        //orb
        if (cpu_info->HW_x64){
            if (_params._desc_size==32)
                 _transform2<L1_32bytes>(features,level,result,result2);
            //full akaze
            else if( _params._desc_size==61 && _params._aligment%8==0)
                _transform2<L1_61bytes>(features,level,result,result2);
            //generic
            else
                _transform2<L1_x64>(features,level,result,result2);
        }
        else  _transform2<L1_x32>(features,level,result,result2);
    }
    else if(features.type()==CV_32FC1){
        if( cpu_info->isSafeAVX() && _params._aligment%32==0){ //AVX version
            if ( _params._desc_size==256)  _transform2<L2_avx_8w>(features,level,result,result2);//specific for surf 256 bytes
            else  _transform2<L2_avx_generic>(features,level,result,result2);//any other
        }
        if( cpu_info->isSafeSSE() && _params._aligment%16==0){//SSE version
            if ( _params._desc_size==256) _transform2<L2_sse3_16w>(features,level,result,result2);//specific for surf 256 bytes
            else _transform2<L2_se3_generic>(features,level,result,result2);//any other
        }
        //generic version
        _transform2<L2_generic>(features,level,result,result2);
    }
    else throw std::runtime_error("Vocabulary::transform invalid feature type. Should be CV_8UC1 or CV_32FC1");

    ///now, normalize
    //L2
    double norm = 0;
    for (auto e : result) {
        norm += e.second * e.second;
    }

    if (norm > 0.0) {
        double inv_norm = 1. / sqrt(norm);
        for(auto& e : result) {
            e.second *= inv_norm;
        }
    }
}

BoWVector Vocabulary::transform(const cv::Mat &features)
{
    BoWVector result;
    if (features.rows==0) return result;
    if (features.type()!=_params._desc_type) throw std::runtime_error("Vocabulary::transform features are of different type than vocabulary");
    if (features.cols *  features.elemSize() !=size_t(_params._desc_size)) throw std::runtime_error("Vocabulary::transform features are of different size than the vocabulary ones");

    //get host info to decide the version to execute
    if (!cpu_info){
        cpu_info=std::make_shared<cpu>();
        cpu_info->detect_host();
    }
    //decide the version to employ according to the type of features, aligment and cpu capabilities
    if (_params._desc_type==CV_8UC1){
        //orb
        if (cpu_info->HW_x64){
            if (_params._desc_size==32)
                result=_transform<L1_32bytes>(features);
            //full akaze
            else if( _params._desc_size==61 && _params._aligment%8==0)
                result=_transform<L1_61bytes>(features);
            //generic
            else
                result=_transform<L1_x64>(features );
        }
        else  result=  _transform<L1_x32>(features );
    }
    else if(features.type()==CV_32FC1){
        if( cpu_info->isSafeAVX() && _params._aligment%32==0){ //AVX version
            if ( _params._desc_size==256) result= _transform<L2_avx_8w>(features);//specific for surf 256 bytes
            else result= _transform<L2_avx_generic>(features);//any other
        }
        if( cpu_info->isSafeSSE() && _params._aligment%16==0){//SSE version
            if ( _params._desc_size==256) result= _transform<L2_sse3_16w>(features);//specific for surf 256 bytes
            else result=_transform<L2_se3_generic>(features);//any other
        }
        //generic version
        result=_transform<L2_generic>(features);
    }
    else throw std::runtime_error("Vocabulary::transform invalid feature type. Should be CV_8UC1 or CV_32FC1");

    ///now, normalize
    //L2
    double norm=0;
    for(auto  e:result) norm += e.second * e.second;

    if(norm > 0.0)
    {
        double inv_norm = 1./sqrt(norm);
        for(auto  &e:result) e.second*=inv_norm ;
    }
    return result;
}



void Vocabulary::clear()
{
    if (_data!=0) AlignedFree(_data);
    _data=0;
    memset(&_params,0,sizeof(_params));
    _params._desc_name_[0]='\0';
}


//loads/saves from a file
void Vocabulary::readFromFile(const std::string &filepath){
    std::ifstream file(filepath,std::ios::binary);
    if (!file) throw std::runtime_error("Vocabulary::readFromFile could not open:"+filepath);
    fromStream(file);
}

void Vocabulary::saveToFile(const std::string &filepath){
	std::ofstream file(filepath, std::ios::binary);
    if (!file) throw std::runtime_error("Vocabulary::saveToFile could not open:"+filepath);
    toStream(file);

}

///save/load to binary streams
void Vocabulary::toStream(std::ostream &str)const{
    //magic number
    uint64_t sig=55824124;
    str.write((char*)&sig,sizeof(sig));
    //save string
    str.write((char*)&_params,sizeof(params));
    str.write(_data,_params._total_size);
}

void Vocabulary::fromStream(std::istream &str)
{
    if (_data!=0) AlignedFree (_data);
    uint64_t sig;
    str.read((char*)&sig,sizeof(sig));
    if (sig!=55824124) throw std::runtime_error("Vocabulary::fromStream invalid signature");
    //read string
    str.read((char*)&_params,sizeof(params));
    _data=(char*)AlignedAlloc(_params._aligment,_params._total_size);
    if (_data==0) throw std::runtime_error("Vocabulary::fromStream Could not allocate data");
    str.read(_data,_params._total_size);
}

double BoWVector::score (const  BoWVector &v1,const BoWVector &v2){


    BoWVector::const_iterator v1_it, v2_it;
    const BoWVector::const_iterator v1_end = v1.end();
    const BoWVector::const_iterator v2_end = v2.end();

    v1_it = v1.begin();
    v2_it = v2.begin();

    double score = 0;

    while(v1_it != v1_end && v2_it != v2_end)
    {
        const auto& vi = v1_it->second;
        const auto& wi = v2_it->second;

        if(v1_it->first == v2_it->first)
        {
            score += vi * wi;

            // move v1 and v2 forward
            ++v1_it;
            ++v2_it;
        }
        else if(v1_it->first < v2_it->first)
        {
            // move v1 forward
//            v1_it = v1.lower_bound(v2_it->first);
            while(v1_it!=v1_end&& v1_it->first<v2_it->first)
                ++v1_it;
        }
        else
        {
            // move v2 forward
//            v2_it = v2.lower_bound(v1_it->first);
            while(v2_it!=v2_end && v2_it->first<v1_it->first)
                ++v2_it;

            // v2_it = (first element >= v1_it.id)
        }
    }

    // ||v - w||_{L2} = sqrt( 2 - 2 * Sum(v_i * w_i) )
    //		for all i | v_i != 0 and w_i != 0 )
    // (Nister, 2006)
    if(score >= 1) // rounding errors
        score = 1.0;
    else
        score = 1.0 - sqrt(1.0 - score); // [0..1]

    return score;
}
uint64_t BoWVector::hash()const{
    uint64_t seed = 0;
    for(auto e:*this)
        seed^= e.first +  int(e.second*1000)+ 0x9e3779b9 + (seed << 6) + (seed >> 2);

    return seed;

}

uint64_t Vocabulary::hash()const{

    uint64_t seed = 0;
    for(uint64_t i=0;i<_params._total_size;i++)
        seed^= _data[i] + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
}
void BoWVector::toStream(std::ostream &str) const   {
    uint32_t _size=size();
    str.write((char*)&_size,sizeof(_size));
    for(const auto & e:*this)
        str.write((char*)&e,sizeof(e));
}
void BoWVector::fromStream(std::istream &str)    {
    clear();
    uint32_t _size;
    str.read((char*)&_size,sizeof(_size));
    for(uint32_t i=0;i<_size;i++){
        std::pair<uint32_t,_float> e;
        str.read((char*)&e,sizeof(e));
        insert(e);
    }
}

void BoWFeatVector::toStream(std::ostream &str) const   {
    uint32_t _size=size();
    str.write((char*)&_size,sizeof(_size));
    for(const auto &e:*this){
        str.write((char*)&e.first,sizeof(e.first));
        //now the vector
        _size=e.second.size();
        str.write((char*)&_size,sizeof(_size));
        str.write((char*)&e.second[0],sizeof(e.second[0])*e.second.size());
    }
}

void BoWFeatVector::fromStream(std::istream &str)    {
    uint32_t _sizeMap,_sizeVec;
    std::vector<uint32_t> vec;
    uint32_t key;

    clear();
    str.read((char*)&_sizeMap,sizeof(_sizeMap));
    for(uint32_t i=0;i<_sizeMap;i++){
        str.read((char*)&key,sizeof(key));
        str.read((char*)&_sizeVec,sizeof(_sizeVec));//vector size
        vec.resize(_sizeVec);
        str.read((char*)&vec[0],sizeof(vec[0])*_sizeVec);
        insert({key,vec});
    }
}

uint64_t BoWFeatVector::hash()const{


    uint64_t seed = 0;


    for(const auto &e:*this){
        seed^= e.first + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        for(const auto &idx:e.second)
            seed^= idx + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

    return seed;

}

}
