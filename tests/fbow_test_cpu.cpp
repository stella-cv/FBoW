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

#include "cpu.h"

#include <iostream>

using namespace std;
using namespace fbow;

void print(const char* label, bool yes) {
    cout << label;
    cout << (yes ? "Yes" : "No") << endl;
}

void print(cpu host_info) {
    cout << "CPU Vendor:" << endl;
    print("    AMD         = ", host_info.Vendor_AMD);
    print("    Intel       = ", host_info.Vendor_Intel);
    cout << endl;

    cout << "OS Features:" << endl;
#ifdef _WIN32
    print("    64-bit      = ", host_info.OS_x64);
#endif
    print("    OS AVX      = ", host_info.OS_AVX);
    print("    OS AVX512   = ", host_info.OS_AVX512);
    cout << endl;

    cout << "Hardware Features:" << endl;
    print("    MMX         = ", host_info.HW_MMX);
    print("    x64         = ", host_info.HW_x64);
    print("    ABM         = ", host_info.HW_ABM);
    print("    RDRAND      = ", host_info.HW_RDRAND);
    print("    BMI1        = ", host_info.HW_BMI1);
    print("    BMI2        = ", host_info.HW_BMI2);
    print("    ADX         = ", host_info.HW_ADX);
    print("    MPX         = ", host_info.HW_MPX);
    print("    PREFETCHWT1 = ", host_info.HW_PREFETCHWT1);
    cout << endl;

    cout << "SIMD: 128-bit" << endl;
    print("    SSE         = ", host_info.HW_SSE);
    print("    SSE2        = ", host_info.HW_SSE2);
    print("    SSE3        = ", host_info.HW_SSE3);
    print("    SSSE3       = ", host_info.HW_SSSE3);
    print("    SSE4a       = ", host_info.HW_SSE4a);
    print("    SSE4.1      = ", host_info.HW_SSE41);
    print("    SSE4.2      = ", host_info.HW_SSE42);
    print("    AES-NI      = ", host_info.HW_AES);
    print("    SHA         = ", host_info.HW_SHA);
    cout << endl;

    cout << "SIMD: 256-bit" << endl;
    print("    AVX         = ", host_info.HW_AVX);
    print("    XOP         = ", host_info.HW_XOP);
    print("    FMA3        = ", host_info.HW_FMA3);
    print("    FMA4        = ", host_info.HW_FMA4);
    print("    AVX2        = ", host_info.HW_AVX2);
    cout << endl;

    cout << "SIMD: 512-bit" << endl;
    print("    AVX512-F    = ", host_info.HW_AVX512_F);
    print("    AVX512-CD   = ", host_info.HW_AVX512_CD);
    print("    AVX512-PF   = ", host_info.HW_AVX512_PF);
    print("    AVX512-ER   = ", host_info.HW_AVX512_ER);
    print("    AVX512-VL   = ", host_info.HW_AVX512_VL);
    print("    AVX512-BW   = ", host_info.HW_AVX512_BW);
    print("    AVX512-DQ   = ", host_info.HW_AVX512_DQ);
    print("    AVX512-IFMA = ", host_info.HW_AVX512_IFMA);
    print("    AVX512-VBMI = ", host_info.HW_AVX512_VBMI);
    cout << endl;

    cout << "Summary:" << endl;
    print("    Safe to use AVX:     ", host_info.HW_AVX && host_info.OS_AVX);
    print("    Safe to use AVX512:  ", host_info.HW_AVX512_F && host_info.OS_AVX512);
    cout << endl;
}

int main() {
    cout << endl;
    cout << "CPU Vendor String: " << cpu::get_vendor_string() << endl;
    cout << endl;
    cpu features;
    features.detect_host();
    print(features);

    return EXIT_SUCCESS;
}
