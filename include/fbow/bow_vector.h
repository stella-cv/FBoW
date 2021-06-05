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

#ifndef FBOW_BOW_VECTOR_H_
#define FBOW_BOW_VECTOR_H_

#include "fbow_exports.h"
#include "type.h"
#include <iostream>
#include <map>

namespace fbow {

/**Bag of words
 */
struct FBOW_API BoWVector : std::map<uint32_t, _float> {
    void toStream(std::ostream& str) const;
    void fromStream(std::istream& str);

    //returns a hash identifying this
    uint64_t hash() const;
    //returns the similitude score between to image descriptors using L2 norm
    static double score(const BoWVector& v1, const BoWVector& v2);
};

} // namespace fbow

#endif // FBOW_BOW_VECTOR_H_