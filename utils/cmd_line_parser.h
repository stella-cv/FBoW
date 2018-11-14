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

#ifndef FBOW_CMD_LINE_PARSER_H
#define FBOW_CMD_LINE_PARSER_H

#include <string>

class CmdLineParser {
    int argc;
    char** argv;
public:
    CmdLineParser(int _argc, char** _argv) : argc(_argc), argv(_argv) {}
    bool operator[](const std::string& param) {
        int idx = -1;
        for (int i = 0; i < argc && idx == -1; i++) {
            if (std::string(argv[i]) == param) {
                idx = i;
            }
        }
        return (idx != -1);
    }
    std::string operator()(const std::string& param, const std::string& defvalue = "-1") {
        int idx = -1;
        for (int i = 0; i < argc && idx == -1; i++) {
            if (std::string(argv[i]) == param) {
                idx = i;
            }
        }
        if (idx == -1) {
            return defvalue;
        }
        else {
            return (argv[idx + 1]);
        }
    }
};

#endif // FBOW_CMD_LINE_PARSER_H
