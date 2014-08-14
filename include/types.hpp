#ifndef CUDA4CPU_TYPES_HPP_
#define CUDA4CPU_TYPES_HPP_

namespace cuda4cpu {

struct dim3 {
    unsigned x, y ,z;

    dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) :
        x(x_), y(y_), z(z_)
    {}
};

#define VECTOR_TYPE(n,t) \
    typedef struct {     \
        t x;             \
    } n##1;              \
                         \
    typedef struct {     \
        t x;             \
        t y;             \
    } n##2;              \
                         \
    typedef struct {     \
        t x;             \
        t y;             \
        t z;             \
    } n##3;              \
                         \
    typedef struct {     \
        t x;             \
        t y;             \
        t z;             \
        t w;             \
    } n##4;

VECTOR_TYPE(char, char)
VECTOR_TYPE(uchar, unsigned char)
VECTOR_TYPE(short, short)
VECTOR_TYPE(ushort, unsigned short)
VECTOR_TYPE(int, int)
VECTOR_TYPE(uint, unsigned)
VECTOR_TYPE(long, int long)
VECTOR_TYPE(ulong, unsigned long)
VECTOR_TYPE(longlong, int long long)
VECTOR_TYPE(ulonglong, unsigned long long)
VECTOR_TYPE(float, float)
VECTOR_TYPE(double, double)

}

#endif // CUDA4CPU_TYPES_HPP_
