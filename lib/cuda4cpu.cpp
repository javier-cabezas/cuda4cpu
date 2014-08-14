#include <cstdlib>

#include "cuda4cpu.hpp"

namespace cuda4cpu {

// static variables
system system::sys;
thread_local thread_block *
thread_block::Current = nullptr;

}
