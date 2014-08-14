set(CUDART_HEADERS cuda4cpu.hpp)

install(FILES ${CUDART_HEADERS}
        DESTINATION include/cuda4cpu
        COMPONENT headers)
