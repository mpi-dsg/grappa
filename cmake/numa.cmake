# Try to find the library
find_library(NUMA_LIBRARY NAMES numa)

# Try to find the header directory
find_path(NUMA_INCLUDE_DIR NAMES numa.h)

# Set the NUMA_FOUND variable if both library and headers were found
if(NUMA_LIBRARY AND NUMA_INCLUDE_DIR)
    set(NUMA_FOUND TRUE)
    set(NUMA_LIBRARIES ${NUMA_LIBRARY})
    set(NUMA_INCLUDE_DIRS ${NUMA_INCLUDE_DIR})

else()
    message(FATAL_ERROR "libnuma not found")
endif()
