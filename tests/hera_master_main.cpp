// tests/hera_master_main.cpp
#include "hera_master.h"
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: ./hera_master <world_size>" << std::endl;
        return 1;
    }
    int size = atoi(argv[1]);
    hera::HeraMaster master(size, 9999);
    master.Run();
    return 0;
}