# cmake -S . -B ./build
make -C build -j$(nproc) --silent

./build/main
