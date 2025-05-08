export LD_LIBRARY_PATH="$HOME/.local/lib/:$LD_LIBRARY_PATH"

cmake -S . -B ./build -DCMAKE_CXX_COMPILER="$GCC_DIR/bin/g++"
make -C ./build -j$(nproc)
./build/main
