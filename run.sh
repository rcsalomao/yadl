export LD_LIBRARY_PATH="$HOME/.local/lib/:$LD_LIBRARY_PATH"

cmake -S . -B ./build
make -C ./build -j8
./build/main
