export LD_LIBRARY_PATH="$HOME/.local/lib/:$HOME/.local/bin_files/gcc/lib64/:$LD_LIBRARY_PATH"

cmake -S . -B ./build
make -C ./build -j8
./build/main
