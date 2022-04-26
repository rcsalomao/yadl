# cd ./build; cmake ..; cd ..
make -C build -j$(nproc) --silent

./build/main
