cmake -B build
cmake --build build --config Release
cp build/bin/llama-quantize .


sudo apt install libcurl4-openssl-dev
