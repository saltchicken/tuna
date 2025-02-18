cmake -B build
cmake --build build --config Release
cp build/bin/llama.quantize .
