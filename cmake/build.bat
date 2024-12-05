cd build
REM rm * -rf
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-std=c++17" ..
cmake --build . --config Release 
cp src/Release/* ../dist/ 
cd ..
dist\ncnn_project.exe