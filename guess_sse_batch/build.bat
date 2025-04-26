@echo off
g++ main.cpp train.cpp guessing.cpp md5.cpp -o main.exe -O1 -msse4.2 -march=native