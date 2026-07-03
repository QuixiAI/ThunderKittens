ARCH ?= SM86
SRC := mha_ampere.cu
OUT := mha_ampere.out
CMD := python gentests.py randn 4096 && ./mha_ampere.out randn_4096_64.txt
CONFIG := standalone

include ../../common.mk
