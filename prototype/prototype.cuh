/**
 * @file
 * @brief A collection of all of ThunderKittens prototypes, that can be filled in to easily build full kernels.
 */

#pragma once

#include "../include/kittens.cuh"

#include "common/common.cuh"
#include "lcf/lcf.cuh"
#include "lcsf/lcsf.cuh"
// lcsc and the interpreter depend on Hopper+ features (TMA descriptors, etc.).
#if defined(KITTENS_SM90) || defined(KITTENS_SM10X) || defined(KITTENS_SM120)
#include "lcsc/lcsc.cuh"
#include "interpreter/interpreter.cuh"
#endif
