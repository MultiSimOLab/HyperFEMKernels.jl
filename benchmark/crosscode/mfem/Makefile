# Build against an installed MFEM (with MPI): make MFEM_DIR=<prefix>
#   Homebrew/spack/conda install: config.mk is in <prefix>/share/mfem/config.mk
#   MFEM build directory:         make CONFIG_MK=<mfem-build>/config/config.mk
MFEM_DIR  ?= /usr/local
CONFIG_MK ?= $(MFEM_DIR)/share/mfem/config.mk
include $(CONFIG_MK)

bench: bench.cpp
	$(MFEM_CXX) $(MFEM_FLAGS) -O3 -std=c++17 $< -o $@ $(MFEM_LIBS)

clean:
	rm -f bench
