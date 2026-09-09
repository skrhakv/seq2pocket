#!/bin/bash
# Installs Foldseek (prebuilt static binary) and builds ProBiS (Konc & Janezic,
# Bioinformatics 2010) from source, both into vendor/ (gitignored, nothing here
# is committed).
set -e
cd "$(dirname "$0")"
mkdir -p vendor

# --- Foldseek: prebuilt static binary, no build needed ---------------------
if [ ! -x vendor/foldseek/bin/foldseek ]; then
    echo "=== Foldseek ==="
    curl -L https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz -o vendor/foldseek.tar.gz
    tar -xzf vendor/foldseek.tar.gz -C vendor
    rm vendor/foldseek.tar.gz
fi
vendor/foldseek/bin/foldseek version

# --- ProBiS: build from source, needs GSL ----------------------------------
if ! command -v gsl-config >/dev/null 2>&1; then
    echo
    echo "GSL not found. This machine has no system GSL and I'm not running"
    echo "sudo myself - install it yourself, then re-run this script:"
    echo
    echo "    sudo dnf install gsl-devel"
    echo
    exit 1
fi

if [ ! -x vendor/probis-src/probis ]; then
    echo "=== ProBiS ==="
    git clone --quiet https://gitlab.com/janezkonc/probis.git vendor/probis-src
    # Upstream motif.h's comparator isn't const-correct, which newer
    # libstdc++ (GCC 13+) rejects for std::set - patch before building.
    sed -i 's/bool operator()(string a, string b) {/bool operator()(const string \&a, const string \&b) const {/' \
        vendor/probis-src/motif.h
    make -C vendor/probis-src probis CC=g++ MPICC=g++ \
        CFLAGS="-std=c++11 -O3 -Wall -Wno-unused-function -Wno-char-subscripts -DNDEBUG"
fi
vendor/probis-src/probis 2>&1 | head -1

echo
echo "Done. foldseek: $(pwd)/vendor/foldseek/bin/foldseek"
echo "      probis:   $(pwd)/vendor/probis-src/probis"
