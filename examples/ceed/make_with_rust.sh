rustc --crate-type=staticlib -Clinker-plugin-lto -Copt-level=3 -C panic=abort ./bruhh.rs
clang -c -O3 -flto=thin -o ex1-volume.o ./ex1-volume.c -I../../include -std=c11
#clang -I../../include -std=c11  -flto=thin -fuse-ld=lld -o ex1-volume -O2 ./ex1-volume.o
clang -flto=thin -fuse-ld=lld -I../../include -std=c11 -O3 -g  ex1-volume.o -o ex1-volume -Wl,-rpath,$(realpath ../../lib) -L../../lib -lceed -L../../examples/ceed -lbruhh -lm
