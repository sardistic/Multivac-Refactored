{ pkgs }: {
  deps = [
    pkgs.gdb
    pkgs.zlib
    pkgs.tk
    pkgs.tcl
    pkgs.openjpeg
    pkgs.libxcrypt
    pkgs.libwebp
    pkgs.libtiff
    pkgs.libjpeg
    pkgs.libimagequant
    pkgs.lcms2
    pkgs.freetype
    pkgs.sqlite.bin
    pkgs.sudo
    pkgs.replitPackages.prybar-python310
    pkgs.replitPackages.stderred
  ];
}