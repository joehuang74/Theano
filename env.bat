   REM configuration of paths
   set VSFORPYTHON="C:\Users\j\AppData\Local\Programs\Common\Microsoft\Visual C++ for Python\9.0"
   set GCC="C:\TDM-GCC-64"

   REM add tdm gcc stuff
   set PATH=%GCC%\bin;%GCC%\x86_64-w64-mingw32\bin;%PATH%

   REM add winpython stuff
   CALL D:\softwares\WinPython-64bit-2.7.10.3\scripts\env.bat

   REM configure path for msvc compilers
   REM for a 32 bit installation change this line to
   REM CALL %VSFORPYTHON%\vcvarsall.bat
   CALL %VSFORPYTHON%\vcvarsall.bat amd64

   REM return a shell
   cmd.exe /k
   