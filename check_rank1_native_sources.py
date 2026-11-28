from pathlib import Path
import subprocess,re,os
root=Path(__file__).resolve().parent
build=root/"build-rank1-sm89"
ninja=r"C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
for f in ("src/models/CMakeFiles/ninfer_model_runtime.dir/inkling2/executor.cpp.obj",
          "src/models/CMakeFiles/ninfer_model_loading.dir/inkling2/model.cpp.obj"):
    cmd=subprocess.check_output([ninja,"-C",str(build),"-t","commands",f],text=True).splitlines()[-1]
    cmd=re.sub(r" /Fo\S+| /Fd\S+| /FS| /showIncludes| -c ", " ",cmd)
    cmd=cmd.replace(" /nologo"," /nologo /Zs")
    print("SYNTAX",f,flush=True)
    subprocess.run(cmd,cwd=build,check=True)
    print("PASS",f,flush=True)
