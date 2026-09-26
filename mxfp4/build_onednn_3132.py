"""Isolated GPU-only oneDNN build, never a system or serving-library update."""
import json,os,pathlib,subprocess,time
root=pathlib.Path('/work');src=root/'onednn-3.13.2-src';build=root/'onednn-3.13.2-build-2026.0';dest=root/'onednn-3.13.2-install'
assert (src/'CMakeLists.txt').exists()
assert subprocess.check_output(['git','-c','safe.directory='+str(src),'-C',str(src),'rev-parse','HEAD'],text=True).strip()=='71094df451909e95ceeefc8f64c275cdcbbcdff5'
compiler='/opt/intel/oneapi/compiler/2026.0/bin/'
env=os.environ.copy();env['PATH']=compiler+'/opt/venv/bin:'+env['PATH']
args=['cmake','-S',str(src),'-B',str(build),'-G','Ninja',
 '-DCMAKE_BUILD_TYPE=Release','-DCMAKE_C_COMPILER='+compiler+'icx',
 '-DCMAKE_CXX_COMPILER='+compiler+'icpx','-DCMAKE_INSTALL_PREFIX='+str(dest),
 '-DDNNL_CPU_RUNTIME=NONE','-DDNNL_GPU_RUNTIME=SYCL','-DDNNL_GPU_VENDOR=INTEL',
 '-DDNNL_BUILD_GRAPH=OFF','-DDNNL_BUILD_TESTS=OFF','-DDNNL_BUILD_EXAMPLES=OFF',
 '-DDNNL_ENABLE_PRIMITIVE=MATMUL;REORDER','-DDNNL_ENABLE_WORKLOAD=INFERENCE',
 '-DDNNL_ENABLE_PRIMITIVE_GPU_ISA=XE2','-DDNNL_LIBRARY_NAME=vllm_xpu_dnnl_3132']
receipt=root/'adaptive'/'onednn-3132-build.json';log=root/'adaptive'/'onednn-3132-build.log'
result={'source_revision':'71094df451909e95ceeefc8f64c275cdcbbcdff5','configure_args':args,'install_prefix':str(dest),'system_update':False,'success':False}
start=time.monotonic()
with log.open('w')as f:
 for cmd in (args,['cmake','--build',str(build),'--parallel','4'],['cmake','--install',str(build)]):
  print('BUILD_STAGE',cmd[0:3],flush=True)
  r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT);f.flush()
  if r.returncode:
   result.update(failed_command=cmd,returncode=r.returncode,elapsed_s=time.monotonic()-start)
   receipt.write_text(json.dumps(result,indent=2));print(log.read_text()[-6000:],flush=True);raise SystemExit(r.returncode)
result.update(success=True,elapsed_s=time.monotonic()-start)
receipt.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
