from pathlib import Path
import json,hashlib,time,tarfile,subprocess
root=Path(__file__).resolve().parent
while not (root/'final-complete.json').exists():time.sleep(1)
# Keep exact source/collector/log/manifest files, not executables or dependency trees.
files=[]
for p in root.rglob('*'):
 if not p.is_file():continue
 rel=p.relative_to(root)
 if rel.parts[0] in {'deps','mpi','mpi-debs','__pycache__'} or '__pycache__' in rel.parts:continue
 if p.name.startswith('._') or 'nvshmem_shim' in rel.parts:continue
 if p.suffix not in {'.py','.cu','.cpp','.c','.json','.log','.txt'} and p.name!='Makefile':continue
 files.append(p)
packages=[]
for p in sorted((root/'mpi-debs').glob('*.deb')):
 packages.append(dict(file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),metadata=subprocess.check_output(['dpkg-deb','-f',str(p),'Package','Version'],text=True)))
(root/'mpi-packages.json').write_text(json.dumps(packages,indent=2)+'\n');files.append(root/'mpi-packages.json')
# Extra host/compiler facts not already covered by the collectors.
for name,cmd in [('host-os',['cat','/etc/os-release']),('host-gcc',['gcc','--version']),('host-sanitizer',['compute-sanitizer','--version'])]:
 p=root/(name+'.log');p.write_text('$ '+' '.join(cmd)+'\n'+subprocess.check_output(cmd,text=True));files.append(p)
files=sorted(set(files));inventory={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(root/'evidence-integrity.json').write_text(json.dumps(dict(files=inventory),indent=2)+'\n')
archive=Path('/home/ubuntu/book-b200-20261119-evidence.tgz')
with tarfile.open(archive,'w:gz') as tar:
 for p in files+[root/'evidence-integrity.json']:tar.add(p,arcname=p.relative_to(root),recursive=False)
print('READY',archive,archive.stat().st_size,flush=True)
