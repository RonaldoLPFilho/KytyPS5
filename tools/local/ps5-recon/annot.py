import subprocess,re,sys
SRC='/home/chen-xiao/nvme1n1/game/PPSA01341-app0/decrypted/eboot.bin'
data=open(SRC,'rb').read()
# string map: vaddr -> text
smap={}
for m in re.finditer(rb'[\x20-\x7e]{2,}',data):
    fo=m.start()
    if 0x2000000<=fo<0x2900000:
        smap.setdefault(fo-0x4000,m.group().decode('latin1'))
def resolve(v):
    if v in smap: return smap[v]
    return None
def annot(startva,endva,label=''):
    blob=data[startva+0x4000:endva+0x4000]
    open('/tmp/ps5recon/w2.bin','wb').write(blob)
    out=subprocess.run(['objdump','-D','-b','binary','-m','i386:x86-64','-M','intel',
        '--adjust-vma=0x%x'%startva,'/tmp/ps5recon/w2.bin'],capture_output=True,text=True).stdout
    print("#### %s  [va 0x%x..0x%x]"%(label,startva,endva))
    for ln in out.splitlines():
        m=re.match(r'\s*([0-9a-f]+):\s+((?:[0-9a-f]{2} )+)\s*(.*)',ln)
        if not m: continue
        a=int(m.group(1),16); asm=m.group(3)
        t=re.search(r'# 0x([0-9a-f]+)',asm)
        note=''
        if t:
            v=int(t.group(1),16); s=resolve(v)
            if s: note='   <<< STRING @0x%x: %r'%(v,s[:110])
            else: note='   [data @0x%x]'%v
        print("  0x%08x  %-42s%s"%(a,asm,note))
if __name__=='__main__':
    a,b=[int(x,16) for x in sys.argv[1:3]]
    annot(a,b,sys.argv[3] if len(sys.argv)>3 else '')
