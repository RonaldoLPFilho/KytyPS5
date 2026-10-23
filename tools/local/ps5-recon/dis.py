import subprocess,sys,os
SRC='/home/chen-xiao/nvme1n1/game/PPSA01341-app0/decrypted/eboot.bin'
data=open(SRC,'rb').read()
def va2fo(v): return v+0x4000
def dis(v,spread=64,back=48):
    start=max(0,va2fo(v)-back)
    end=min(len(data),va2fo(v)+spread)
    blob=data[start:end]
    open('/tmp/ps5recon/w.bin','wb').write(blob)
    startva=start-0x4000
    out=subprocess.run(['objdump','-D','-b','binary','-m','i386:x86-64','-M','intel',
        '--adjust-vma=0x%x'%startva,'/tmp/ps5recon/w.bin'],capture_output=True,text=True).stdout
    return out
if __name__=='__main__':
    for spec in sys.argv[1:]:
        name,v=spec.split('=')
        v=int(v,16)
        print("#"*100); print("### %s   code site va=0x%x"%(name,v)); print("#"*100)
        print(dis(v))
