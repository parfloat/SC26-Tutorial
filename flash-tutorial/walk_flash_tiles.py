"""Numbers for the deck's tile-by-tile slides: the 28-line flash loop of
pdf-papers/flash-attention.png (kv_block_size=2) on query F of A5881F, block 2 head 1,
checked against plain softmax attention.  Run from flash-tutorial/: python3 walk_flash_tiles.py"""
import torch, math, sys
sys.path.insert(0,'.')
from tinygpt import load, encode, rope, EOS
m=load()
ids=torch.tensor([encode('5881')+[EOS]]).cuda()
cap={}
for i,b in enumerate(m.blocks): b.ln1.register_forward_hook(lambda mod,inp,out,i=i: cap.__setitem__(i,out))
with torch.no_grad():
    m(ids)
    bi,h=2,1
    x=cap[bi]; q,k,v=m.blocks[bi].qkv(x).view(1,6,3,4,32).permute(2,0,3,1,4)
    q,k=rope(q,0),rope(k,0); q,k,v=[t.half().double()[0,h].cpu() for t in (q,k,v)]
    qF=q[5:6]; scale=1/math.sqrt(32)
    M=torch.full((1,1),-math.inf,dtype=torch.double); L=torch.zeros(1,1,dtype=torch.double); U=torch.zeros(1,32,dtype=torch.double)
    f=lambda t: ' '.join('%.4g'%x for x in t.flatten().tolist())
    for t,st in enumerate(range(0,6,2)):
        kt,vt=k[st:st+2],v[st:st+2]
        raw=qF@kt.T; s=raw*scale
        nm=torch.maximum(M,s.max(-1,keepdim=True).values)
        a=torch.exp(M-nm); E=torch.exp(s-nm)
        L=a*L+E.sum(-1,keepdim=True); U=a*U+E@vt; M=nm
        print(f'tile{t}: raw {f(raw)} | s {f(s)} | newmax {f(nm)} | alpha {f(a)} | E {f(E)} | L {f(L)} | U {f(U[0,:3])} .. {f(U[0,31:])}')
    O=U/L; print('O',f(O[0,:3]),f(O[0,31:]))
    s=(qF@k.T)*scale; P=s.softmax(-1); print('P',f(P)); Oref=P@v
    print('Oref',f(Oref[0,:3]),f(Oref[0,31:]),'maxdiff',(O-Oref).abs().max().item())
    print('v rows'); 
    for j in range(6): print(j,f(v[j,:3]),f(v[j,31:]))
