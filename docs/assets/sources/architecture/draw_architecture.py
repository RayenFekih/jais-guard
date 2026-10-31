"""Editable vector architecture figure. Python 3.12; matplotlib 3.10.8.
Run: python draw_architecture.py [--metrics ../upload/jaisguard_readme_metrics.json]
"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle

P=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--metrics',type=Path,default=P/'jaisguard_readme_metrics.json')
a=parser.parse_args(); spec=json.loads(a.metrics.read_text())['architecture']
assert spec['transformer_layers']==28 and spec['hidden_size']==1536
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none'})
fig=plt.figure(figsize=(14,11),facecolor='white'); ax=fig.add_axes([0,0,1,1]); ax.set(xlim=(0,1400),ylim=(1100,0));ax.axis('off')
INK='#292929'; BLUE='#6489AA'; CORAL='#D77B5D'; GRAY='#DCD9D0'; LIGHT='#F5F4F0'; PALE='#F9EBE5'; BLUELIGHT='#EBF0F5'
def text(x,y,s,size=13,color=INK,weight='normal',ha='center'):
    ax.text(x,y,s,fontsize=size,color=color,weight=weight,ha=ha,va='center',linespacing=1.5)
def rect(x,y,w,h,fill='white',stroke='#9A9992',lw=1):
    ax.add_patch(Rectangle((x,y),w,h,facecolor=fill,edgecolor=stroke,linewidth=lw))
def box(x,y,w,h,s,fill=LIGHT,stroke='#9A9992',size=13):
    rect(x,y,w,h,fill,stroke);text(x+w/2,y+h/2,s,size)
def arrow(points,color=INK,dash=False):
    for i in range(len(points)-1):
        start,end=points[i:i+2]
        if i==len(points)-2:
            ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=11,linewidth=1.1,color=color,linestyle='--' if dash else '-'))
        else:
            ax.plot([start[0],end[0]],[start[1],end[1]],color=color,lw=1.1,linestyle='--' if dash else '-')
def plus(x,y):
    ax.add_patch(Circle((x,y),10,facecolor='white',edgecolor=BLUE,lw=1));text(x,y,'+',12)
text(700,48,'JaisGuard',30,weight='medium')
text(700,91,'Adapting a causal language model for Arabic gibberish detection',17,color='#50504C')
text(65,145,'01  SHARED BACKBONE',13,BLUE,'bold',ha='left')
text(760,145,'02  REPLACE THE GENERATIVE OUTPUT PATH',13,CORAL,'bold',ha='left')
# Input through repeated, expanded decoder layers.
box(65,180,220,55,'Arabic input',size=15)
box(65,265,220,65,'Tokenizer\nToken IDs + attention mask',size=10.5)
arrow([(175,235),(175,265)])
box(65,360,220,60,'Token embeddings',BLUELIGHT,BLUE,14)
arrow([(175,330),(175,360)])
arrow([(285,390),(325,390)])
# repeated layer boundaries and internal computation
rect(325,180,365,447,'white',BLUE)
text(507,205,'Jais 3 0.9B',17,weight='bold')
text(507,233,'28 decoder layers · d = 1,536',12,color='#555550')
box(347,256,321,35,'Layer 1: attention + MLP',BLUELIGHT,BLUE,12)
arrow([(507,291),(507,318)],BLUE)
text(507,309,'⋮',17,BLUE)
rect(347,327,321,230,'#FAFBFC',BLUE)
text(507,346,'Decoder layer 28 (expanded)',12,BLUE,'bold')
box(396,369,233,31,'RMSNorm',size=11)
box(396,412,233,39,'Causal self-attention',BLUELIGHT,BLUE,12)
arrow([(512,400),(512,412)],BLUE)
arrow([(512,451),(512,468)],BLUE);plus(512,479)
arrow([(512,358),(369,358),(369,479),(501,479)],BLUE)
box(396,497,233,31,'RMSNorm → gated MLP',size=11)
arrow([(512,489),(512,497)],BLUE)
arrow([(512,528),(512,538)],BLUE);plus(512,548)
arrow([(512,489),(650,489),(650,548),(523,548)],BLUE)
box(396,580,233,30,'Final RMSNorm',size=11)
arrow([(512,558),(512,580)],BLUE)
arrow([(512,610),(512,648)],BLUE)
box(325,650,365,64,'Final hidden representations\nH ∈ ℝᵀ × ¹⁵³⁶',BLUELIGHT,BLUE,14)
# Original path, visually de-emphasized.
arrow([(690,682),(725,682),(725,240),(760,240)],'#929087',True)
text(1030,181,'Original Jais output',14,color='#6D6B65',weight='bold')
box(760,216,170,49,'LM head',LIGHT,size=14)
box(963,216,340,49,'Vocabulary logits',LIGHT,size=14)
arrow([(930,240),(963,240)],'#929087')
arrow([(1133,265),(1133,294)],'#929087')
box(963,295,340,60,'Autoregressive next-token\ngeneration',LIGHT,size=13)
text(1030,386,'LM head omitted from JaisGuard',12,color='#737169')
# Classifier branch.
text(1030,447,'JaisGuard: direct classification',18,CORAL,'bold')
arrow([(690,682),(737,682),(737,514),(760,514)],CORAL)
box(760,483,543,62,'Select final non-padding token\nt* = last position with attention_mask = 1','white',CORAL,13)
arrow([(1031,545),(1031,574)],CORAL)
box(760,575,543,52,'h = H[t*]  ∈ ℝ¹⁵³⁶','white',CORAL,15)
arrow([(1031,627),(1031,655)],CORAL)
box(760,656,543,58,'New FP32 linear head: 1,536 → 2',PALE,CORAL,16)
text(1080,736,'3,074 parameters = 1,536 × 2 + 2 biases',10,color='#6D5145')
arrow([(855,714),(855,765)],CORAL)
box(760,765,190,47,'Softmax',size=14)
box(989,765,314,47,'VALID / GIBBERISH',PALE,CORAL,14)
arrow([(950,788),(989,788)],CORAL)
text(1032,838,'Class probabilities; label uses a decision threshold',11,color='#66645E')
# Two training variants, aligned as scientific comparison rows.
text(65,758,'03  TRAINING VARIANTS',13,BLUE,'bold',ha='left')
text(65,794,'Same classifier architecture; different trainable weights',12,color='#66645E',ha='left')
ax.plot([65,1335],[875,875],color='#DCDCD5',lw=1)
text(65,914,'Frozen + Head',16,BLUE,'bold',ha='left')
box(330,889,299,48,'Original backbone frozen',BLUELIGHT,BLUE,12)
arrow([(629,913),(673,913)],BLUE)
box(674,889,280,48,'Trainable FP32 head',PALE,CORAL,12)
text(985,914,'3,074 trainable parameters',12,ha='left')
text(65,983,'LoRA + Head',16,CORAL,'bold',ha='left')
box(330,959,299,48,'Original backbone frozen',BLUELIGHT,BLUE,12)
arrow([(629,983),(673,983)],BLUE)
box(674,959,280,48,'Trainable adapters + head',PALE,CORAL,12)
text(985,978,'Rank 8 in q_proj + v_proj',12,ha='left')
text(985,1001,'Every decoder layer (28)',11,ha='left')
rect(65,1049,16,12,BLUELIGHT,BLUE);text(91,1055,'Frozen weights',11,ha='left')
rect(265,1049,16,12,PALE,CORAL);text(291,1055,'Trainable weights',11,ha='left')
text(1335,1055,'Decoder detail simplified; sequence length T ≤ 64 tokens',11,color='#66645E',ha='right')
fig.savefig(P/'jaisguard_architecture.svg')
fig.savefig(P/'jaisguard_architecture.png',dpi=300)
