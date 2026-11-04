"""Shared file readers and manifests; inference rows retain reads and length only."""
from pathlib import Path
import gzip,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
def open_text(path,mode='rt'):
    encoding='utf-8-sig' if 'r' in mode else 'utf8'
    return gzip.open(path,mode,encoding=encoding) if str(path).endswith('.gz') else open(path,mode,encoding=encoding)
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def read_json(path):return json.loads(Path(path).read_text(encoding='utf8'))
def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf8')
def known_length(value):
    if isinstance(value,bool) or not isinstance(value,int) or not 3<=value<=4096:
        raise ValueError('Known target_length must be an integer in 3..4096')
    return value
def iter_binned(handle):
    """Yield (reference, reads) blocks, including clusters with no reads."""
    block=[]
    for line in handle:
        line=line.strip()
        if line:
            block.append(line)
        elif block:
            if len(block)<2 or set(block[1])!={'*'}:
                raise ValueError('Expected reference / asterisks / optional reads, separated by blank lines')
            yield block[0],block[2:]
            block=[]
    if block:
        if len(block)<2 or set(block[1])!={'*'}:
            raise ValueError('Expected reference / asterisks / optional reads, separated by blank lines')
        yield block[0],block[2:]

def load_records(path,length=None,training=False):
    """Load inputs; training=True also retains the reference for scoring."""
    path=Path(path);name=path.name[:-3] if path.name.endswith('.gz') else path.name
    rows=[]
    with open_text(path) as h:
        if name.endswith('.jsonl'):
            for line in h:
                if not line.strip():continue
                item=json.loads(line);L=length if length is not None else item.get('target_length')
                if L is None:raise ValueError('Supply --length or target_length in every JSONL record')
                row=dict(id=item.get('id',len(rows)),reads=item['reads'],target_length=known_length(L))
                if training:row['reference']=item['reference']
                rows.append(row)
        else:
            if length is None:raise ValueError('Binned TXT requires externally supplied --length')
            for reference,reads in iter_binned(h):
                row=dict(id=len(rows),reads=reads,target_length=known_length(length))
                if training:row['reference']=reference
                rows.append(row)
    if not rows:raise ValueError('Input contains no clusters')
    if training:
        for r in rows:
            if len(r['reference'])!=r['target_length'] or set(r['reference'])-set('ACGT'):
                raise ValueError('Evaluation reference must be ACGT and match target_length')
    return rows
