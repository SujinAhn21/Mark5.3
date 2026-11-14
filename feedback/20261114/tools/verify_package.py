"""Verify reviewed source, standalone notebook payloads and release file hashes."""
from pathlib import Path
import argparse,ast,base64,hashlib,io,json,zipfile
import nbformat
BASE=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def files():return sorted(p for p in BASE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and p.suffix not in ('.pyc','.pyo') and p.name!='release_manifest.json')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--refresh',action='store_true');args=ap.parse_args()
    source={p.relative_to(BASE/'src').as_posix():p.read_bytes() for p in files() if (BASE/'src') in p.parents}
    expected=(BASE/'notebooks/KAMPA_추가실험_실행코드.zip').read_bytes()
    for p in sorted((BASE/'notebooks').glob('*.ipynb')):
        n=nbformat.read(p,as_version=4);nbformat.validate(n);payload=None
        for cell in n.cells:
            if cell.cell_type!='code':continue
            assert not cell.outputs and cell.execution_count is None
            tree=ast.parse(cell.source)
            for node in tree.body:
                if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PAYLOAD' for t in node.targets):payload=base64.b64decode(ast.literal_eval(node.value))
        assert payload==expected,p
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            assert set(archive.namelist())==set(source),p
            for name,data in source.items():assert archive.read(name)==data,(p,name)
    for p in files():
        if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
    manifest=BASE/'release_manifest.json'
    current={p.relative_to(BASE).as_posix():sha(p) for p in files()}
    if args.refresh:manifest.write_text(json.dumps(current,ensure_ascii=False,indent=2)+'\n')
    expected_hashes=json.loads(manifest.read_text());assert expected_hashes==current,'Release inventory/hash mismatch'
    print(f'Verified {len(current)} package files, 2 notebook payloads, {len(source)} bundled source files, Python syntax.')
if __name__=='__main__':main()
