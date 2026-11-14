"""Save full ripgrep inventories for changed code and decoded notebook cells."""
from pathlib import Path
import ast,hashlib,json,subprocess,tempfile
BASE=Path(__file__).resolve().parents[1]
PATTERNS={
 'device_and_interrupts':r'\.cuda\(|torch\.cuda|cuda|timeout|SIGALRM|setitimer|terminate\(',
 'input_and_version':r'sha256|load_inputs|verified_audio|verified_teacher_scores|legacy_scores|resolve_path',
 'completion_and_resume':r'complete|last\.pt|best_model|history|os\.replace|identity|runtime',
 'split_and_label_use':r'group|oracle|unlabeled|teacher_train|forbidden|refit|test|validation',
 'old_paths_and_placeholders':r'/mnt/d/|AImodels|KAMPA_feedback_20261114_v1|fresh_teacher_scores|TODO|FIXME|NotImplementedError|GPU를 선택한 뒤',
}
def main():
    source=sorted(p for folder in ['src','tests','tools'] for p in (BASE/folder).rglob('*.py') if '__pycache__' not in p.parts and p!=Path(__file__))
    logs=[];results={}
    with tempfile.TemporaryDirectory(prefix='kampa-notebook-grep-') as d:
        for nb in (BASE/'notebooks').glob('*.ipynb'):
            n=json.loads(nb.read_text());chunks=[]
            for i,cell in enumerate(n['cells']):
                if cell['cell_type']!='code':continue
                text=''.join(cell['source']);tree=ast.parse(text)
                # The complete embedded ZIP is separately checked byte-for-byte.
                lines=text.splitlines()
                for node in reversed(tree.body):
                    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PAYLOAD' for t in node.targets):lines[node.lineno-1:node.end_lineno]=['PAYLOAD = "<verified embedded ZIP>"']
                chunks.append(f'# notebook cell {i}\n'+'\n'.join(lines))
            p=Path(d)/(nb.stem+'.py');p.write_text('\n\n'.join(chunks));source.append(p)
        for category,pattern in PATTERNS.items():
            r=subprocess.run(['rg','--line-number','--no-heading','--color','never',pattern,*map(str,source)],capture_output=True,text=True)
            assert r.returncode in (0,1),(category,r.stderr)
            normalized=r.stdout.replace(str(BASE)+'/', '').replace(d+'/', 'notebook_cells/')
            logs.append(f'[{category}] pattern={pattern}\n'+normalized)
            results[category]=dict(matching_lines=len(normalized.splitlines()),exit_code=r.returncode)
    inventory={p.relative_to(BASE).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in source if BASE in p.parents}
    (BASE/'review/grep_full_matches.txt').write_text('\n\n'.join(logs))
    (BASE/'review/grep_inventory.json').write_text(json.dumps(dict(files=inventory,decoded_notebooks=2,queries=results,embedded_source_verification='tools/verify_package.py'),ensure_ascii=False,indent=2))
    print(json.dumps(results,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
