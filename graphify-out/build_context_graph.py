"""Assemble this reviewed context snapshot; run with the Graphify tool interpreter."""
from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from graphify.analyze import god_nodes, suggest_questions, surprising_connections
from graphify.build import build_from_json
from graphify.cache import save_semantic_cache
from graphify.cluster import cluster, score_all
from graphify.diagnostics import diagnose_extraction, format_diagnostic_report
from graphify.export import to_json
from graphify.report import generate

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'graphify-out'
SPEC = 'C:/Users/LENOVO/.codex/skills/graphify/references/extraction-spec.md'


def read(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def stem(path):
    return re.sub(r'[^a-z0-9_]', '_', path.with_suffix('').as_posix().lower())


def main():
    ast = read('.graphify_ast.json')
    detection = read('.graphify_detect.json')
    fragments = sorted(OUT.glob('.graphify_chunk_*.json'))
    if len(fragments) < 5:
        raise SystemExit('Semantic extraction is incomplete: expected 5 or more fragment files.')
    nodes = {n['id']: n for n in ast['nodes']}
    edges = list(ast['edges'])
    hyperedges = []
    semantic = {'nodes': [], 'edges': [], 'hyperedges': [], 'input_tokens': 0, 'output_tokens': 0}
    for p in fragments:
        f = json.loads(p.read_text(encoding='utf-8'))
        for n in f['nodes']:
            if n['file_type'] not in {'code', 'document', 'paper', 'image', 'rationale', 'concept'}:
                raise ValueError(f'Invalid file type: {p}: {n}')
            nodes.setdefault(n['id'], n)
        for e in f['edges']:
            if e['confidence'] == 'EXTRACTED' and e['confidence_score'] != 1:
                raise ValueError(f'Invalid confidence: {p}: {e}')
        edges.extend(f['edges'])
        hyperedges.extend(f.get('hyperedges', []))
        for k in ('nodes', 'edges', 'hyperedges'):
            semantic[k].extend(f.get(k, []))

    # File nodes connect semantic concepts to actual source files and AST symbols.
    # This is explicit provenance, not inferred thematic similarity.
    file_nodes = {}
    def ensure_file(path):
        path = Path(path)
        if not path.is_absolute():
            path = ROOT / path
        path = path.resolve()
        try:
            rel = path.relative_to(ROOT)
        except ValueError:
            return None
        if not path.is_file():
            return None
        key = rel.as_posix()
        if key in file_nodes:
            return file_nodes[key]
        ident = stem(rel)
        if ident not in nodes:
            nodes[ident] = {'id': ident, 'label': rel.name, 'file_type': 'document',
                            'source_file': str(path), 'source_location': 'L1',
                            'metadata': {'kind': 'file'}}
        file_nodes[key] = ident
        return ident

    def edge(a, b, relation, source, line='L1'):
        if a and b and a != b:
            edges.append({'source': a, 'target': b, 'relation': relation,
                          'confidence': 'EXTRACTED', 'confidence_score': 1.0,
                          'source_file': str(source), 'source_location': line, 'weight': 1.0})

    for node in list(nodes.values()):
        source = node.get('source_file')
        if source:
            fid = ensure_file(source)
            if node.get('_origin') != 'ast':
                edge(fid, node['id'], 'contains', source, node.get('source_location') or 'L1')

    docs = {Path(f) for f in detection['files']['document']}
    docs.update(OUT.glob('*CONTEXT.md'))
    docs.add(ROOT / 'paper/main.tex')
    explicit_paths = re.compile(r'`((?:src/|scripts/|configs/|docs/|outputs/|tests/|paper/)[^`\n]+)`')
    markdown_links = re.compile(r'\[[^\]]*\]\(([^)]+)\)')
    for doc in sorted(docs):
        if not doc.exists():
            continue
        fid = ensure_file(doc)
        for line, text in enumerate(doc.read_text(encoding='utf-8', errors='replace').splitlines(), 1):
            paths = [(m, ROOT) for m in explicit_paths.findall(text)]
            paths += [(m, doc.parent) for m in markdown_links.findall(text)]
            for raw, base in paths:
                raw = raw.split('#', 1)[0]
                raw = re.sub(r':\d+(?:[–-]\d+)?$', '', raw)
                if not raw or '://' in raw or '{' in raw:
                    continue
                edge(fid, ensure_file(base / raw), 'references', doc, f'L{line}')

    # Remove exact repeat records only. Keep different relationship types/locations
    # between the same endpoints in the evidence export, even if the simple graph folds them.
    seen = set()
    unique_edges = []
    for e in edges:
        e.setdefault('confidence_score', {'EXTRACTED': 1.0, 'INFERRED': .85, 'AMBIGUOUS': .2}.get(e.get('confidence'), .2))
        key = tuple(str(e.get(k, '')) for k in ('source', 'target', 'relation', 'source_file', 'source_location', 'context'))
        if key not in seen:
            unique_edges.append(e)
            seen.add(key)
    extraction = {'nodes': list(nodes.values()), 'edges': unique_edges, 'hyperedges': hyperedges,
                  'input_tokens': 0, 'output_tokens': 0,
                  'token_usage_note': 'Host-session token accounting unavailable; zeros are placeholders, not measured free usage.'}
    write('.graphify_semantic.json', semantic)
    write('.graphify_extract.json', extraction)
    write('graph-evidence.json', extraction)
    allowed = sorted({n.get('source_file') for n in semantic['nodes'] if n.get('source_file')})
    cached = save_semantic_cache(semantic['nodes'], semantic['edges'], semantic['hyperedges'],
                                 root=str(ROOT), allowed_source_files=allowed, prompt_file=SPEC)
    print('Cached semantic files:', cached)
    diagnostic = diagnose_extraction(extraction, directed=False, root=str(ROOT))
    write('graph-health.json', diagnostic)
    (OUT / 'GRAPH_HEALTH.md').write_text('# Graph integrity diagnostics\n\n```text\n' +
        format_diagnostic_report(diagnostic) + '\n```\n\nThe AST extractor can emit unresolved references. '
        'The visualization uses a simple undirected graph, which folds relationships sharing endpoints. '
        '`graph-evidence.json` preserves the extracted directed relationship records, confidence, and provenance. '
        'Graph reachability alone is not proof of an implementation dependency.\n', encoding='utf-8')
    G = build_from_json(extraction, root=str(ROOT), directed=False)
    if not G:
        raise SystemExit('Empty graph')
    communities = cluster(G)
    cohesion = score_all(G, communities)
    gods = god_nodes(G)
    surprises = surprising_connections(G, communities)
    labels = {cid: f'Community {cid}' for cid in communities}
    questions = suggest_questions(G, communities, labels)
    if not to_json(G, communities, str(OUT / 'graph.json')):
        raise SystemExit('Graph shrink guard refused export')
    report = generate(G, communities, cohesion, labels, gods, surprises, detection,
                      {'input': 0, 'output': 0}, str(ROOT), suggested_questions=questions)
    report += '\n\n## Coverage and interpretation\n\nRead `PROJECT_CONTEXT.md` for the reconciled current research state. '
    report += 'Read `GRAPH_HEALTH.md` for extraction warnings. The evidence export retains relationships folded by the visualization. '
    report += 'Token usage for host-agent extraction is unavailable; displayed zeros are placeholders, not a measured total.\n'
    (OUT / 'GRAPH_REPORT.md').write_text(report, encoding='utf-8')
    write('.graphify_analysis.json', {'communities': {str(k): v for k, v in communities.items()},
          'cohesion': {str(k): v for k, v in cohesion.items()}, 'gods': gods, 'surprises': surprises,
          'questions': questions})
    summaries = []
    for cid, members in communities.items():
        member_nodes = [G.nodes[n] for n in members]
        common = Counter(n.get('source_file', '') for n in member_nodes).most_common(5)
        ranked = sorted(members, key=lambda n: G.degree(n), reverse=True)[:8]
        summaries.append({'id': cid, 'size': len(members), 'files': common,
                          'labels': [G.nodes[n].get('label', n) for n in ranked]})
    write('community-review.json', summaries)
    write('build-summary.json', {'created_at': datetime.now(timezone.utc).isoformat(),
          'nodes': G.number_of_nodes(), 'edges': G.number_of_edges(), 'communities': len(communities),
          'raw_nodes': len(nodes), 'raw_edges': len(unique_edges), 'semantic_fragments': len(fragments),
          'detected_files': detection['total_files'], 'detected_words': detection['total_words'],
          'token_usage': None, 'external_api_calls_for_semantic_extraction': 0})
    print(json.dumps(read('build-summary.json')))


if __name__ == '__main__':
    main()
