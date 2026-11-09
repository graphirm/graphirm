import assert from 'node:assert/strict';
import { test } from 'node:test';
import { looksLikeHtmlReply, parseHtmlPieces } from './pieceStream.ts';
import { piecePartsForInteraction } from '../utils/chatSegments.ts';
import type { GraphData, GraphEdge, GraphNode } from '../types/graph';

const PAGE = `<nav id="index">
<ul>
<li><a href="#cause">1.0.0 [statement] Why the test fails</a></li>
<li><a href="#fix">2.0.0 [code] The fix in stats.py</a></li>
<li><a href="#ran">3.0.0 [result] Test run after the fix</a></li>
</ul>
</nav>
<section id="cause" class="statement">
<h2>1.0.0 Why the test fails</h2>
<p data-n="1.0.1" data-src="file:stats.py">1.0.1 median returns the upper middle value.</p>
<p data-n="1.0.2" data-src="file:stats.py">1.0.2 The list has 4 items &amp; no sort.</p>
</section>
<section id="fix" class="code">
<h2>2.0.0 The fix in stats.py</h2>
<pre data-n="2.0.1"><code>if n &lt; 2:
    return xs[0]</code></pre>
</section>
<section id="ran" class="result">
<h2>3.0.0 Test run after the fix</h2>
<p data-n="3.0.1" data-src="tool:bash">3.0.1 Ran pytest; 2 of 2 tests passed.</p>
</section>
`;

test('an HTML index page is recognised; markdown and JSON are not', () => {
  assert.equal(looksLikeHtmlReply(PAGE), true);
  assert.equal(looksLikeHtmlReply('  <section class="statement">'), true);
  assert.equal(looksLikeHtmlReply('Use `<section>` tags.'), false);
  assert.equal(looksLikeHtmlReply('## Heading\n\nText'), false);
  assert.equal(looksLikeHtmlReply('{"segments":[]}'), false);
});

test('a complete page becomes one block per section with numbered lines', () => {
  const parts = parseHtmlPieces(PAGE, true);
  assert.deepEqual(parts, [
    {
      type: 'statement',
      content: '',
      n: 1,
      title: 'Why the test fails',
      state: 'done',
      items: ['median returns the upper middle value.', 'The list has 4 items & no sort.'],
    },
    {
      type: 'code',
      content: 'if n < 2:\n    return xs[0]',
      n: 2,
      title: 'The fix in stats.py',
      state: 'done',
    },
    {
      type: 'result',
      content: '',
      n: 3,
      title: 'Test run after the fix',
      state: 'done',
      items: ['Ran pytest; 2 of 2 tests passed.'],
    },
  ]);
});

test('a streamed prefix shows closed parts done and the open one streaming, with no tags', () => {
  const cut = PAGE.indexOf('1.0.2 The list');
  const parts = parseHtmlPieces(PAGE.slice(0, cut + '1.0.2 The list'.length), false);
  assert.equal(parts.length, 1);
  assert.equal(parts[0].state, 'streaming');
  assert.deepEqual(parts[0].items, ['median returns the upper middle value.', 'The list']);

  // Cut inside a tag: the half tag is not shown.
  const midTag = PAGE.indexOf('data-n="3.0.1"') + 4;
  const later = parseHtmlPieces(PAGE.slice(0, midTag), false);
  assert.deepEqual(later.map(p => p.state), ['done', 'done', 'streaming']);
  for (const part of later) {
    for (const text of [part.content, part.title ?? '', ...(part.items ?? [])]) {
      assert.doesNotMatch(text, /<\/?[a-z]|data-n/i);
    }
  }
});

test('the index alone yields no parts yet', () => {
  assert.deepEqual(parseHtmlPieces(PAGE.slice(0, PAGE.indexOf('<section')), false), []);
});

function content(id: string, contentType: string, body: string, metadata: Record<string, unknown>): GraphNode {
  return {
    id,
    node_type: { type: 'Content', content_type: contentType, body },
    created_at: '2026-11-09T00:00:00Z',
    metadata,
  } as GraphNode;
}

function contains(source: string, target: string): GraphEdge {
  return { id: `${source}->${target}`, source, target, edge_type: 'contains', metadata: {} } as GraphEdge;
}

test('stored parts read back as blocks in order, lines by position', () => {
  const graph: GraphData = {
    nodes: [
      content('p2', 'reply_part', 'Next', { order: 2, kind: 'steps', heading: 'Next' }),
      content('p1', 'reply_part', '', { order: 1, kind: 'statement', heading: null }),
      content('l1', 'reply_line', 'Only line.', { position: 1 }),
      content('l3', 'reply_line', '2.0.2 Run it.', { position: 2 }),
      content('l2', 'reply_line', '2.0.1 Build it.', { position: 1 }),
    ],
    edges: [contains('m', 'p2'), contains('m', 'p1'), contains('p1', 'l1'), contains('p2', 'l3'), contains('p2', 'l2')],
  };
  assert.deepEqual(piecePartsForInteraction('m', graph), [
    { type: 'statement', content: 'Only line.', n: 1, title: undefined },
    { type: 'steps', content: '', n: 2, title: 'Next', items: ['Build it.', 'Run it.'] },
  ]);
  assert.equal(piecePartsForInteraction('other', graph), undefined);
});
