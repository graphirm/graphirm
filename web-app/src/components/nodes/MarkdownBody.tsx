import { useMemo } from 'react';
import { marked } from 'marked';
import hljs from './hljs-core';
import 'highlight.js/styles/github-dark.css';

// Configure marked once — renderer applies hljs to fenced code blocks.
marked.setOptions({
  // @ts-expect-error — marked types don't expose highlight callback in v15 options
  highlight: (code: string, lang: string) => {
    if (lang && hljs.getLanguage(lang)) {
      return hljs.highlight(code, { language: lang }).value;
    }
    return hljs.highlightAuto(code).value;
  },
  gfm: true,
  breaks: true,
});

interface MarkdownBodyProps {
  content: string;
  maxHeight?: number;
}

export function MarkdownBody({ content, maxHeight = 400 }: MarkdownBodyProps) {
  const html = useMemo(() => {
    try {
      return marked.parse(content) as string;
    } catch {
      return `<pre>${content}</pre>`;
    }
  }, [content]);

  return (
    <div
      className="markdown-body"
      style={{
        maxHeight,
        overflowY: 'auto',
        fontSize: 12,
        lineHeight: 1.6,
        color: 'var(--fg)',
      }}
      // eslint-disable-next-line react/no-danger
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
