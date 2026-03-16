import { useMemo } from 'react';
import hljs from './hljs-core';
import 'highlight.js/styles/github-dark.css';

interface CodeBodyProps {
  code: string;
  language?: string;
  maxHeight?: number;
}

export function CodeBody({ code, language, maxHeight = 400 }: CodeBodyProps) {
  const highlighted = useMemo(() => {
    if (language && hljs.getLanguage(language)) {
      return hljs.highlight(code, { language }).value;
    }
    return hljs.highlightAuto(code).value;
  }, [code, language]);

  return (
    <pre
      style={{
        maxHeight,
        overflowY: 'auto',
        background: '#0d0d0d',
        borderRadius: 4,
        padding: '8px 10px',
        fontSize: 11,
        fontFamily: 'monospace',
        whiteSpace: 'pre',
        overflowX: 'auto',
        margin: 0,
      }}
    >
      <code
        // eslint-disable-next-line react/no-danger
        dangerouslySetInnerHTML={{ __html: highlighted }}
        style={{ color: '#d4d4d4' }}
      />
    </pre>
  );
}
