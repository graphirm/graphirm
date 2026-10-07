import { useCallback, useEffect, useRef, useState, type FormEvent } from 'react';
import { api } from '../api/client';
import type { GraphNode } from '../types/graph';
import styles from './RulesTab.module.css';

const PIN_LIMIT = 200;

function errorText(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

function knowledgeFields(node: GraphNode): { entity: string; summary: string } | null {
  const nt = node.node_type;
  if (nt.type !== 'Knowledge') return null;
  return { entity: nt.entity, summary: nt.summary };
}

export function RulesTab() {
  const [items, setItems] = useState<GraphNode[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [name, setName] = useState('');
  const [note, setNote] = useState('');
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draft, setDraft] = useState('');
  const [busy, setBusy] = useState(false);
  const loadGen = useRef(0);

  const loadPinned = useCallback(async () => {
    const gen = ++loadGen.current;
    try {
      const nodes = await api.listPinnedKnowledge(PIN_LIMIT);
      if (gen !== loadGen.current) return;
      setItems(nodes);
      setLoaded(true);
      setError(null);
    } catch (err: unknown) {
      if (gen !== loadGen.current) return;
      setError(errorText(err));
    }
  }, []);

  useEffect(() => {
    void loadPinned();
    return () => {
      loadGen.current += 1;
    };
  }, [loadPinned]);

  async function onCreate(e: FormEvent) {
    e.preventDefault();
    const entity = name.trim();
    const summary = note.trim();
    if (!entity || !summary || busy) return;
    loadGen.current += 1;
    setError(null);
    setBusy(true);
    try {
      await api.createKnowledge({
        entity,
        entity_type: 'rule',
        summary,
        pinned: true,
      });
      setName('');
      setNote('');
      await loadPinned();
    } catch (err) {
      setError(errorText(err));
    } finally {
      setBusy(false);
    }
  }

  function startEdit(node: GraphNode, summary: string) {
    setEditingId(node.id);
    setDraft(summary);
    setError(null);
  }

  async function saveEdit(id: string) {
    const summary = draft.trim();
    if (!summary || busy) return;
    loadGen.current += 1;
    setError(null);
    setBusy(true);
    try {
      await api.patchKnowledge(id, { summary });
      setEditingId(null);
      await loadPinned();
    } catch (err) {
      setError(errorText(err));
    } finally {
      setBusy(false);
    }
  }

  async function remove(id: string) {
    if (busy) return;
    loadGen.current += 1;
    setError(null);
    setBusy(true);
    try {
      await api.deleteKnowledge(id);
      if (editingId === id) setEditingId(null);
      await loadPinned();
    } catch (err) {
      setError(errorText(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className={styles.root}>
      <form className={styles.form} onSubmit={(e) => void onCreate(e)}>
        <label className={styles.field}>
          Name
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            autoComplete="off"
          />
        </label>
        <label className={styles.field}>
          Note
          <textarea value={note} onChange={(e) => setNote(e.target.value)} />
        </label>
        <button type="submit" disabled={busy || !name.trim() || !note.trim()}>
          Pin rule
        </button>
      </form>
      {error && <p className={styles.error}>{error}</p>}
      {loaded && items.length === 0 ? (
        <div className={styles.empty}>
          <p className={styles.emptyTitle}>No pinned rules.</p>
          <p className={styles.emptyBody}>A pinned rule stays in the briefing for every session.</p>
        </div>
      ) : null}
      {items.length === PIN_LIMIT ? (
        <p className={styles.truncated}>
          Showing 200 pinned rules. Newer pins are not in this list.
        </p>
      ) : null}
      <ul className={styles.list}>
        {items.map((node) => {
          const fields = knowledgeFields(node);
          const summary = fields?.summary ?? '';
          const editing = editingId === node.id;
          return (
            <li key={node.id} className={styles.row}>
              {fields && <div className={styles.name}>{fields.entity}</div>}
              {editing ? (
                <div className={styles.edit}>
                  <textarea
                    value={draft}
                    onChange={(e) => setDraft(e.target.value)}
                    aria-label="Summary"
                  />
                  <div className={styles.actions}>
                    <button type="button" disabled={busy || !draft.trim()} onClick={() => void saveEdit(node.id)}>
                      Save
                    </button>
                    <button
                      type="button"
                      className="secondary"
                      disabled={busy}
                      onClick={() => setEditingId(null)}
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              ) : (
                <>
                  <p className={styles.summary}>{summary}</p>
                  <div className={styles.actions}>
                    <button type="button" className="secondary" disabled={busy} onClick={() => startEdit(node, summary)}>
                      Edit
                    </button>
                    <button type="button" className="danger" disabled={busy} onClick={() => void remove(node.id)}>
                      Delete
                    </button>
                  </div>
                </>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
