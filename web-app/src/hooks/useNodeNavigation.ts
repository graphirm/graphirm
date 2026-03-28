import { useState, useEffect, useRef } from "react";
import type { Node, Edge } from "@xyflow/react";

const NAV_EDGE_TYPES = new Set(["produces", "responds_to", "contains"]);

export function useNodeNavigation(nodes: Node[], edges: Edge[]) {
  const [focusedNodeId, setFocusedNodeId] = useState<string | null>(null);
  const nodesRef = useRef(nodes);
  const edgesRef = useRef(edges);
  nodesRef.current = nodes;
  edgesRef.current = edges;

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      const target = e.target as HTMLElement;
      if (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;

      const arrows = ["ArrowRight", "ArrowLeft", "ArrowUp", "ArrowDown"];
      if (e.key === "Escape") { setFocusedNodeId(null); return; }
      if (!arrows.includes(e.key)) return;
      e.preventDefault();

      setFocusedNodeId(current => {
        const nodes = nodesRef.current;
        const edges = edgesRef.current;
        const navigable = nodes.filter(n => n.type !== "group" && n.type !== "annotation" && !n.hidden);

        if (!current) return navigable[0]?.id ?? null;

        const nodeMap = new Map(nodes.map(n => [n.id, n]));
        const getEdgeType = (e: Edge) => (e.data as { edge_type?: string } | undefined)?.edge_type ?? "";

        if (e.key === "ArrowRight") {
          const targets = edges
            .filter(ed => ed.source === current && NAV_EDGE_TYPES.has(getEdgeType(ed)))
            .map(ed => nodeMap.get(ed.target))
            .filter((n): n is Node => !!n && n.type !== "group" && n.type !== "annotation" && !n.hidden);
          return targets[0]?.id ?? current;
        }
        if (e.key === "ArrowLeft") {
          const sources = edges
            .filter(ed => ed.target === current && NAV_EDGE_TYPES.has(getEdgeType(ed)))
            .map(ed => nodeMap.get(ed.source))
            .filter((n): n is Node => !!n && n.type !== "group" && n.type !== "annotation" && !n.hidden);
          return sources[0]?.id ?? current;
        }
        if (e.key === "ArrowUp" || e.key === "ArrowDown") {
          const currentNode = nodeMap.get(current);
          const parentId = currentNode?.parentId ?? null;
          const siblings = navigable
            .filter(n => (n.parentId ?? null) === parentId && n.id !== current)
            .sort((a, b) => (a.position.y ?? 0) - (b.position.y ?? 0));
          if (!siblings.length) return current;
          const allWithCurrent = [...siblings];
          const idx = allWithCurrent.findIndex(n => n.id === current);
          const adjustedIdx = idx === -1 ? 0 : idx;
          if (e.key === "ArrowUp") return allWithCurrent[Math.max(0, adjustedIdx - 1)]?.id ?? current;
          return allWithCurrent[Math.min(allWithCurrent.length - 1, adjustedIdx + 1)]?.id ?? current;
        }
        return current;
      });
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []); // empty dep array - uses refs

  return { focusedNodeId };
}
