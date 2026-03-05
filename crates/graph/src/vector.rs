//! HNSW vector index for knowledge node embeddings.

use instant_distance::{Builder, HnswMap, Search};

use crate::NodeId;

pub struct VectorIndex {
    dimension: usize,
    pending: Vec<(NodeId, Vec<f32>)>,
    map: Option<HnswMap<Point, NodeId>>,
}

#[derive(Clone)]
struct Point(Vec<f32>);

impl instant_distance::Point for Point {
    fn distance(&self, other: &Self) -> f32 {
        self.0
            .iter()
            .zip(other.0.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

impl VectorIndex {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            pending: Vec::new(),
            map: None,
        }
    }

    pub fn from_pairs(dimension: usize, pairs: Vec<(NodeId, Vec<f32>)>) -> Self {
        let mut index = Self::new(dimension);
        for (id, vec) in pairs {
            index.insert(id, vec);
        }
        index.rebuild();
        index
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn len(&self) -> usize {
        self.pending.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    pub fn insert(&mut self, id: NodeId, embedding: Vec<f32>) {
        debug_assert_eq!(
            embedding.len(),
            self.dimension,
            "embedding dimension mismatch: expected {}, got {}",
            self.dimension,
            embedding.len()
        );
        self.pending.push((id, embedding));
        self.map = None;
    }

    pub fn rebuild(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        let points: Vec<Point> = self.pending.iter().map(|(_, v)| Point(v.clone())).collect();
        let values: Vec<NodeId> = self.pending.iter().map(|(id, _)| id.clone()).collect();
        self.map = Some(Builder::default().build(points, values));
    }

    /// Search for the `k` nearest neighbours to `query`.
    ///
    /// Returns `Vec<(NodeId, distance)>` sorted by ascending distance.
    /// Returns an empty vec if the index hasn't been built yet.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(NodeId, f32)> {
        let map = match &self.map {
            Some(m) => m,
            None => return Vec::new(),
        };

        let query_point = Point(query.to_vec());
        let mut search = Search::default();
        let mut results: Vec<(NodeId, f32)> = map
            .search(&query_point, &mut search)
            .take(k)
            .map(|item| (item.value.clone(), item.distance))
            .collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_index_new() {
        let index = VectorIndex::new(128);
        assert_eq!(index.dimension(), 128);
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn test_vector_index_insert_and_search() {
        let mut index = VectorIndex::new(3);

        let id1 = NodeId("node-1".to_string());
        let id2 = NodeId("node-2".to_string());
        let id3 = NodeId("node-3".to_string());

        index.insert(id1.clone(), vec![1.0, 0.0, 0.0]);
        index.insert(id2.clone(), vec![0.0, 1.0, 0.0]);
        index.insert(id3.clone(), vec![0.9, 0.1, 0.0]);

        // Must rebuild after inserts
        index.rebuild();

        // Query close to id1 and id3
        let results = index.search(&[0.95, 0.05, 0.0], 2);
        assert_eq!(results.len(), 2);

        // id1 and id3 are equidistant from (0.95, 0.05, 0.0) — just verify both appear
        let ids: Vec<&NodeId> = results.iter().map(|(id, _)| id).collect();
        assert!(ids.contains(&&id1), "id1 should be in top-2 results");
        assert!(ids.contains(&&id3), "id3 should be in top-2 results");
        assert!(!ids.contains(&&id2), "id2 should not be in top-2 results");
    }

    #[test]
    fn test_vector_index_search_k_larger_than_size() {
        let mut index = VectorIndex::new(2);
        index.insert(NodeId("only-one".to_string()), vec![1.0, 0.0]);
        index.rebuild();

        let results = index.search(&[1.0, 0.0], 10);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_vector_index_empty_search() {
        let index = VectorIndex::new(4);
        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_vector_index_many_vectors() {
        let mut index = VectorIndex::new(8);

        for i in 0..100 {
            let mut vec = vec![0.0f32; 8];
            vec[i % 8] = 1.0;
            vec[(i + 1) % 8] = 0.5;
            index.insert(NodeId(format!("node-{}", i)), vec);
        }
        index.rebuild();

        let results = index.search(&[1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 5);
        assert_eq!(results.len(), 5);

        for (_, dist) in &results {
            assert!(*dist >= 0.0);
        }
    }

    #[test]
    fn test_vector_index_rebuild_from_pairs() {
        let pairs: Vec<(NodeId, Vec<f32>)> = vec![
            (NodeId("a".into()), vec![1.0, 0.0, 0.0]),
            (NodeId("b".into()), vec![0.0, 1.0, 0.0]),
            (NodeId("c".into()), vec![0.0, 0.0, 1.0]),
        ];

        let index = VectorIndex::from_pairs(3, pairs);
        assert_eq!(index.len(), 3);

        let results = index.search(&[1.0, 0.0, 0.0], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, NodeId("a".into()));
    }
}
