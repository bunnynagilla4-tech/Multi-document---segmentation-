"""Unit tests for similarity and event clustering in NewsLens."""

import unittest
from src.similarity import ArticleSimilarity
from src.event_clustering import EventClusterer


class TestClustering(unittest.TestCase):
    def setUp(self):
        self.texts = [
            "Heavy rainfall and floods caused traffic disruptions across Mumbai on Monday.",
            "Floodwaters submerged railway lines and roads in suburban Mumbai after heavy rain.",
            "Novatech unveiled a 1000-qubit superconducting quantum processor with error mitigation.",
            "The new quantum computing chip Hyperion marks a breakthrough in error suppression.",
            "India defeated Australia by two wickets to win the cricket world championship final.",
            "Rohit Sharma hit 92 runs as India clinched a thrilling cricket title over Australia."
        ]
        self.titles = [
            "Mumbai Rain Disruptions",
            "Floods Submerge Mumbai Rails",
            "Novatech 1000-Qubit Chip",
            "Quantum Processor Breakthrough",
            "India Wins Cricket Trophy",
            "Rohit Leads Cricket Win"
        ]

    def test_similarity_computation(self):
        """Test similarity matrix symmetry and top match pairing."""
        sim = ArticleSimilarity()
        matrix = sim.fit_transform(self.texts)
        self.assertEqual(matrix.shape, (6, 6))

        # Check that article 0 (rain) is most similar to article 1 (rain)
        top_matches_0 = sim.get_most_similar(0, top_k=1, titles=self.titles)
        self.assertEqual(top_matches_0[0]["article_index"], 1)

        # Check distance matrix properties
        dist = sim.get_distance_matrix()
        self.assertEqual(dist[0, 0], 0.0)
        self.assertTrue((dist >= 0.0).all() and (dist <= 1.0).all())

    def test_event_clustering(self):
        """Test agglomerative clustering groups articles into corresponding events."""
        clusterer = EventClusterer(distance_threshold=0.85)
        labels = clusterer.fit_predict(self.texts, titles=self.titles)

        self.assertEqual(len(labels), 6)
        # Articles 0 and 1 should share a cluster (Rain)
        self.assertEqual(labels[0], labels[1])
        # Articles 2 and 3 should share a cluster (Quantum)
        self.assertEqual(labels[2], labels[3])
        # Articles 4 and 5 should share a cluster (Cricket)
        self.assertEqual(labels[4], labels[5])
        # Rain and Quantum should NOT share the same cluster
        self.assertNotEqual(labels[0], labels[2])

        summary = clusterer.get_cluster_summary()
        self.assertEqual(len(summary), 3)


if __name__ == "__main__":
    unittest.main()
