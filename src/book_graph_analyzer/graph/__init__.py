"""Graph database interface."""

from book_graph_analyzer.graph.connection import get_driver, check_neo4j_connection

__all__ = ["get_driver", "check_neo4j_connection"]
