// Neo4j import script for CSV outputs from build-graph.
// Place graph_nodes.csv and graph_edges.csv into Neo4j import directory.

:param nodesFile => "file:///graph_nodes.csv";
:param edgesFile => "file:///graph_edges.csv";

CREATE CONSTRAINT entity_id IF NOT EXISTS
FOR (n:Entity)
REQUIRE n.id IS UNIQUE;

LOAD CSV WITH HEADERS FROM $nodesFile AS row
MERGE (n:Entity {id: row.id})
SET n.name = row.name,
    n.entity_type = row.entity_type,
    n.source = row.source;

LOAD CSV WITH HEADERS FROM $edgesFile AS row
MATCH (source:Entity {id: row.source})
MATCH (target:Entity {id: row.target})
MERGE (source)-[r:RELATION {relation: row.relation}]->(target)
SET r.essential = toBoolean(row.essential),
    r.evidence_count = toInteger(row.evidence_count),
    r.proficiency_level = row.proficiency_level,
    r.experience_years = CASE row.experience_years WHEN "" THEN null ELSE toFloat(row.experience_years) END;
