// Example queries for the career-skill KG.
// Adjust names to match your data.

// 1) 职业画像: 查看职业相关的技能/知识/工具/资格
MATCH (o:Entity {entity_type: "OCC", name: "Data Scientist"})-[r:RELATION]->(e:Entity)
RETURN o.name AS occupation, r.relation AS relation, e.name AS entity, e.entity_type AS type, r.evidence_count AS evidence
ORDER BY r.relation, r.evidence_count DESC;

// 2) 技能差距: 用户技能缺失列表
WITH ["Python", "PyTorch"] AS userSkills
MATCH (o:Entity {entity_type: "OCC", name: "Data Scientist"})-[r:RELATION]->(e:Entity)
WHERE r.relation = "REQUIRES_SKILL" AND NOT e.name IN userSkills
RETURN e.name AS missing_skill, r.evidence_count AS evidence
ORDER BY r.evidence_count DESC;

// 3) 岗位推荐: 根据用户技能匹配职业
WITH ["Python", "PyTorch", "A/B Testing"] AS userSkills
MATCH (o:Entity {entity_type: "OCC"})-[r:RELATION]->(e:Entity)
WHERE r.relation = "REQUIRES_SKILL" AND e.name IN userSkills
WITH o, count(e) AS hit_count, sum(r.evidence_count) AS score
RETURN o.name AS occupation, hit_count, score
ORDER BY score DESC, hit_count DESC
LIMIT 5;
