#!/bin/bash

echo "========================================"
echo "STRESS TEST SESSION 1: Heavy File Exploration"
echo "========================================"

S1=$(curl -s -X POST http://localhost:5555/api/sessions -H "Content-Type: application/json" -d '{}' | jq -r '.id')
echo "Session ID: $S1"

curl -s -X POST "http://localhost:5555/api/sessions/$S1/prompt" \
  -H "Content-Type: application/json" \
  -d '{"message": "Read workflow.rs, escalation.rs, session.rs, and config.rs. For each, list all functions. Then explain how they integrate."}'

sleep 4

echo ""
echo "S1 Escalations:"
curl -s "http://localhost:5555/api/sessions/$S1/escalations" | jq '.' > s1_escalations.json
cat s1_escalations.json

echo ""
echo "S1 Status:"
curl -s "http://localhost:5555/api/sessions/$S1" | jq '.metrics' > s1_metrics.json
cat s1_metrics.json

echo ""
echo "========================================"
echo "STRESS TEST SESSION 2: Deep Code Tracing"
echo "========================================"

S2=$(curl -s -X POST http://localhost:5555/api/sessions -H "Content-Type: application/json" -d '{}' | jq -r '.id')
echo "Session ID: $S2"

curl -s -X POST "http://localhost:5555/api/sessions/$S2/prompt" \
  -H "Content-Type: application/json" \
  -d '{"message": "Trace soft escalation step-by-step: 1) Record tool calls 2) Count patterns 3) Check thresholds 4) Trigger synthesis. Read relevant code files for each step."}'

sleep 4

echo ""
echo "S2 Escalations:"
curl -s "http://localhost:5555/api/sessions/$S2/escalations" | jq '.' > s2_escalations.json
cat s2_escalations.json

echo ""
echo "S2 Status:"
curl -s "http://localhost:5555/api/sessions/$S2" | jq '.metrics' > s2_metrics.json
cat s2_metrics.json

echo ""
echo "========================================"
echo "STRESS TEST SESSION 3: Codebase Search"
echo "========================================"

S3=$(curl -s -X POST http://localhost:5555/api/sessions -H "Content-Type: application/json" -d '{}' | jq -r '.id')
echo "Session ID: $S3"

curl -s -X POST "http://localhost:5555/api/sessions/$S3/prompt" \
  -H "Content-Type: application/json" \
  -d '{"message": "Find all uses of escalation_detector in the codebase. Where is it created? Where is it called? Are there any bugs?"}'

sleep 4

echo ""
echo "S3 Escalations:"
curl -s "http://localhost:5555/api/sessions/$S3/escalations" | jq '.' > s3_escalations.json
cat s3_escalations.json

echo ""
echo "S3 Status:"
curl -s "http://localhost:5555/api/sessions/$S3" | jq '.metrics' > s3_metrics.json
cat s3_metrics.json

