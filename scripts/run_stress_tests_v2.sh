#!/bin/bash

echo "========================================"
echo "STRESS TEST SESSION 1: Heavy File Exploration"
echo "========================================"

S1=$(curl -s -X POST http://localhost:5555/api/sessions -H "Content-Type: application/json" -d '{}' | jq -r '.id')
echo "Session ID: $S1"

echo "Submitting prompt..."
curl -s -X POST "http://localhost:5555/api/sessions/$S1/prompt" \
  -H "Content-Type: application/json" \
  -d '{"content": "Read workflow.rs, escalation.rs, session.rs, and config.rs. For each, list all functions. Then explain how they integrate."}'

echo ""
echo "Waiting 10 seconds for agent loop..."
sleep 10

echo ""
echo "S1 Full Status:"
curl -s "http://localhost:5555/api/sessions/$S1" | jq '.' | tee s1_status.json

echo ""
echo "S1 Escalations:"
curl -s "http://localhost:5555/api/sessions/$S1/escalations" | jq '.' | tee s1_escalations.json

echo ""
echo "S1 Messages:"
curl -s "http://localhost:5555/api/sessions/$S1/messages" | jq '.' | head -50

echo ""
echo "========================================"
echo "STRESS TEST SESSION 2: Deep Code Tracing"
echo "========================================"

S2=$(curl -s -X POST http://localhost:5555/api/sessions -H "Content-Type: application/json" -d '{}' | jq -r '.id')
echo "Session ID: $S2"

echo "Submitting prompt..."
curl -s -X POST "http://localhost:5555/api/sessions/$S2/prompt" \
  -H "Content-Type: application/json" \
  -d '{"content": "Trace soft escalation step-by-step: 1) Record tool calls 2) Count patterns 3) Check thresholds 4) Trigger synthesis. Read relevant code files for each step."}'

echo ""
echo "Waiting 10 seconds for agent loop..."
sleep 10

echo ""
echo "S2 Full Status:"
curl -s "http://localhost:5555/api/sessions/$S2" | jq '.' | tee s2_status.json

echo ""
echo "S2 Escalations:"
curl -s "http://localhost:5555/api/sessions/$S2/escalations" | jq '.' | tee s2_escalations.json

echo ""
echo "========================================"
echo "STRESS TEST SESSION 3: Codebase Search"
echo "========================================"

S3=$(curl -s -X POST http://localhost:5555/api/sessions -H "Content-Type: application/json" -d '{}' | jq -r '.id')
echo "Session ID: $S3"

echo "Submitting prompt..."
curl -s -X POST "http://localhost:5555/api/sessions/$S3/prompt" \
  -H "Content-Type: application/json" \
  -d '{"content": "Find all uses of escalation_detector in the codebase. Where is it created? Where is it called? Are there any bugs?"}'

echo ""
echo "Waiting 10 seconds for agent loop..."
sleep 10

echo ""
echo "S3 Full Status:"
curl -s "http://localhost:5555/api/sessions/$S3" | jq '.' | tee s3_status.json

echo ""
echo "S3 Escalations:"
curl -s "http://localhost:5555/api/sessions/$S3/escalations" | jq '.' | tee s3_escalations.json

