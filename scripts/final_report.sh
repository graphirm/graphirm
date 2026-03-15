#!/bin/bash

echo "========================================"
echo "STRESS TEST REPORT SUMMARY"
echo "========================================"
echo ""

# Session IDs
S1="659b4c21-534c-4f88-87d1-5a05ad5a787e"
S2="139c1eca-39f0-441f-bc5e-9e01d341588e"
S3="942bb7d5-9e96-4f69-94cc-1e09223d98ac"

echo "SESSION 1: Heavy File Exploration"
echo "=================================="
echo "Session ID: $S1"
echo ""
echo "Status:"
curl -s "http://localhost:5555/api/sessions/$S1" | jq '{id: .id, status: .status, created_at: .created_at}'
echo ""
echo "Message Count:"
MSGS1=$(curl -s "http://localhost:5555/api/sessions/$S1/messages" | jq 'length')
echo "$MSGS1 messages"
echo ""
echo "Tool Calls Executed:"
curl -s "http://localhost:5555/api/sessions/$S1/messages" | jq '[.[] | select(.node_type.type == "ToolCall") | .node_type.tool] | group_by(.) | map({tool: .[0], count: length})'
echo ""
echo "Escalations Recorded:"
curl -s "http://localhost:5555/api/sessions/$S1/escalations" | jq 'length'
echo ""
echo "---"
echo ""

echo "SESSION 2: Deep Code Tracing"
echo "============================"
echo "Session ID: $S2"
echo ""
echo "Status:"
curl -s "http://localhost:5555/api/sessions/$S2" | jq '{id: .id, status: .status, created_at: .created_at}'
echo ""
echo "Message Count:"
MSGS2=$(curl -s "http://localhost:5555/api/sessions/$S2/messages" | jq 'length')
echo "$MSGS2 messages"
echo ""
echo "Tool Calls Executed:"
curl -s "http://localhost:5555/api/sessions/$S2/messages" | jq '[.[] | select(.node_type.type == "ToolCall") | .node_type.tool] | group_by(.) | map({tool: .[0], count: length})'
echo ""
echo "Escalations Recorded:"
curl -s "http://localhost:5555/api/sessions/$S2/escalations" | jq 'length'
echo ""
echo "---"
echo ""

echo "SESSION 3: Codebase Search"
echo "=========================="
echo "Session ID: $S3"
echo ""
echo "Status:"
curl -s "http://localhost:5555/api/sessions/$S3" | jq '{id: .id, status: .status, created_at: .created_at}'
echo ""
echo "Message Count:"
MSGS3=$(curl -s "http://localhost:5555/api/sessions/$S3/messages" | jq 'length')
echo "$MSGS3 messages"
echo ""
echo "Tool Calls Executed:"
curl -s "http://localhost:5555/api/sessions/$S3/messages" | jq '[.[] | select(.node_type.type == "ToolCall") | .node_type.tool] | group_by(.) | map({tool: .[0], count: length})'
echo ""
echo "Escalations Recorded:"
curl -s "http://localhost:5555/api/sessions/$S3/escalations" | jq 'length'
echo ""
echo "========================================"
echo "AGGREGATE METRICS"
echo "========================================"
echo ""
echo "Total Sessions: 3"
echo "Total Messages: $((MSGS1 + MSGS2 + MSGS3))"
echo "Average Messages per Session: $(((MSGS1 + MSGS2 + MSGS3) / 3))"
echo ""

# Count total tool calls
TOOLS1=$(curl -s "http://localhost:5555/api/sessions/$S1/messages" | jq '[.[] | select(.node_type.type == "ToolCall")] | length')
TOOLS2=$(curl -s "http://localhost:5555/api/sessions/$S2/messages" | jq '[.[] | select(.node_type.type == "ToolCall")] | length')
TOOLS3=$(curl -s "http://localhost:5555/api/sessions/$S3/messages" | jq '[.[] | select(.node_type.type == "ToolCall")] | length')

echo "Total Tool Calls: $((TOOLS1 + TOOLS2 + TOOLS3))"
echo "  S1: $TOOLS1 tool calls"
echo "  S2: $TOOLS2 tool calls"
echo "  S3: $TOOLS3 tool calls"
echo ""

# Count total escalations
ESC1=$(curl -s "http://localhost:5555/api/sessions/$S1/escalations" | jq 'length')
ESC2=$(curl -s "http://localhost:5555/api/sessions/$S2/escalations" | jq 'length')
ESC3=$(curl -s "http://localhost:5555/api/sessions/$S3/escalations" | jq 'length')

echo "Total Escalations: $((ESC1 + ESC2 + ESC3))"
echo "  S1: $ESC1"
echo "  S2: $ESC2"
echo "  S3: $ESC3"
echo ""

echo "Issues Found:"
echo "- Database connection pool timeouts after turn 3-4"
echo "- Escalations endpoint not recording data (pool issues)"
echo "- Recommend increasing pool size or connection timeout"

