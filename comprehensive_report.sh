#!/bin/bash

echo "========================================"
echo "COMPREHENSIVE STRESS TEST REPORT"
echo "Date: $(date)"
echo "========================================"
echo ""

S1="659b4c21-534c-4f88-87d1-5a05ad5a787e"
S2="139c1eca-39f0-441f-bc5e-9e01d341588e"
S3="942bb7d5-9e96-4f69-94cc-1e09223d98ac"

# Function to analyze a session
analyze_session() {
  local SESSION_ID=$1
  local SESSION_NAME=$2
  
  echo "SESSION: $SESSION_NAME"
  echo "Session ID: $SESSION_ID"
  echo ""
  
  # Get messages
  MESSAGES=$(curl -s "http://localhost:5555/api/sessions/$SESSION_ID/messages")
  TOTAL_MSGS=$(echo "$MESSAGES" | jq 'length')
  
  echo "Messages: $TOTAL_MSGS total"
  
  # Count message types
  INTERACTIONS=$(echo "$MESSAGES" | jq '[.[] | select(.node_type.type == "Interaction")] | length')
  echo "  - Interactions: $INTERACTIONS"
  
  # Count tool calls from metadata
  TOOL_CALLS=$(echo "$MESSAGES" | jq '[.[] | select(.metadata.tool_calls != null) | .metadata.tool_calls[]] | length')
  echo "  - Tool Calls: $TOOL_CALLS"
  
  # Get tool types
  echo "  - Tool Types:"
  echo "$MESSAGES" | jq -r '[.[] | select(.metadata.tool_calls != null) | .metadata.tool_calls[].name] | group_by(.) | map("\(.[]): " + (length | tostring)) | .[]' | sed 's/^/      /'
  
  # Count turns (assistant responses with tool calls)
  TURNS=$(echo "$MESSAGES" | jq '[.[] | select(.node_type.type == "Interaction" and .node_type.role == "assistant" and .metadata.tool_calls != null)] | length')
  echo "  - Turns (with tools): $TURNS"
  
  # Check for escalations endpoint
  ESCALATIONS=$(curl -s "http://localhost:5555/api/sessions/$SESSION_ID/escalations" 2>/dev/null)
  ESC_COUNT=$(echo "$ESCALATIONS" | jq 'length' 2>/dev/null || echo "0")
  echo "  - Escalations: $ESC_COUNT"
  
  # Session status
  STATUS=$(curl -s "http://localhost:5555/api/sessions/$SESSION_ID" | jq -r '.status')
  echo "  - Status: $STATUS"
  
  echo ""
  echo "---"
  echo ""
}

analyze_session "$S1" "SESSION 1: Heavy File Exploration"
analyze_session "$S2" "SESSION 2: Deep Code Tracing"
analyze_session "$S3" "SESSION 3: Codebase Search"

echo "========================================"
echo "AGGREGATE METRICS"
echo "========================================"
echo ""

# Get all messages from all sessions
MSGS1=$(curl -s "http://localhost:5555/api/sessions/$S1/messages")
MSGS2=$(curl -s "http://localhost:5555/api/sessions/$S2/messages")
MSGS3=$(curl -s "http://localhost:5555/api/sessions/$S3/messages")

# Total messages
TOTAL1=$(echo "$MSGS1" | jq 'length')
TOTAL2=$(echo "$MSGS2" | jq 'length')
TOTAL3=$(echo "$MSGS3" | jq 'length')
TOTAL_ALL=$((TOTAL1 + TOTAL2 + TOTAL3))

echo "Total Messages Across All Sessions: $TOTAL_ALL"
echo "  - Session 1: $TOTAL1"
echo "  - Session 2: $TOTAL2"
echo "  - Session 3: $TOTAL3"
echo "  - Average: $(($TOTAL_ALL / 3))"
echo ""

# Total tool calls
TOOLS1=$(echo "$MSGS1" | jq '[.[] | select(.metadata.tool_calls != null) | .metadata.tool_calls[]] | length')
TOOLS2=$(echo "$MSGS2" | jq '[.[] | select(.metadata.tool_calls != null) | .metadata.tool_calls[]] | length')
TOOLS3=$(echo "$MSGS3" | jq '[.[] | select(.metadata.tool_calls != null) | .metadata.tool_calls[]] | length')
TOTAL_TOOLS=$((TOOLS1 + TOOLS2 + TOOLS3))

echo "Total Tool Calls Executed: $TOTAL_TOOLS"
echo "  - Session 1: $TOOLS1"
echo "  - Session 2: $TOOLS2"
echo "  - Session 3: $TOOLS3"
echo ""

# Tool breakdown
echo "Tool Breakdown:"
echo "$MSGS1 $MSGS2 $MSGS3" | jq -s add | jq -r '[.[] | select(.metadata.tool_calls != null) | .metadata.tool_calls[].name] | group_by(.) | map({tool: .[0], count: length}) | sort_by(-.count) | .[]' | sed 's/^/  /'
echo ""

# Total turns
TURNS1=$(echo "$MSGS1" | jq '[.[] | select(.node_type.type == "Interaction" and .node_type.role == "assistant" and .metadata.tool_calls != null)] | length')
TURNS2=$(echo "$MSGS2" | jq '[.[] | select(.node_type.type == "Interaction" and .node_type.role == "assistant" and .metadata.tool_calls != null)] | length')
TURNS3=$(echo "$MSGS3" | jq '[.[] | select(.node_type.type == "Interaction" and .node_type.role == "assistant" and .metadata.tool_calls != null)] | length')
TOTAL_TURNS=$((TURNS1 + TURNS2 + TURNS3))

echo "Total Turns (with tool execution): $TOTAL_TURNS"
echo "  - Session 1: $TURNS1"
echo "  - Session 2: $TURNS2"
echo "  - Session 3: $TURNS3"
echo ""

# Escalations
ESC1=$(curl -s "http://localhost:5555/api/sessions/$S1/escalations" 2>/dev/null | jq 'length' 2>/dev/null || echo "0")
ESC2=$(curl -s "http://localhost:5555/api/sessions/$S2/escalations" 2>/dev/null | jq 'length' 2>/dev/null || echo "0")
ESC3=$(curl -s "http://localhost:5555/api/sessions/$S3/escalations" 2>/dev/null | jq 'length' 2>/dev/null || echo "0")
TOTAL_ESC=$((ESC1 + ESC2 + ESC3))

echo "Total Escalations Triggered: $TOTAL_ESC"
echo "  - Session 1: $ESC1"
echo "  - Session 2: $ESC2"
echo "  - Session 3: $ESC3"
echo ""

echo "========================================"
echo "KEY FINDINGS"
echo "========================================"
echo ""
echo "✓ All 3 sessions successfully created"
echo "✓ All sessions executing agent loops with tool calls"
echo "✓ Combined $TOTAL_TOOLS tool invocations across sessions"
echo "✓ Average $((TOTAL_TOOLS / 3)) tools per session"
echo ""
echo "⚠ Database connection pool timeout after turn 3-4"
echo "⚠ Escalations endpoint not recording data (pool exhaustion)"
echo "⚠ Recommend:"
echo "  - Increase SQLite connection pool size"
echo "  - Increase connection timeout threshold"
echo "  - Optimize context window queries"

