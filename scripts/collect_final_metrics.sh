#!/bin/bash

# Session IDs from previous runs
S1="659b4c21-534c-4f88-87d1-5a05ad5a787e"
S2="139c1eca-39f0-441f-bc5e-9e01d341588e"
S3="942bb7d5-9e96-4f69-94cc-1e09223d98ac"

echo "========================================"
echo "FINAL METRICS - SESSION 1"
echo "========================================"
echo "Status:"
curl -s "http://localhost:5555/api/sessions/$S1" | jq '.status' 

echo ""
echo "Full session info:"
curl -s "http://localhost:5555/api/sessions/$S1" | jq '.' | tee s1_status.json

echo ""
echo "Escalations:"
curl -s "http://localhost:5555/api/sessions/$S1/escalations" | jq '.' | tee s1_escalations.json

echo ""
echo "Metrics:"
curl -s "http://localhost:5555/api/sessions/$S1" | jq '.metrics // "no metrics"'

echo ""
echo "Message count:"
curl -s "http://localhost:5555/api/sessions/$S1/messages" | jq 'length'

echo ""
echo "========================================"
echo "FINAL METRICS - SESSION 2"
echo "========================================"
echo "Status:"
curl -s "http://localhost:5555/api/sessions/$S2" | jq '.status'

echo ""
echo "Full session info:"
curl -s "http://localhost:5555/api/sessions/$S2" | jq '.' | tee s2_status.json

echo ""
echo "Escalations:"
curl -s "http://localhost:5555/api/sessions/$S2/escalations" | jq '.' | tee s2_escalations.json

echo ""
echo "Message count:"
curl -s "http://localhost:5555/api/sessions/$S2/messages" | jq 'length'

echo ""
echo "========================================"
echo "FINAL METRICS - SESSION 3"
echo "========================================"
echo "Status:"
curl -s "http://localhost:5555/api/sessions/$S3" | jq '.status'

echo ""
echo "Full session info:"
curl -s "http://localhost:5555/api/sessions/$S3" | jq '.' | tee s3_status.json

echo ""
echo "Escalations:"
curl -s "http://localhost:5555/api/sessions/$S3/escalations" | jq '.' | tee s3_escalations.json

echo ""
echo "Message count:"
curl -s "http://localhost:5555/api/sessions/$S3/messages" | jq 'length'

