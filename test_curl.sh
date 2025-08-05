#!/bin/bash

# Make sure the API server is running on localhost:8000
clear

echo "Testing API with cURL..."
echo "=================================="

# Test 1: Health check
echo -e "\n1. Testing health endpoint..."
curl -X GET "http://localhost:8000/health" \
  -H "Content-Type: application/json" \
  -s | jq

# Test 2: Basic chat completion
echo -e "\n2. Testing basic chat completion..."
curl -X POST "http://localhost:8000/agent" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Hello! What can you do?"
      }
    ],
    "model": "llm"
  }' \
  -s | jq -r '.choices[0].message.content'

# Test 3: Simple question
echo -e "\n3. Testing advanced chat completion..."
curl -X POST "http://localhost:8000/agent" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "From 2020 to 2050, how many elderly people will there be in Japan? What is their consumption potential across various aspects such as clothing, food, housing, and transportation? Based on population projections, elderly consumer willingness, and potential changes in their consumption habits, please produce a market size analysis report for the elderly demographic."
      }
    ]
  }' \
  -s | jq -r '.choices[0].message.content'

echo -e "\n=================================="
echo "Test completed!"
