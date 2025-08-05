#!/bin/bash

# Script to evaluate prompts from evaluation.jsonl using the brandonbot API
# Make sure the API server is running on localhost:8000

clear

echo "Evaluating prompts from evaluation.jsonl..."
echo "=========================================="

# Check if evaluation.jsonl exists
if [ ! -f "tmp/evaluation.jsonl" ]; then
    echo "Error: tmp/evaluation.jsonl not found!"
    exit 1
fi

# Create output directory for responses
mkdir -p tmp/responses

# Initialize counter
counter=1

# Loop through each line in the JSONL file
while IFS= read -r line; do
    # Skip empty lines
    if [ -z "$line" ]; then
        continue
    fi

    # Extract the prompt using jq
    prompt=$(echo "$line" | jq -r '.prompt')
    id=$(echo "$line" | jq -r '.id')

    # Skip if prompt is null or empty
    if [ "$prompt" = "null" ] || [ -z "$prompt" ]; then
        echo "Skipping line $counter: No prompt found"
        ((counter++))
        continue
    fi

    echo "Processing prompt $counter (ID: $id)..."
    echo "Prompt: ${prompt:0:100}..."

    # Submit the prompt to the API
    response=$(curl -X POST "http://localhost:8000/agent" \
        -H "Content-Type: application/json" \
        -d "{
            \"messages\": [
                {
                    \"role\": \"user\",
                    \"content\": $(echo "$prompt" | jq -R .)
                }
            ]
        }" \
        -s)

    # Check if the request was successful
    if [ $? -eq 0 ]; then
        # Extract the response content
        response_content=$(echo "$response" | jq -r '.choices[0].message.content // "No content found"')

        # Save the full response to a file
        echo "$response" > "tmp/responses/response_${id}.json"

        # Save just the content to a text file
        echo "$response_content" > "tmp/responses/content_${id}.txt"

        echo "✓ Response saved for ID $id"
    else
        echo "✗ Failed to get response for ID $id"
        echo "Error response: $response" > "tmp/responses/error_${id}.txt"
    fi

    echo "----------------------------------------"

    # Increment counter
    ((counter++))

    # Optional: Add a small delay to avoid overwhelming the server
    sleep 1

done < "tmp/evaluation.jsonl"

echo "=========================================="
echo "Evaluation completed!"
echo "Responses saved in tmp/responses/"
echo "Total prompts processed: $((counter - 1))"
