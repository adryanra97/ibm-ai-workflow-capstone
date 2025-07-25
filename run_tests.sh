#!/bin/bash

# Run all unit tests for the AI Workflow Capstone project
# This script runs API, model, and logging tests
# Author: Adryan R A

set -e  # Exit on any error

echo "======================================"
echo "Running AI Workflow Capstone Tests"
echo "======================================"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_DIR="$SCRIPT_DIR/api"

# Change to API directory
cd "$API_DIR"

echo ""
echo "Setting up test environment..."

# Create test database and directories
mkdir -p tests/data
mkdir -p tests/logs

echo ""
echo "Installing test dependencies..."

# Install test dependencies
pip install pytest pytest-asyncio pytest-cov > /dev/null 2>&1

echo ""
echo "Running unit tests..."

# Set Python path to include current directory
export PYTHONPATH="$API_DIR:$PYTHONPATH"

# Run all tests with coverage
pytest tests/ -v --cov=. --cov-report=html --cov-report=term-missing --cov-fail-under=80

# Store exit code
TEST_EXIT_CODE=$?

echo ""
echo "======================================"

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "All tests passed successfully!"
    echo ""
    echo "Test coverage report generated in htmlcov/"
    echo "View detailed coverage: open htmlcov/index.html"
else
    echo "Some tests failed. Exit code: $TEST_EXIT_CODE"
    exit $TEST_EXIT_CODE
fi

echo "======================================"

echo ""
echo "Test Summary:"
echo "- API Tests: Passed"
echo "- Model Tests: Passed" 
echo "- Logging Tests: Passed"
echo "- Coverage: Above 80%"
echo ""
echo "All unit tests completed successfully!"
echo "Production deployment ready!"
