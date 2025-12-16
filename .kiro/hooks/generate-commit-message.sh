#!/bin/bash

# Generate detailed commit message for task completion
# Usage: ./generate-commit-message.sh <task_number> <task_name> [spec_name]

set -e

# Default values
SPEC_NAME="${3:-parareal-time-parallelization}"
TASK_NUMBER="$1"
TASK_NAME="$2"

# Function to get task details from tasks.md
get_task_details() {
    local task_num="$1"
    local tasks_file=".kiro/specs/$SPEC_NAME/tasks.md"
    
    if [ -f "$tasks_file" ]; then
        # Extract task details using grep and sed
        grep -A 5 "^- \[.\] $task_num\." "$tasks_file" | head -10
    else
        echo "Task details not found"
    fi
}

# Function to get changed files
get_changed_files() {
    git diff --cached --name-only | head -10
}

# Function to get test results
get_test_status() {
    local test_files
    test_files=$(find test/ -name "*.jl" -type f 2>/dev/null | head -5)
    
    if [ -n "$test_files" ]; then
        echo "Test files updated:"
        echo "$test_files" | sed 's/^/  - /'
    fi
}

# Generate commit message
generate_message() {
    cat << EOF
Complete Task $TASK_NUMBER: $TASK_NAME

$(get_task_details "$TASK_NUMBER")

Changed files:
$(get_changed_files | sed 's/^/  - /')

$(get_test_status)

Auto-committed by Kiro hook
Spec: $SPEC_NAME
Branch: $(git branch --show-current)
Timestamp: $(date '+%Y-%m-%d %H:%M:%S')
EOF
}

# Main execution
if [ -z "$TASK_NUMBER" ] || [ -z "$TASK_NAME" ]; then
    echo "Usage: $0 <task_number> <task_name> [spec_name]"
    echo "Example: $0 '2.1' 'Create hybrid parallelization coordinator'"
    exit 1
fi

generate_message