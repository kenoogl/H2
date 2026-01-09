#!/bin/bash

# Auto Git Commit & Push Hook for Task Completion
# This script is triggered when a task is marked as completed

set -e  # Exit on any error

# Function to log messages
log_message() {
    local message="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$message"
    # Also log to file if possible
    echo "$message" >> .kiro/hooks/hooks.log 2>/dev/null || true
}

# Function to get current branch name
get_current_branch() {
    git branch --show-current
}

# Function to check if there are changes to commit
has_changes() {
    ! git diff --quiet || ! git diff --cached --quiet || [ -n "$(git ls-files --others --exclude-standard)" ]
}

# Main execution
main() {
    local task_name="${1:-Unknown Task}"
    local task_number="${2:-}"
    local spec_name="${3:-parareal-time-parallelization}"
    
    log_message "=== AUTO-COMMIT HOOK TRIGGERED ==="
    log_message "Task name: $task_name"
    log_message "Task number: $task_number"
    log_message "Spec name: $spec_name"
    log_message "Working directory: $(pwd)"
    log_message "User: $(whoami)"
    log_message "Arguments: $*"
    
    # Check if we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        log_message "Error: Not in a git repository"
        exit 1
    fi
    
    # Check if there are changes to commit
    if ! has_changes; then
        log_message "No changes to commit"
        exit 0
    fi
    
    # Get current branch
    local current_branch
    current_branch=$(get_current_branch)
    log_message "Current branch: $current_branch"
    
    # Stage all changes
    log_message "Staging changes..."
    git add .
    
    # Create commit message
    local commit_message
    if [ -n "$task_number" ]; then
        commit_message="Complete Task $task_number: $task_name

Auto-committed by Kiro hook on task completion
Spec: $spec_name
Branch: $current_branch
Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
    else
        commit_message="Complete task: $task_name

Auto-committed by Kiro hook on task completion
Spec: $spec_name
Branch: $current_branch
Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
    fi
    
    # Commit changes
    log_message "Committing changes..."
    git commit -m "$commit_message"
    
    # Push to remote
    log_message "Pushing to remote..."
    if git push origin "$current_branch"; then
        log_message "Successfully pushed to origin/$current_branch"
    else
        log_message "Warning: Failed to push to remote. Changes are committed locally."
        exit 1
    fi
    
    log_message "Auto-commit completed successfully"
}

# Execute main function with all arguments
main "$@"