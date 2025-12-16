#!/bin/bash

# Simple test hook to verify hook system is working

echo "[$(date '+%Y-%m-%d %H:%M:%S')] TEST HOOK EXECUTED" >> .kiro/hooks/test-hook.log
echo "Arguments: $*" >> .kiro/hooks/test-hook.log
echo "Working directory: $(pwd)" >> .kiro/hooks/test-hook.log
echo "---" >> .kiro/hooks/test-hook.log

exit 0