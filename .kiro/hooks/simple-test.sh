#!/bin/bash

# Very simple test hook
echo "HOOK EXECUTED AT $(date)" >> /tmp/kiro-hook-test.log
echo "PWD: $(pwd)" >> /tmp/kiro-hook-test.log
echo "ARGS: $*" >> /tmp/kiro-hook-test.log
echo "---" >> /tmp/kiro-hook-test.log

exit 0