#!/bin/bash
# RunPod passes the account's SSH public key in PUBLIC_KEY; put it where sshd reads it, start sshd, then stay up.
set -e
if [ -n "$PUBLIC_KEY" ]; then
    echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
fi
ssh-keygen -A >/dev/null 2>&1
/usr/sbin/sshd
echo "co3 image ready: $(python -c 'import torch;print(torch.__version__, torch.cuda.is_available())')"
sleep infinity
