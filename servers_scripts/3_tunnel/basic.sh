

autossh -M 0 -N -T -R 20000:localhost:22 ec2-user@13.212.111.39 -i wang_bin_KP.pem -o ServerAliveInterval=30 -o ServerAliveCountMax=3


autossh -M 0 -N -T -o ServerAliveInterval=60 -o ServerAliveCountMax=999999 -R 20000:localhost:22 ec2-user@13.212.111.39




autossh -M 0 -N -T -R 8439:localhost:22 wangbin@13.229.126.191