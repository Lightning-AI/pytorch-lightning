import os
import signal

# ps -aef | grep -i 'test.py' | grep -v 'grep' | awk '{ print $2 }'
os.kill(1774059, signal.SIGUSR1)
os.kill(1775127, signal.SIGUSR1)
os.kill(1775383, signal.SIGUSR1)
