from subprocess import Popen


def optimize_dns(enable):
    with open("/etc/resolv.conf") as f:
        lines = f.readlines()

    if (enable and any("127.0.0.53" in line for line in lines)) or (not enable and any("127.0.0.1" in line for line in lines)):
        Popen(f"sudo /home/zeus/miniconda3/envs/cloudspace/bin/python -c 'from lightning.data.processing import _optimize_dns; _optimize_dns({enable})'", shell=True).wait()

def _optimize_dns(enable=False):
    with open("/etc/resolv.conf") as f:
        lines = f.readlines()

    write_lines = []
    for line in lines:
        if "nameserver 127" in line:
            if enable:
                write_lines.append('nameserver 127.0.0.1\n')
            else:
                write_lines.append('nameserver 127.0.0.53\n')
        else:
            write_lines.append(line)

    with open("/etc/resolv.conf", "w") as f:
        for line in write_lines:
            f.write(line)
