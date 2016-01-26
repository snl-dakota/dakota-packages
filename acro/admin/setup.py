
import os
import sys
import subprocess
import signal
import glob

admin_dir = os.path.dirname(os.path.abspath(__file__))
packages=[('admin','acro-admin'), ('tpl','cxxtest','python'), ('packages','*','python')]

def signal_handler(signum, frame):
    pid=os.getpid()
    pgid=os.getpgid(pid)
    if pgid == -1:
        sys.stderr.write("  ERROR: invalid pid %s\n" % (pid,))
        sys.exit(1)
    os.killpg(pgid,signal.SIGTERM)
    sys.exit(1)

signal.signal(signal.SIGINT,signal_handler)
if sys.platform[0:3] != "win":
    signal.signal(signal.SIGHUP, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def run():
    trunk = (len(sys.argv) > 1 and '--trunk' in sys.argv)
    if trunk:
        sys.argv.remove('--trunk')
    rc=0
    if not os.path.exists('python'):
        cmd = [ sys.executable,
            os.path.join('bin','pyomo_install'),
            #'--logfile',os.path.join('tpl','python.log'),
            '--config', os.path.join(admin_dir,'vpy','dev.ini'), '--venv', 'python' ]
        if trunk:
            sys.stdout.write("Installing Python from trunk\n")
            cmd.append('--trunk')
        else:
            sys.stdout.write("Installing Python from cached packages\n")
            cmd.extend(['--zip', os.path.join(admin_dir,'vpy','python.zip')])

        print cmd
        sys.stdout.flush()
        rc = subprocess.call(cmd)
    if rc != 0:
        sys.exit(rc)
    rc = subprocess.call([os.path.join('.','bootstrap','bootstrap'), 'all'])
    if rc != 0:
        sys.exit(rc)
    dir_ = os.getcwd()
    abs_python = os.path.abspath( os.path.join('.', 'python', 'bin', 'python') )
    for package in packages:
        tmp_ = package
        for pkg in glob.glob( os.path.join(*tmp_) ):
            os.chdir(pkg)
            rc = subprocess.call([ abs_python, 'setup.py', 'develop'])
    os.chdir(dir_)
    if len(sys.argv) > 1:
        rc = subprocess.call([
                os.path.join('.','python','bin','python'), 
                os.path.join('python','bin','driver') ] + sys.argv[1:])
    sys.exit(rc)
    
