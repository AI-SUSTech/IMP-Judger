import os
import json
import shutil
import random
import string
import docker
import asyncio
import aiohttp
from io import BytesIO
from zipfile import ZipFile
import msg_types
import logging

from errors import *
from ie import estimate_async, SolutionError

IMAGE_NAME = 'cs303/oj_worker:v1'
TMP_DIR = '/tmp/cs303/oj_worker/parameter'
SANDBOX_TMP_DIR = '/parameter'

WORKING_DIR = '/home/zhaoy/ncs_dev/carp_judge_worker/'
SANDBOX_WORKING_DIR = '/workspace'

OLMP_IMAGE_NAME = 'youngwilliam/olmp:gpu_python3.6'
OLMP_WORKING_DIR = '/home/zhaoy/ncs_dev/OLMP'
OLMP_DATA_DIR = '/home/zhaoy/ncs_dev/data/'
DATA_DIR = '/home/data/'


if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR, exist_ok=True)

_docker_client = docker.from_env()


def id_generator(size=8, chars=string.ascii_letters + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


LIST_GPU = [0] * 8

async def allocate_GPU():
    while True:
        for i, state in enumerate(LIST_GPU):
            if state == 0:
                LIST_GPU[i] = 1
                logging.info('Get GPU device: {}'.format(i))
                return i
        logging.info('do not have GPU, wait 60 s')
        await asyncio.sleep(60)


def release_GPU(index):
    if LIST_GPU[index] != 0:
        logging.info('release fail for device: {}, not used yet'.format{index})
    else:
        LIST_GPU[index] = 0
        logging.info('release success for device: {}'.format{index})


class NCSCase:
    def __init__(self, zip_data, cid="id", ctype=msg_types.NCS, dataset=json.loads('{}')):
        self.cid = cid
        self.ctype = ctype
        self._zipdata = zip_data
        self._dataset = dataset
        self._tempdir = id_generator()
        self._container = None
        self._stdout = b''
        self._stderr = b''
        self._timedout = False
        self._statuscode = -1

        self.ncs_para = None
        self.ncs_res = 1e300

    def __enter__(self):
        # Load data
        zipdata = BytesIO(self._zipdata)
        zipfile = ZipFile(zipdata)
        filelist = zipfile.namelist()
        # Load config.json
        if 'config.json' not in filelist:
            raise ArchiveError('No config.json in archive')
        with zipfile.open('config.json') as file:
            config = json.loads(file.read())
        self.entry = config['entry']
        if 'data' in config:
            self.data = config['data']
        else:
            self.data = ''

        self.parameters = config['parameters']
        print(self.parameters)
        self.time = config['time']
        self.memory = config['memory']
        self.cpu = config['cpu']
        if 'seed' in config:
            self.seed = config['seed']
        if self.entry == '':
            raise ArchiveError('No entry point')

        if self.entry not in filelist:
            raise ArchiveError('Entry file not found: ' + self.entry)

        # Prepare sandbox
        progdir = os.path.join(TMP_DIR, self._tempdir)
        if not os.path.exists(progdir):
            os.makedirs(progdir)
        new_path = os.path.join(progdir, self.entry)
        # check dir or not
        if self.entry[-1] == '/':
            os.makedirs(new_path)
        outpath = os.path.join(progdir, self.entry)
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        with open(outpath, 'wb') as outfile:
            with zipfile.open(self.entry) as file:
                data = file.read()
                data.replace(b'\r', b'')
                outfile.write(data)

        
        self.parameters = self.parameters.replace('$data',  str(self._dataset["problem_index"]))
        self.parameters = self.parameters.replace('$configure', os.path.join(SANDBOX_TMP_DIR, self._tempdir, self.entry))
        if 'seed' in config:
            self.parameters = self.parameters.replace('$seed', str(self.seed))
        return self

    async def _wait_container(self):
        with aiohttp.UnixConnector('/var/run/docker.sock') as conn:
            timeout = aiohttp.ClientTimeout(total=self.time)
            async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
                async with session.post('http://localhost/containers/{}/wait'.format(self._container.id)) as resp:
                    try:
                        return False, await resp.json()
                    except asyncio.TimeoutError:
                        return True, None

    async def run(self, stdout=True, stderr=True):
        
        # todo use docker container
        if self._container is not None:
            raise SandboxError('Container already exists!')
        # Build command

        gpu_id = None
        if self._dataset["problem_index"] == 29:
            gpu_id = await allocate_GPU()
            command = 'python3 exp_lenet300100_3.py -g {gpu} {parameters}'.format(
                # program=os.path.join(SANDBOX_TMP_DIR, 'algorithm_ncs','ncs_client.py'),
                gpu = gpu_id,
                parameters=self.parameters
            )
            # command = 'nvidia-smi'
            _volumes = {TMP_DIR: {'bind': SANDBOX_TMP_DIR, 'mode': 'ro'}}
            _IMAGE_NAME = OLMP_IMAGE_NAME
            _runtime="nvidia"
            _working_dir='/opt/caffe'
        else:
            command = 'python3 -m algorithm_ncs.ncs_client {parameters}'.format(
                # program=os.path.join(SANDBOX_TMP_DIR, 'algorithm_ncs','ncs_client.py'),
                parameters=self.parameters
            )
            _volumes = {WORKING_DIR: {'bind': SANDBOX_WORKING_DIR, 'mode': 'ro'},
                     TMP_DIR: {'bind': SANDBOX_TMP_DIR, 'mode': 'ro'}}
            _IMAGE_NAME = IMAGE_NAME
            _runtime=None
            _working_dir=SANDBOX_WORKING_DIR
        self.memory=1024
        self._container = _docker_client.containers.run(
            image=_IMAGE_NAME,
            command=command,
            name=self.cid,
            auto_remove=False,
            detach=True,
            read_only=True,
            nano_cpus=self.cpu * 1000000000,
            mem_limit=str(self.memory) + 'm',
            memswap_limit=str(int(self.memory * 1.5)) + 'm',
            pids_limit=1024,
            network_mode='none',
            stop_signal='SIGKILL',
            volumes=_volumes,
            runtime=_runtime,
            working_dir=_working_dir,
            tmpfs={
                '/tmp': 'rw,size=1g',
                '/run': 'rw,size=1g'
            },
            stdout=stdout,
            stderr=stderr,
            log_config={
                'config': {
                    'mode': 'non-blocking',
                    'max-size': '1m',
                    'max-file': '2'
                }
            }
        )
        timedout, response = await self._wait_container()
        if gpu_id not None:
            release_GPU(gpu_id)

        statuscode = -1
        if timedout:
            try:
                self._container.kill()
            except:
                pass
        else:
            statuscode = response['StatusCode']
        if stdout:
            _stdout = self._container.logs(
                stdout=True,
                stderr=False
            )
        else:
            _stdout = b''
        if stderr:
            _stderr = self._container.logs(
                stdout=False,
                stderr=True
            )
        else:
            _stderr = b''

        # put last line in the first line
        if _stdout != b'':
            lines = _stdout.decode('ascii').splitlines()
            if len(lines) > 0:
                _stdout = (lines[-1] + "\n").encode('ascii') + _stdout

        #### Test
        # await asyncio.sleep(10)
        # from algorithm_ncs import ncs_c

        # _lambda = self.ncs_para["lambda"]
        # r = self.ncs_para["r"]
        # epoch = self.ncs_para["epoch"]
        # n= self.ncs_para["n"]
        # ncs_para = ncs_c.NCS_CParameter(tmax=300000, lambda_exp=_lambda, r=r, epoch=epoch, N=n)
        # p = self._dataset["problem_index"]
        # print("************ start problem %d **********" % p)
        # ncs_c2 = ncs_c.NCS_C(ncs_para, p)
        # self.ncs_res = ncs_c2.loop(quiet=True, seeds=0)

        # timedout = False
        # _stdout = "parameter: {}".format(self.ncs_para).encode()
        # _stderr = b'are you ok'
        # statuscode = 0
        #### 

        self._stdout = _stdout
        self._stderr = _stderr
        self._timedout = timedout
        self._statuscode = statuscode

        return timedout, _stdout, _stderr, statuscode

    async def check_result(self):
        if self._timedout:
            return False, 0., 'Timed out'
        if self._statuscode == 137:
            return False, 0., 'Killed (Out of memory)'
        if self._statuscode != 0:
            return False, 0., 'Exit code is not zero'
        if not self._stdout:
            return False, 0., 'No output'
        stdout = self._stdout.decode('utf8')
        print("DEBUG!!!!", stdout.split('\n'))
        reason = 'success'
        result = float(stdout.split('\n')[0])
        valid = True
        
        return valid, result, reason

    def close(self):
        try:
            self._container.remove(force=True)
        except:
            pass
        shutil.rmtree(os.path.join(TMP_DIR, self._tempdir), ignore_errors=True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
