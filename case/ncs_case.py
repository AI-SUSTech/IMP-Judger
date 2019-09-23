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

from errors import *
from ie import estimate_async, SolutionError

IMAGE_NAME = 'carp_judge'
TMP_DIR = '/tmp/carp_judge'
SANDBOX_TMP_DIR = '/workspace'

if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR, exist_ok=True)

_docker_client = docker.from_env()


def id_generator(size=8, chars=string.ascii_letters + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


class NCSCase:
    def __init__(self, zip_data, cid=0, ctype=msg_types.NCS, dataset=json.loads('{}')):
        self.cid = cid
        self.ctype = ctype
        self._zipdata = zip_data
        self._dataset = dataset
        self._tempdir = os.path.join(TMP_DIR, id_generator())
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
        self.time = config['time']
        self.memory = config['memory']
        self.cpu = config['cpu']
        if 'seed' in config:
            self.seed = config['seed']
        if self.entry == '':
            raise ArchiveError('No entry point')

        if self.entry not in filelist:
            raise ArchiveError('Entry file not found: ' + self.entry)

        with zipfile.open(self.entry) as file:
            try:
                ncs_para = json.loads(file.read())
            except:
                raise Exception("not a json format file")


        self.ncs_para = ncs_para

        self.parameters = self.parameters.replace('$time', str(self.time))
        self.parameters = self.parameters.replace('$cpu', str(self.cpu))
        self.parameters = self.parameters.replace('$memory', str(self.memory))
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
        '''
        todo use docker container
        if self._container is not None:
            raise SandboxError('Container already exists!')
        # Build command
        command = 'python3 {program} {parameters}'.format(
            program=os.path.join(SANDBOX_TMP_DIR, 'program', self.entry),
            parameters=self.parameters
        )
        self._container = _docker_client.containers.run(
            image=IMAGE_NAME,
            command=command,
            name=self.cid,
            auto_remove=False,
            detach=True,
            read_only=True,
            nano_cpus=self.cpu * 1000000000,
            mem_limit=str(self.memory) + 'm',
            memswap_limit=str(int(self.memory * 1.5)) + 'm',
            pids_limit=64,
            network_mode='none',
            stop_signal='SIGKILL',
            volumes={self._tempdir: {'bind': SANDBOX_TMP_DIR, 'mode': 'ro'}},
            working_dir=os.path.join(SANDBOX_TMP_DIR, 'program'),
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

        '''
        #### Test
        # await asyncio.sleep(10)
        from algorithm_ncs import ncs_c

        _lambda = self.ncs_para["lambda"]
        r = self.ncs_para["r"]
        epoch = self.ncs_para["epoch"]
        n= self.ncs_para["n"]
        ncs_para = ncs_c.NCS_CParameter(tmax=300000, lambda_exp=_lambda, r=r, epoch=epoch, N=n)
        p = self._dataset["problem_index"]
        print("************ start problem %d **********" % p)
        ncs_c2 = ncs_c.NCS_C(ncs_para, p)
        self.ncs_res = ncs_c2.loop(quiet=True, seeds=0)

        timedout = False
        _stdout = "parameter: {}".format(self.ncs_para).encode()
        _stderr = b'are you ok'
        statuscode = 0
        #### 

        self._stdout = _stdout
        self._stderr = _stderr
        self._timedout = timedout
        self._statuscode = statuscode

        return timedout, _stdout, _stderr, statuscode



    def close(self):
        try:
            self._container.remove(force=True)
        except:
            pass
        shutil.rmtree(self._tempdir, ignore_errors=True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
