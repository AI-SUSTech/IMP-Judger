import json
import logging
import coloredlogs
import asyncio
import aiohttp
import base64
import websockets
import config
import traceback
import time
from msg_types import *
from case import carp_case, ncs_case
from errors import *

coloredlogs.install(level=config.log_level)

send_queue = asyncio.Queue()
receive_queue = asyncio.Queue()
judge_queue = asyncio.Queue()

uid = None


async def __message_handler():
    while True:
        message = await receive_queue.get()
        try:
            obj = json.loads(message)
            type = obj['type']
            if type == CASE_DATA:
                await judge_queue.put(obj['payload'])
            elif type == WORKER_TICK:
                obj = {'type': WORKER_TICK}
                await send_queue.put(json.dumps(obj))
            elif type == WORKER_INFO:
                obj = {'uid': uid, 'type': WORKER_INFO, 'maxTasks': config.parallel_judge_tasks}
                await send_queue.put(json.dumps(obj))
        except Exception as e:
            logging.error(e)


async def __judge_worker(idx):
    while True:
        obj = await judge_queue.get()
        cid = ''
        try:
            cid = obj['cid']
            data = base64.b64decode(obj['data'])
            ctype = obj['type']
            dataset = obj['dataset']
            logging.info('Enter judge for id: ' + cid)
            with CARPCase(data, cid, ctype, dataset) as case:
                logging.info('[{}]({}) Start judge'.format(idx, cid))
                obj = {
                    'type': CASE_START,
                    'cid': cid,
                    'timestamp': time.time()
                }
                await send_queue.put(json.dumps(obj))
                timedout, stdout, stderr, exitcode = await case.run(stdout=True, stderr=True)
                logging.info('[{}]({}) Judge finished: {}, {}'.format(idx, cid, timedout, exitcode))
                stdout_overflow = False
                stderr_overflow = False
                stdout = stdout.decode('utf8')
                stderr = stderr.decode('utf8')
                finish_time = time.time()
                if len(stdout) > config.log_limit_bytes:
                    stdout = stdout[-config.log_limit_bytes:]
                    stdout_overflow = True
                if len(stderr) > config.log_limit_bytes:
                    stderr = stderr[-config.log_limit_bytes:]
                    stderr_overflow = True
                if ctype == IMP:
                    valid, influence, reason = await case.check_imp_result()
                else:
                    valid = False
                    influence = 0.
                    reason = ''
                ret = {
                    'cid': cid,
                    'type': CASE_RESULT,
                    'timedout': timedout,
                    'stdout': stdout,
                    'stdout_overflow': stdout_overflow,
                    'stderr': stderr,
                    'stderr_overflow': stderr_overflow,
                    'exitcode': exitcode,
                    'timestamp': finish_time,
                    'valid': valid,
                    'influence': influence,
                    'reason': reason
                }
                await send_queue.put(json.dumps(ret))
        except ArchiveError as e:
            logging.error('[{}] {}'.format(idx, e))
        except Exception as e:
            logging.error('[{}] {}'.format(idx, e))
            traceback.print_exc()
            ret = {
                'cid': cid,
                'type': CASE_ERROR,
                'message': str(e)
            }
            await send_queue.put(json.dumps(ret))


async def __message_dispatcher(ws):
    while True:
        message = await send_queue.get()
        await ws.send(message)


async def __tick_sender(ws):
    obj = {'type': WORKER_TICK, 'uid': uid}
    data = json.dumps(obj)
    while True:
        await asyncio.sleep(60)
        await send_queue.put(data)


async def __message_receiver(ws):
    async for message in ws:
        await receive_queue.put(message)


async def main():
    global uid
    while True:
        try:
            uid = None
            # Auth with server
            logging.info('Logging in with username ' + config.username)
            timeout = aiohttp.ClientTimeout(total=10)
            cookie = None
            async with aiohttp.ClientSession(timeout=timeout) as session:
                while uid is None:
                    try:
                        async with session.get(config.init_url) as iresp:
                            if iresp.status == 200:
                                cookie = iresp.headers['Set-Cookie']
                                csrf_token = cookie.split('XSRF-TOKEN=')[1].split(';')[0]
                                post_data = {
                                    'username': config.username,
                                    'password': config.password
                                }
                                headers = {'Cookie': cookie, 'X-XSRF-TOKEN': csrf_token}
                                async with session.post(config.login_url, json=post_data, headers=headers) as resp:
                                    cookie = resp.headers['Set-Cookie']
                                    obj = await resp.json()
                                    if resp.status == 200:
                                        if obj['type'] != 300:
                                            logging.error('Invalid worker account!')
                                            return
                                        uid = obj['uid']
                                    else:
                                        logging.error('[{}] {}'.format(resp.status, obj['message']))
                                        await asyncio.sleep(5)
                            else:
                                logging.error('[{}] {}'.format(resp.status, obj['message']))
                                await asyncio.sleep(5)
                    except Exception as e:
                        logging.error('Connection failed: ' + repr(e))
                        await asyncio.sleep(5)
            logging.info('Logged in as ' + uid)
            logging.info('Cookie: ' + cookie)
            logging.info('Connecting to ' + config.websocket_url)
            headers = {'Cookie': cookie}
            async with websockets.connect(config.websocket_url, extra_headers=headers, max_size=2 ** 24) as ws:
                logging.info('Connected')
                # Create tasks
                handler_task = asyncio.ensure_future(__message_handler())
                dispatcher_task = asyncio.ensure_future(__message_dispatcher(ws))
                receiver_task = asyncio.ensure_future(__message_receiver(ws))
                tick_task = asyncio.ensure_future(__tick_sender(ws))
                judge_tasks = []
                for i in range(config.parallel_judge_tasks):
                    judge_tasks.append(asyncio.ensure_future(__judge_worker(i)))
                # asyncio.get_event_loop().create_task(__fake_server())
                done, pending = await asyncio.wait(
                    [handler_task, dispatcher_task, receiver_task, tick_task] + judge_tasks,
                    return_when=asyncio.FIRST_COMPLETED
                )
                for task in pending:
                    task.cancel()
        except Exception as e:
            logging.error(str(type(e)))
            raise e
        finally:
            logging.error('Disconnected, retry after 5 secs')
            await asyncio.sleep(5.0)


if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(main())
