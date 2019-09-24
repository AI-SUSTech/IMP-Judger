import asyncio
import unittest
import msg_types
from case.ncs_case import NCSCase
from ie import estimate, estimate_async, SolutionError


class TestAPlusBCase(unittest.TestCase):

    def setUp(self):
        with open('./examples/ncs-example.zip', 'rb') as zipfile:
            self.case = NCSCase(zipfile.read()).__enter__()
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(None)

    def test_config(self):
        self.assertEqual('parameter.json', self.case.entry)
        self.assertEqual('-d 12 -c algorithm_ncs/parameter.json', self.case.parameters)
        self.assertEqual(10, self.case.time)
        self.assertEqual(256, self.case.memory)
        self.assertEqual(8, self.case.cpu)

    def test_run(self):
        async def run_main():
            timedout, stdout, stderr, exitcode = await self.case.run()
            stdout = stdout.decode('utf8')
            stderr = stderr.decode('utf8')
            print(stdout)
            print(stderr)
            print("IS TIMEOUT {}".format(timedout))
            self.assertEqual(0, exitcode)
            self.assertEqual('35', stdout.strip())
            self.assertEqual('', stderr.strip())
        self.loop.run_until_complete(asyncio.wait([run_main()]))

    def tearDown(self):
        self.case.close()
        self.loop.close()



if __name__ == '__main__':
    unittest.main()