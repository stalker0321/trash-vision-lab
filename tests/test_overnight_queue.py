import unittest

from scripts.overnight_queue import QueueStopped, StageFailed, execute_experiments


class QueueTests(unittest.TestCase):
    def jobs(self):
        return [{'name': 'nano640', 'status': 'pending'}, {'name': 'nano320', 'status': 'pending'}]

    def test_sequential_preflight_training_and_validation(self):
        jobs, calls = self.jobs(), []
        execute_experiments(jobs, lambda j, stage: calls.append((j['name'], stage)))
        self.assertEqual(calls, [('nano640','smoke'), ('nano320','smoke'), ('nano640','train'),
                                 ('nano640','val'), ('nano320','train'), ('nano320','val')])
        self.assertTrue(all(j['status'] == 'complete' and j['training_complete'] for j in jobs))

    def test_first_training_failure_does_not_block_second(self):
        jobs, calls = self.jobs(), []
        def invoke(job, stage):
            calls.append((job['name'], stage))
            if (job['name'], stage) == ('nano640', 'train'):
                raise StageFailed('simulated OOM')
        execute_experiments(jobs, invoke)
        self.assertEqual(jobs[0]['status'], 'failed')
        self.assertNotIn(('nano640', 'val'), calls)
        self.assertEqual(jobs[1]['status'], 'complete')

    def test_preflight_failure_skips_only_that_job(self):
        jobs, calls = self.jobs(), []
        def invoke(job, stage):
            calls.append((job['name'], stage))
            if job['name'] == 'nano640':
                raise StageFailed('preflight failed')
        execute_experiments(jobs, invoke)
        self.assertEqual(calls, [('nano640','smoke'), ('nano320','smoke'), ('nano320','train'), ('nano320','val')])

    def test_stop_request_aborts_queue_without_starting_second(self):
        jobs, calls = self.jobs(), []
        def invoke(job, stage):
            calls.append((job['name'], stage))
            if stage == 'train':
                raise QueueStopped('manual stop')
        with self.assertRaises(QueueStopped):
            execute_experiments(jobs, invoke)
        self.assertNotIn(('nano320', 'train'), calls)


if __name__ == '__main__':
    unittest.main()
