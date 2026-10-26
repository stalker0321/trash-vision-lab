import unittest
from urllib.error import URLError

from scripts.vast_watch import needs_emergency_stop


class EmergencyStopTests(unittest.TestCase):
    def test_unguarded_network_outage_stops_after_ten_failures(self):
        config = {'deadline_unix': 1000}
        self.assertFalse(needs_emergency_stop(URLError('offline'), 9, config, 100))
        self.assertTrue(needs_emergency_stop(URLError('offline'), 10, config, 100))

    def test_confirmed_remote_deadline_allows_transport_recovery(self):
        config = {'deadline_unix': 1000, 'remote_deadline_guard_confirmed': True}
        self.assertFalse(needs_emergency_stop(URLError('offline'), 57, config, 999))

    def test_deadline_always_stops_even_with_a_confirmed_guard(self):
        config = {'deadline_unix': 1000, 'remote_deadline_guard_confirmed': True}
        self.assertTrue(needs_emergency_stop(URLError('offline'), 1, config, 1000))

    def test_guard_does_not_suppress_integrity_failures(self):
        config = {'deadline_unix': 1000, 'remote_deadline_guard_confirmed': True}
        self.assertTrue(needs_emergency_stop(ValueError('checksum mismatch'), 10, config, 100))


if __name__ == '__main__':
    unittest.main()
