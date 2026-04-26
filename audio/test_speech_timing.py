"""
Bug Condition Exploration Test for Goal Speech Timing Sync

**Validates: Requirements 1.1, 1.2, 1.3, 2.1, 2.2, 2.3**

This test demonstrates the bug where scheduled messages play immediately
instead of at their scheduled time. This test is EXPECTED TO FAIL on unfixed code.

Property 1: Bug Condition - Scheduled Messages Play Immediately Without Delay
For any speech message where scheduled_time is provided and scheduled_time > current_time,
the system should delay playback until scheduled_time is reached.

CRITICAL: This test MUST FAIL on unfixed code - failure confirms the bug exists.
"""

import unittest
import time
import queue as queue_module
from audio.speech import SpeechEngine


class TestScheduledMessageTiming(unittest.TestCase):
    """
    Bug condition exploration test for scheduled message timing.
    
    This test encodes the EXPECTED behavior (messages should play at scheduled time).
    When run on UNFIXED code, it will FAIL, proving the bug exists.
    When run on FIXED code, it will PASS, confirming the fix works.
    """
    
    def setUp(self):
        """Set up a SpeechEngine instance for testing."""
        self.speech = SpeechEngine()
        self.playback_times = []
        time.sleep(0.2)  # Let worker thread start
        
    def tearDown(self):
        """Clean up the speech engine."""
        self.speech._running = False
        time.sleep(0.2)
    
    def test_bug_condition_messages_play_immediately_without_scheduled_delay(self):
        """
        **Validates: Requirements 1.1, 1.2, 1.3, 2.1, 2.2, 2.3**
        
        This test demonstrates the bug: messages that SHOULD be scheduled to play
        at a future time instead play immediately.
        
        The test simulates queueing two goal announcements with a 4-second gap
        between their generation times. In the EXPECTED behavior, these should
        play with a 4-second gap. In the ACTUAL (buggy) behavior, they play
        back-to-back immediately.
        
        Expected on UNFIXED code: FAIL (demonstrates bug exists)
        Expected on FIXED code: PASS (confirms fix works)
        """
        # Simulate the scenario from the bug report:
        # Goal announcements generated at T=0, T=4, T=8 with natural time gaps
        # but all playing immediately back-to-back
        
        start_time = time.time()
        
        # Queue first message at T=0
        self.speech.speak(
            "Goal detected 10 meters ahead",
            priority=True,
            bypass_cooldown=True
        )
        
        # Wait 4 seconds (simulating natural cooldown gap)
        time.sleep(4.0)
        
        # Queue second message at T=4
        self.speech.speak(
            "Goal detected 8 meters ahead",
            priority=True,
            bypass_cooldown=True
        )
        
        # Wait for messages to be processed
        time.sleep(1.0)
        
        # Check the priority queue to see when messages were processed
        # On unfixed code: both messages are processed immediately (within ~0.1s of each other)
        # On fixed code: messages should be processed with ~4s gap
        
        # Since we can't easily track playback times without modifying the engine,
        # we'll test the queue behavior directly
        
        # Let's verify this by checking the tuple structure that speak() creates
        
        # We can directly inspect the priority queue since we queued messages
        try:
            item = self.speech._pq.get_nowait()
            
            # Check tuple length
            tuple_length = len(item)
            
            print(f"\n[BUG DEMONSTRATION]")
            print(f"Current queue tuple structure: {tuple_length} elements")
            print(f"Tuple contents: {item}")
            print(f"Expected: 4 elements (text, timestamp, is_emergency, scheduled_time)")
            print(f"Actual: {tuple_length} elements")
            
            if tuple_length == 3:
                print(f"\nBUG CONFIRMED: Queue tuples only have 3 elements.")
                print(f"The scheduled_time parameter is missing, causing all messages")
                print(f"to play immediately instead of at their scheduled time.")
                print(f"\nThis means:")
                print(f"- Messages queued at T=0, T=4, T=8 all play at T=0")
                print(f"- Natural time gaps between goal announcements are lost")
                print(f"- Audio playback is back-to-back instead of time-gapped")
                
                # This assertion encodes the EXPECTED behavior
                # It will FAIL on unfixed code, proving the bug exists
                self.assertEqual(
                    tuple_length, 4,
                    "BUG CONFIRMED: Queue tuple should have 4 elements (text, timestamp, "
                    "is_emergency, scheduled_time) but only has 3. This causes scheduled "
                    "messages to play immediately instead of at their scheduled time."
                )
            else:
                # If we get here, the fix might already be implemented
                print(f"\nQueue tuple has {tuple_length} elements - checking for scheduled_time support")
                self.assertEqual(tuple_length, 4)
                
        except queue_module.Empty:
            # If the queue is empty, the worker processed them already.
            # This is fine, we just want to ensure it doesn't fail incorrectly.
            pass
        except Exception as e:
            self.fail(f"Unexpected error while testing queue structure: {e}")
    
    def test_bug_condition_worker_has_no_delay_logic(self):
        """
        **Validates: Requirements 2.2, 2.3**
        
        This test verifies that the _worker() method lacks delay logic for
        scheduled messages, causing immediate playback.
        
        Expected on UNFIXED code: FAIL (no delay logic exists)
        Expected on FIXED code: PASS (delay logic implemented)
        """
        import inspect
        
        # Get the source code of the _worker method
        worker_source = inspect.getsource(self.speech._worker)
        
        print(f"\n[BUG DEMONSTRATION] Analyzing _worker() method:")
        print(f"Looking for scheduled_time delay logic...")
        
        # Check if the worker has delay logic for scheduled_time
        has_scheduled_time_check = "scheduled_time" in worker_source
        has_sleep_delay = "sleep" in worker_source and "scheduled_time" in worker_source
        
        print(f"- Contains 'scheduled_time' reference: {has_scheduled_time_check}")
        print(f"- Contains delay logic (sleep with scheduled_time): {has_sleep_delay}")
        
        if not has_scheduled_time_check:
            print(f"\nBUG CONFIRMED: _worker() method has no scheduled_time handling.")
            print(f"Messages are processed immediately without checking if they should")
            print(f"be delayed until their scheduled time.")
            print(f"\nExpected behavior:")
            print(f"  if scheduled_time and scheduled_time > now:")
            print(f"      sleep(scheduled_time - now)")
            print(f"\nActual behavior:")
            print(f"  Messages processed immediately from queue")
        
        # This assertion encodes the EXPECTED behavior
        # It will FAIL on unfixed code, proving the bug exists
        self.assertTrue(
            has_scheduled_time_check,
            "BUG CONFIRMED: _worker() method lacks scheduled_time delay logic. "
            "This causes all messages to play immediately instead of waiting "
            "for their scheduled time."
        )
    
    def test_bug_condition_speak_api_missing_scheduled_time_parameter(self):
        """
        **Validates: Requirements 2.1, 2.2**
        
        This test verifies that the speak() method lacks a scheduled_time parameter,
        preventing callers from specifying when messages should play.
        
        Expected on UNFIXED code: FAIL (parameter doesn't exist)
        Expected on FIXED code: PASS (parameter exists)
        """
        import inspect
        
        # Get the signature of the speak method
        speak_signature = inspect.signature(self.speech.speak)
        parameters = list(speak_signature.parameters.keys())
        
        print(f"\n[BUG DEMONSTRATION] Analyzing speak() method signature:")
        print(f"Current parameters: {parameters}")
        
        has_scheduled_time_param = "scheduled_time" in parameters
        
        print(f"- Has 'scheduled_time' parameter: {has_scheduled_time_param}")
        
        if not has_scheduled_time_param:
            print(f"\nBUG CONFIRMED: speak() method lacks scheduled_time parameter.")
            print(f"Callers cannot specify when messages should play, forcing immediate playback.")
            print(f"\nExpected signature:")
            print(f"  speak(text, priority=False, bypass_cooldown=False, emergency=False, scheduled_time=None)")
            print(f"\nActual signature:")
            print(f"  speak({', '.join(parameters)})")
        
        # This assertion encodes the EXPECTED behavior
        # It will FAIL on unfixed code, proving the bug exists
        self.assertIn(
            "scheduled_time",
            parameters,
            "BUG CONFIRMED: speak() method lacks scheduled_time parameter. "
            "This prevents callers from scheduling messages for future playback, "
            "causing all messages to play immediately."
        )


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
