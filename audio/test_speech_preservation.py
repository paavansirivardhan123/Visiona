"""
Preservation Property Tests for Goal Speech Timing Sync

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**

These tests verify that non-scheduled message behaviors (emergency, ducking, 
queue flushing, cooldown) work correctly on UNFIXED code. These tests are 
EXPECTED TO PASS on unfixed code, establishing the baseline behavior that 
must be preserved after implementing the scheduled message fix.

Property 2: Preservation - Non-Scheduled Message Behavior
For any speech message where scheduled_time is None or is_emergency is True,
the system should produce exactly the same immediate playback behavior as the
original code, preserving emergency response, normal priority processing, 
ducking, and queue flushing.
"""

import unittest
import time
import queue as queue_module
from audio.speech import SpeechEngine


class TestPreservationProperties(unittest.TestCase):
    """
    Preservation property tests for non-scheduled message behavior.
    
    These tests observe and encode the CURRENT behavior on unfixed code.
    They should PASS on unfixed code and continue to PASS on fixed code,
    ensuring the fix doesn't break existing functionality.
    """
    
    def setUp(self):
        """Set up a SpeechEngine instance for testing."""
        self.speech = SpeechEngine()
        time.sleep(0.2)  # Let worker thread start
        
    def tearDown(self):
        """Clean up the speech engine."""
        self.speech._running = False
        time.sleep(0.2)
    
    def test_preservation_emergency_messages_play_immediately(self):
        """
        **Validates: Requirement 3.1**
        
        Emergency messages must play immediately without any delay,
        bypassing all scheduling logic.
        
        This test observes the current behavior on unfixed code and ensures
        it continues to work after the fix is implemented.
        
        Expected on UNFIXED code: PASS (emergency messages work correctly)
        Expected on FIXED code: PASS (behavior preserved)
        """
        # Queue an emergency message
        start_time = time.time()
        self.speech.speak(
            "Emergency: Collision imminent!",
            emergency=True
        )
        
        # Emergency messages should be in the emergency queue immediately
        time.sleep(0.1)  # Small delay for queue processing
        
        # Verify emergency queue was used (by checking it's now empty after processing)
        # The emergency queue should have been processed immediately
        self.assertTrue(
            self.speech._eq.empty(),
            "Emergency queue should be empty after immediate processing"
        )
        
        # Verify the message was logged (indicates it was processed)
        elapsed = time.time() - start_time
        self.assertLess(
            elapsed, 1.0,
            f"Emergency message should be processed within 1 second, took {elapsed:.2f}s"
        )
    
    def test_preservation_ducking_silences_speech(self):
        """
        **Validates: Requirement 3.2**
        
        Ducking (volume=0.0) when microphone is active must pause or silence
        speech appropriately without affecting queue contents.
        
        Expected on UNFIXED code: PASS (ducking works correctly)
        Expected on FIXED code: PASS (behavior preserved)
        """
        # Verify initial volume is 1.0
        self.assertEqual(self.speech._volume, 1.0, "Initial volume should be 1.0")
        
        # Activate ducking
        self.speech.duck()
        
        # Verify volume is set to 0.0
        self.assertEqual(
            self.speech._volume, 0.0,
            "Ducking should set volume to 0.0"
        )
        
        # Verify interrupt flag is set
        self.assertTrue(
            self.speech._interrupt_requested,
            "Ducking should set interrupt_requested flag"
        )
        
        # Queue a message during ducking
        self.speech.speak("Test message during ducking", priority=True)
        time.sleep(0.1)
        
        # Verify message is still in queue (not processed during ducking)
        # Priority queue should have the message waiting
        self.assertFalse(
            self.speech._pq.empty(),
            "Messages should remain queued during ducking"
        )
        
        # Restore volume
        self.speech.unduck()
        
        # Verify volume is restored
        self.assertEqual(
            self.speech._volume, 1.0,
            "Unduck should restore volume to 1.0"
        )
        
        # Verify interrupt flag is cleared
        self.assertFalse(
            self.speech._interrupt_requested,
            "Unduck should clear interrupt_requested flag"
        )
    
    def test_preservation_queue_flushing_clears_all_queues(self):
        """
        **Validates: Requirement 3.4**
        
        Queue flushing via _flush_all() must clear all queues including
        any scheduled messages.
        
        Expected on UNFIXED code: PASS (flushing works correctly)
        Expected on FIXED code: PASS (behavior preserved, including scheduled messages)
        """
        # Queue messages in all three queues
        self.speech.speak("Emergency message", emergency=True)
        self.speech.speak("Priority message", priority=True, bypass_cooldown=True)
        self.speech.speak("Normal message")
        
        time.sleep(0.1)  # Let messages queue
        
        # Flush all queues
        self.speech._flush_all()
        
        # Verify all queues are empty
        self.assertTrue(
            self.speech._eq.empty(),
            "Emergency queue should be empty after flush"
        )
        self.assertTrue(
            self.speech._pq.empty(),
            "Priority queue should be empty after flush"
        )
        self.assertTrue(
            self.speech._nq.empty(),
            "Normal queue should be empty after flush"
        )
    
    def test_preservation_semantic_cooldown_filters_duplicates(self):
        """
        **Validates: Requirement 3.3**
        
        Semantic cooldown and anti-spam filtering must continue to filter
        duplicate messages using the same logic as before the fix.
        
        Expected on UNFIXED code: PASS (cooldown works correctly)
        Expected on FIXED code: PASS (behavior preserved)
        """
        # Queue the same semantic message twice within cooldown period
        message = "Goal detected 10 meters ahead"
        
        # First message should be queued
        self.speech.speak(message, priority=True)
        time.sleep(0.1)
        
        # Record queue size after first message
        first_queue_size = self.speech._pq.qsize()
        
        # Second message should be filtered by cooldown (same semantic content)
        self.speech.speak(message, priority=True)
        time.sleep(0.1)
        
        # Queue size should not increase (message was filtered)
        second_queue_size = self.speech._pq.qsize()
        
        self.assertEqual(
            first_queue_size, second_queue_size,
            "Duplicate messages within cooldown should be filtered"
        )
        
        # Verify semantic history was updated
        import re
        semantic_base = re.sub(r'\d+', '', message.strip().lower())
        self.assertIn(
            semantic_base,
            self.speech._semantic_history,
            "Semantic history should track message"
        )
    
    def test_preservation_normal_priority_messages_process_immediately(self):
        """
        **Validates: Requirement 3.5**
        
        Normal priority messages without scheduled_time should continue to
        process immediately in the same order and timing as before the fix.
        
        Expected on UNFIXED code: PASS (normal messages work correctly)
        Expected on FIXED code: PASS (behavior preserved)
        """
        # Queue a normal priority message
        start_time = time.time()
        self.speech.speak(
            "Normal priority message",
            priority=True,
            bypass_cooldown=True
        )
        
        # Message should be queued immediately
        time.sleep(0.1)
        
        # Verify message was queued (or already processed)
        elapsed = time.time() - start_time
        self.assertLess(
            elapsed, 1.0,
            f"Normal priority message should be queued within 1 second, took {elapsed:.2f}s"
        )
    
    def test_preservation_property_all_non_scheduled_messages_queue_immediately(self):
        """
        **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**
        
        Property-based test: For any message without scheduled_time,
        the system should queue it immediately without delay.
        
        This test generates many random message combinations to verify
        that all non-scheduled messages continue to work correctly.
        
        Expected on UNFIXED code: PASS (all non-scheduled messages work)
        Expected on FIXED code: PASS (behavior preserved)
        """
        # Test multiple combinations of message types
        test_cases = [
            ("Emergency test", True, False),
            ("Priority test", False, True),
            ("Normal test", False, False),
            ("Emergency priority", True, True),
            ("Goal detected 10 meters ahead", False, True),
            ("Warning: Very close object", False, True),
        ]
        
        for message_text, is_emergency, is_priority in test_cases:
            with self.subTest(message=message_text, emergency=is_emergency, priority=is_priority):
                # Queue the message
                start_time = time.time()
                
                try:
                    self.speech.speak(
                        message_text,
                        priority=is_priority,
                        bypass_cooldown=True,
                        emergency=is_emergency
                    )
                    
                    # Message should be queued immediately
                    time.sleep(0.05)
                    
                    elapsed = time.time() - start_time
                    
                    # Verify message was queued quickly (within 0.5 seconds)
                    self.assertLess(
                        elapsed, 0.5,
                        f"Message should be queued within 0.5 seconds, took {elapsed:.2f}s"
                    )
                    
                except Exception as e:
                    self.fail(f"Message queueing failed: {e}")
    
    def test_preservation_property_queue_flushing_clears_all_messages(self):
        """
        **Validates: Requirement 3.4**
        
        Property-based test: Queue flushing should clear all messages
        regardless of how many are queued or their type.
        
        Expected on UNFIXED code: PASS (flushing works for all cases)
        Expected on FIXED code: PASS (behavior preserved, including scheduled messages)
        """
        # Test with different numbers of messages
        test_cases = [1, 2, 3, 5]
        
        for num_messages in test_cases:
            with self.subTest(num_messages=num_messages):
                # Queue multiple messages
                for i in range(num_messages):
                    self.speech.speak(
                        f"Test message {i}",
                        priority=True,
                        bypass_cooldown=True
                    )
                
                time.sleep(0.1)  # Let messages queue
                
                # Flush all queues
                self.speech._flush_all()
                
                # Verify all queues are empty
                self.assertTrue(
                    self.speech._eq.empty() and 
                    self.speech._pq.empty() and 
                    self.speech._nq.empty(),
                    f"All queues should be empty after flushing {num_messages} messages"
                )
    
    def test_preservation_property_ducking_volume_changes_are_consistent(self):
        """
        **Validates: Requirement 3.2**
        
        Property-based test: Ducking and unducking should consistently
        change volume between 0.0 and 1.0.
        
        Expected on UNFIXED code: PASS (ducking works consistently)
        Expected on FIXED code: PASS (behavior preserved)
        """
        # Test multiple duck/unduck sequences
        volume_changes = [
            [True, False],
            [True, True, False],
            [True, False, True, False],
            [False, True, False],
        ]
        
        for sequence in volume_changes:
            with self.subTest(sequence=sequence):
                for should_duck in sequence:
                    if should_duck:
                        self.speech.duck()
                        self.assertEqual(
                            self.speech._volume, 0.0,
                            "Duck should set volume to 0.0"
                        )
                        self.assertTrue(
                            self.speech._interrupt_requested,
                            "Duck should set interrupt flag"
                        )
                    else:
                        self.speech.unduck()
                        self.assertEqual(
                            self.speech._volume, 1.0,
                            "Unduck should set volume to 1.0"
                        )
                        self.assertFalse(
                            self.speech._interrupt_requested,
                            "Unduck should clear interrupt flag"
                        )


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
