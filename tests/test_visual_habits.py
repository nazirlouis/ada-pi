import asyncio
import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from backend.posture import PostureStore
from backend.visual_habits import VisualHabitService, desk_descriptor, descriptor_change, phone_near_user


class FakePose:
    latest_result = None
    def office_occupied(self): return True


class FakeLive:
    def __init__(self): self.messages=[]
    async def send_text_turn(self, message): self.messages.append(message)


class VisualHabitTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.store=PostureStore(Path(self.tmp.name)/"habits.db")
        self.live=FakeLive(); self.service=VisualHabitService(None, FakePose(), None, self.store, self.live)

    async def asyncTearDown(self): self.tmp.cleanup()

    async def test_sitting_gap_latch_and_break_reset(self):
        await self.service._process_sitting(1000, True)
        await self.service._process_sitting(1010, False)
        self.assertIsNotNone(self.service.states["sitting_too_long"]["candidate_since"])
        await self.service._process_sitting(4600, True)
        self.assertTrue(self.service.states["sitting_too_long"]["latched"])
        await self.service._process_sitting(4700, False)
        self.assertTrue(self.service.states["sitting_too_long"]["latched"])
        await self.service._process_sitting(5001, False)
        self.assertFalse(self.service.states["sitting_too_long"]["latched"])
        self.assertEqual(1, self.store.habit_profile("sitting_too_long")["rolling_occurrences"])

    async def test_phone_requires_proximity_and_one_episode(self):
        pose={"keypoints":[{"name":"left_wrist","x":.5,"y":.5,"score":.9}]}
        phone={"label":"cell phone","score":.9,"x":.48,"y":.48,"width":.05,"height":.1}
        self.assertTrue(phone_near_user([phone],pose))
        self.assertFalse(phone_near_user([{**phone,"x":.05,"y":.05}],pose))
        for now in range(1000,1121,2): await self.service._process_phone(now, True)
        for now in range(1122,1180,2): await self.service._process_phone(now, True)
        self.assertEqual(1, self.store.habit_profile("phone_distraction")["rolling_occurrences"])

    def test_descriptor_is_numeric_and_lighting_normalized(self):
        def jpeg(level):
            out=io.BytesIO(); Image.new("RGB",(64,48),(level,level,level)).save(out,"JPEG"); return out.getvalue()
        first, second=desk_descriptor(jpeg(80)),desk_descriptor(jpeg(180))
        self.assertGreater(len(first),100); self.assertLess(descriptor_change(first,second),.03)
        state=self.store.monitor_state("desk_clutter_calibration")
        self.assertNotIn("jpeg",state)

    async def test_settings_and_calibration_survive_clear(self):
        self.service.update_settings("sitting_too_long",{"maximum_sitting_minutes":90})
        self.store.save_monitor_state("desk_clutter_calibration",{"descriptor":[1.0]})
        self.store.record_habit_occurrence("phone_distraction","2026-01-01T12:00:00+00:00")
        self.store.clear_habit_history()
        restarted=VisualHabitService(None,FakePose(),None,self.store,self.live)
        self.assertEqual(90,restarted.settings["sitting_too_long"]["maximum_sitting_minutes"])
        self.assertEqual([1.0],self.store.monitor_state("desk_clutter_calibration")["descriptor"])
        self.assertIsNone(self.store.habit_profile("phone_distraction"))


if __name__ == "__main__": unittest.main()
