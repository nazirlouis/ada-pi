import asyncio
import io
import tempfile
import unittest
from unittest.mock import AsyncMock, patch
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

from PIL import Image

from backend.posture import PostureStore
from backend.realtime_provider import ProviderEvent
from backend.visual_habits import VisualHabitService, desk_descriptor, descriptor_change, hand_near_face, phone_near_user


class FakePose:
    latest_result = None
    def office_occupied(self): return True


class FakeLive:
    def __init__(self): self.messages=[]
    async def send_text_turn(self, message): self.messages.append(message)


class ChallengeLive(FakeLive):
    def __init__(self, challenge_id): super().__init__(); self.challenge_id=challenge_id; self.frames=[]
    async def send_video(self, frame): self.frames.append(frame)
    async def events(self):
        yield ProviderEvent("response_completed",{})
        yield ProviderEvent("habit_observation",{"challenge_id":self.challenge_id,"habit_key":"not_drinking_enough_water","observed":True,"confidence":.9,"reason":"visible drink"})


class FreshCamera:
    latest_frame=b"jpeg"
    latest_frame_at=10**12
    def __init__(self): self._generation=0
    @property
    def generation(self): self._generation+=1; return self._generation


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

    async def test_late_work_records_once_for_the_night_across_midnight(self):
        zone=ZoneInfo("America/New_York")
        late=datetime(2026,8,23,22,1,tzinfo=zone).timestamp()
        after_midnight=datetime(2026,8,24,1,0,tzinfo=zone).timestamp()
        await self.service._process_late_work(late, True)
        await self.service._process_late_work(after_midnight, True)
        self.assertEqual(1,self.store.habit_profile("working_too_late")["rolling_occurrences"])
        self.assertEqual("2026-08-23",self.service.states["working_too_late"]["night_key"])

    async def test_manual_late_check_returns_to_ada_without_false_occurrence(self):
        noon=datetime(2026,8,23,12,0,tzinfo=ZoneInfo("America/New_York")).timestamp()
        await self.service.trigger_check("working_too_late",now=noon)
        self.assertIsNone(self.store.habit_profile("working_too_late"))
        self.assertIn("not past",self.live.messages[-1])

    async def test_water_accumulates_presence_and_confirmed_miss_records(self):
        state=self.service.states["not_drinking_enough_water"]
        state.update(accumulated_seconds=3590,last_tick=1000,last_present=1000)
        await self.service._process_water(1005,True)
        self.assertEqual(3595,state["accumulated_seconds"])
        challenge={"id":"water-1"}; state["challenge"]=challenge
        await self.service._apply_observation("not_drinking_enough_water",challenge,{"challenge_id":"water-1","habit_key":"not_drinking_enough_water","observed":False,"confidence":.9,"reason":"No drinking visible"})
        self.assertEqual(0,state["accumulated_seconds"])
        self.assertEqual(1,self.store.habit_profile("not_drinking_enough_water")["rolling_occurrences"])
        self.assertFalse(state["latched"])

    async def test_visible_water_completion_does_not_create_habit(self):
        state=self.service.states["not_drinking_enough_water"]
        challenge={"id":"water-2"}; state.update(challenge=challenge,accumulated_seconds=3600)
        await self.service._apply_observation("not_drinking_enough_water",challenge,{"challenge_id":"water-2","habit_key":"not_drinking_enough_water","observed":True,"confidence":.8,"reason":"Drank from bottle"})
        self.assertIsNone(self.store.habit_profile("not_drinking_enough_water"))
        self.assertEqual("completed",state["state"])

    async def test_low_confidence_water_is_inconclusive(self):
        state=self.service.states["not_drinking_enough_water"]
        challenge={"id":"water-3"}; state.update(challenge=challenge,accumulated_seconds=3600)
        await self.service._apply_observation("not_drinking_enough_water",challenge,{"challenge_id":"water-3","habit_key":"not_drinking_enough_water","observed":False,"confidence":.4,"reason":"unclear"})
        self.assertIsNone(self.store.habit_profile("not_drinking_enough_water"))
        self.assertEqual(3600,state["accumulated_seconds"])

    def test_hand_to_face_cue_requires_visible_wrist_and_face(self):
        self.assertTrue(hand_near_face({"keypoints":[{"name":"nose","x":.5,"y":.4,"score":.9},{"name":"left_wrist","x":.55,"y":.45,"score":.8}]}))
        self.assertFalse(hand_near_face({"keypoints":[{"name":"nose","x":.5,"y":.4,"score":.9},{"name":"left_wrist","x":.1,"y":.9,"score":.8}]}))

    async def test_confirmed_junk_food_latches_and_resets_after_clear_period(self):
        state=self.service.states["junk_food"]; challenge={"id":"junk-1"}; state["challenge"]=challenge
        await self.service._apply_observation("junk_food",challenge,{"challenge_id":"junk-1","habit_key":"junk_food","observed":True,"confidence":.91,"reason":"Eating chips","item_identified":"potato chips","consumption_visible":True,"classified_unhealthy":True})
        self.assertTrue(state["latched"])
        await self.service._process_junk_food(state["last_evidence"]+29*60,False)
        self.assertTrue(state["latched"])
        await self.service._process_junk_food(state["last_evidence"]+30*60+1,False)
        self.assertFalse(state["latched"])

    async def test_hand_gesture_or_ambiguous_chewing_cannot_record_junk_food(self):
        state=self.service.states["junk_food"]; challenge={"id":"junk-gesture"}; state["challenge"]=challenge
        await self.service._apply_observation("junk_food",challenge,{"challenge_id":"junk-gesture","habit_key":"junk_food","observed":True,"confidence":.95,"reason":"Hand touched face","item_identified":"","consumption_visible":False,"classified_unhealthy":False})
        self.assertIsNone(self.store.habit_profile("junk_food"))
        self.assertFalse(state["latched"])

    async def test_water_challenge_sends_each_fresh_frame_and_uses_live_tool_verdict(self):
        challenge={"id":"burst-1","habit_key":"not_drinking_enough_water","phase":"prompting","created_at":1000}
        live=ChallengeLive("burst-1"); service=VisualHabitService(FreshCamera(),FakePose(),None,self.store,live)
        service.settings["not_drinking_enough_water"]["response_window_seconds"]=5
        service.states["not_drinking_enough_water"].update(challenge=challenge,accumulated_seconds=3600)
        with patch("backend.visual_habits.asyncio.sleep",new=AsyncMock()):
            await service._run_challenge("not_drinking_enough_water",challenge)
        self.assertEqual(5,len(live.frames))
        self.assertEqual("completed",service.states["not_drinking_enough_water"]["state"])
        self.assertIn("report_habit_observation",live.messages[-1])


if __name__ == "__main__": unittest.main()
