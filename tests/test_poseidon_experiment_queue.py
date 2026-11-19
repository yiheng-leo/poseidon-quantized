"""CPU-only checks of the actual experiment matrix and queue coordination."""
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/poseidon_experiment_queue.py"
spec = importlib.util.spec_from_file_location("poseidon_queue", SCRIPT)
queue_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue_module)
q = queue_module


def make_queue(root, jobs, groups=()):
    queue = root / "queue"
    for name in ("states", "claims", "workers", "additional", "selections", "job_records", "gpu_locks"):
        (queue / name).mkdir(parents=True)
    plan = {"root": str(root), "jobs": jobs, "groups": list(groups), "source_sha256": {},
            "source_revision": "test-fixture", "poll_seconds": 0.01, "maximum_idle_gpu_memory_mib": 1000,
            "evaluation_protocol": {"protocol": "safe-svd", "final_time": 10}, "counts": {}}
    q.atomic_json(queue / "plan.json", plan)
    for job in jobs:
        q.atomic_json(q.state_path(queue, job), {"id": job["id"], "status": "queued"})
    return queue, plan


class ExperimentQueueTests(unittest.TestCase):
    def test_requested_matrix_and_seed_pairings(self):
        plan = q.build_plan(q.SERVER_ROOT)
        formal = [j for j in plan["jobs"] if j["kind"] == "formal"]
        self.assertEqual(len(formal), 63)
        self.assertEqual(sum(j["bits"] == 8 for j in formal), 36)
        self.assertEqual(sum(j["bits"] != 8 and j["method"] == "sbpq" for j in formal), 27)
        low_sweeps = [j for j in plan["jobs"] if j["kind"] == "tuning" and j["bits"] != 8]
        self.assertEqual(len(low_sweeps), 9)
        self.assertEqual({j["label"] for j in low_sweeps}, {"NS-PwC"})
        self.assertEqual(sum(j["external"] for j in plan["jobs"]), 3)
        self.assertEqual(len({j["run_dir"] for j in plan["jobs"]}), 81)
        for job in plan["jobs"]:
            expected = job["expected"]
            self.assertIn(job["seed"], (1, 7, 42))
            self.assertEqual((expected["random_calibration"], expected["calibration_seed"]),
                             ("n", 42) if job["seed"] == 1 else ("y", job["seed"]))
            self.assertIn("/sbpq-led-v2/", job["run_dir"])
            self.assertEqual(expected["traj_num"], 16)
            self.assertEqual(expected["final_time"], 10)
            if job["method"] == "ppq-layerwise":
                self.assertEqual((expected["init_bits"], expected["mc_samples"], expected["epochs"]), (8, 10, 5))
            if job["kind"] == "formal":
                command = job["evaluate"]
                self.assertEqual(command[command.index("--final_time") + 1], "10")
                self.assertEqual(command[command.index("--output_path") + 1], job["eval_output"])

    def test_two_processes_cannot_claim_one_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = q.make_job(root, "NS-PwC", "uniform", 8, 1)
            queue, _ = make_queue(root, [job])
            code = """import importlib.util,json,sys
from pathlib import Path
s=importlib.util.spec_from_file_location('queue',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
p=Path(sys.argv[2]);j=m.claim_next(p,m.read_json(p/'plan.json'),16376,{'host':'fixture','worker_pid':__import__('os').getpid()})
print(j['id'] if j else 'NONE')
"""
            children = [subprocess.Popen([sys.executable, "-c", code, str(SCRIPT), str(queue)],
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(2)]
            outputs = []
            for child in children:
                out, err = child.communicate(timeout=20)
                self.assertEqual(child.returncode, 0, err)
                outputs.append(out.strip())
            self.assertEqual(sorted(outputs), sorted([job["id"], "NONE"]))

    def test_ppq_formal_waits_for_matching_dataset_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); job = q.make_job(root, "CE-RM", "ppq-layerwise", 8, 7)
            queue, plan = make_queue(root, [job])
            self.assertIsNone(q.claim_next(queue, plan, 16376, {}))
            selected = {"group": "CE-RM-bit8", "bits": 8, "status": "selected", "gamma": 1e-5, "avg_weight_bits": 7.96}
            q.atomic_json(q.selection_path(queue, "CE-RM-bit8"), selected)
            resolved = q.claim_next(queue, plan, 16376, {})
            self.assertEqual(resolved["expected"]["gamma"], 1e-5)
            self.assertNotIn(q.GAMMA_PLACEHOLDER, resolved["quantize"])
            selected["group"] = "NS-PwC-bit8"
            q.atomic_json(q.selection_path(queue, "CE-RM-bit8"), selected)
            with self.assertRaises(ValueError):
                q.resolve_job(queue, job)

    def test_gamma_match_expansion_bisection_and_nonmonotonicity(self):
        points = [{"gamma": 1e-5, "avg_weight_bits": 8.4}, {"gamma": 1e-4, "avg_weight_bits": 7.5}]
        kind, token = q.next_gamma(points, 8)
        self.assertEqual(kind, "candidate")
        self.assertAlmostEqual(float(token), math.sqrt(1e-9), places=10)
        self.assertEqual(q.next_gamma([{"gamma": 1e-5, "avg_weight_bits": 8.4}], 8), ("candidate", "1e-4"))
        self.assertEqual(q.next_gamma([{"gamma": 1e-5, "avg_weight_bits": 7.5}], 8), ("candidate", "1e-6"))
        kind, selected = q.next_gamma(points + [{"gamma": 2e-5, "avg_weight_bits": 7.98}], 8)
        self.assertEqual((kind, selected["avg_weight_bits"]), ("selected", 7.98))
        self.assertEqual(q.next_gamma(list(reversed([{"gamma": 1e-5, "avg_weight_bits": 7.0},
                                                    {"gamma": 1e-4, "avg_weight_bits": 9.0}])), 8)[0], "needs_attention")

    def test_coordinator_waits_for_all_coarse_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = [q.make_job(root, "NS-PwC", "ppq-layerwise", 4, 1, gamma=g, tuning=True) for g in q.GAMMAS[4]]
            group = {"id": "NS-PwC-bit4", "label": "NS-PwC", "bits": 4, "max_additional_candidates": 12}
            queue, plan = make_queue(root, jobs, [group])
            artifacts = {jobs[0]["id"]: {"avg_weight_bits": 4.5}, jobs[1]["id"]: {"avg_weight_bits": 3.96}}
            with patch.object(q, "completed_artifact", side_effect=lambda job: artifacts.get(job["id"])):
                q.reconcile(queue, plan)
                self.assertEqual(q.read_json(q.selection_path(queue, group["id"]))["status"], "waiting_candidates")
                artifacts[jobs[2]["id"]] = {"avg_weight_bits": 3.2}
                q.reconcile(queue, plan)
                self.assertEqual(q.read_json(q.selection_path(queue, group["id"]))["gamma"], 0.003)

    def test_supervisor_runs_quantization_before_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); job = q.make_job(root, "NS-PwC", "uniform", 8, 1)
            queue, plan = make_queue(root, [job])
            run = Path(job["run_dir"]); output = Path(job["eval_output"])
            quant = """import json,os,pathlib,sys
p=pathlib.Path(sys.argv[1]);p.mkdir(parents=True)
assert os.environ['CUDA_VISIBLE_DEVICES']=='fixture-uuid'
(p/'compressed_model.pt').write_text('fixture, not a model')
(p/'metadata.json').write_text(json.dumps({'settings':json.loads(sys.argv[2]),'quantization':{'weight_bits':8}}))
(p/'run_state.json').write_text(json.dumps({'status':'completed'}))
"""
            evaluate = """import json,pathlib,sys
assert (pathlib.Path(sys.argv[1])/'run_state.json').is_file()
p=pathlib.Path(sys.argv[2]);p.parent.mkdir(parents=True,exist_ok=True)
p.write_text(json.dumps({'avg_weight_bits':8,'metrics':{'fixture':{'origin':1,'compressed':1.1}}}))
"""
            job["quantize"] = ["{python}", "-c", quant, str(run), json.dumps(job["expected"])]
            job["evaluate"] = ["{python}", "-c", evaluate, str(run), str(output)]
            owner = {"host": "fixture", "gpu_uuid": "fixture-uuid", "worker_pid": 1}
            with patch.object(q, "gpu_inventory", return_value={"used_mib": 0}):
                q.run_job(queue, plan, job, owner, lambda **_: None)
            state = q.read_json(q.state_path(queue, job))
            self.assertEqual(state["status"], "completed", state)
            record = q.read_json(queue / "job_records" / f'{job["id"]}.json')
            self.assertEqual([stage["stage"] for stage in record["stages"]], ["quantization", "evaluation"])
            self.assertTrue(q.evaluation_complete(job))


if __name__ == "__main__":
    unittest.main()
