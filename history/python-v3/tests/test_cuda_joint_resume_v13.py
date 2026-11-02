"""Native diagnostic checkpoint gate; does not bypass production panel gates."""
import gc
import json
import tempfile
from pathlib import Path
import unittest
import torch
from research_v3.cli.probe_gpu_fullwidth_joint import _model
from research_v3.data.stream_solver_dataset_v13 import make_root
from research_v3.gpu.record_batch_v13 import own_records_to_cuda
from research_v3.gpu.joint_records_v13 import complete_joint_basis_records
from research_v3.train.checkpoint import save_checkpoint, load_checkpoint
from research_v3.train.runner import guarded_step
from research_v3.train.schedule import Sampler, Progress
from research_v3.train.neural_refresh_solver_v13 import NeuralRefreshSolver
from research_v3.train.disk_replay_refresh_v13 import DiskReplayRefresh


@unittest.skipUnless(torch.cuda.is_available(), 'Molab CUDA gate')
class CudaJointResumeTests(unittest.TestCase):
    def test_eight_equals_four_plus_four_parameters_optimizer_rng_and_replay(self):
        config = json.loads((Path(__file__).resolve().parents[2] /
                             'full_training_config.json').read_text())
        torch.set_num_threads(config['threads'])
        record, _ = make_root('train', 0, 126, 0, 7000)
        contract = {'diagnostic': 'full29-joint-fixed-own-root-resume-v13', 'config': config}
        def initialize(path):
            torch.manual_seed(config['seed'])
            model = _model(config).train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
            generator = torch.Generator(device='cuda').manual_seed(9137)
            solver = NeuralRefreshSolver(model,
                search_config=dict(beam_width=2, shortlist=56, max_depth=1,
                                   inference_batch=2, max_expanded=100),
                field_config=dict(mc_samples=config['diffusion']['mc_samples'],
                                  mc_seed=config['feature_seed'],
                                  max_pair_states=config['diffusion']['max_pair_states'], head='qhat'),
                source_sha256={})
            replay = DiskReplayRefresh(solver, journal_path=path,
                                       every_updates=1000, source_sha256={})
            return model, optimizer, generator, replay, Sampler(1, 17), Progress()
        def run(state, count):
            model, optimizer, generator, replay, sampler, progress = state
            for _ in range(count):
                sampler.next(1)
                records = replay((record,), progress.attempted + 1)
                batch = own_records_to_cuda(records)
                loss, _ = complete_joint_basis_records(model, batch, records,
                    generator=generator, choice_count=2, action_chunk=32, samples=2,
                    max_evaluations=5000, max_bytes=268435456, weight=.1,
                    qhat_weight=.1, supervision_weight=1., teacher_weight=.1, holdout=None,
                    mc_samples=config['diffusion']['mc_samples'], mc_seed=config['feature_seed'],
                    max_pair_states=config['diffusion']['max_pair_states'])
                self.assertTrue(guarded_step(model, optimizer, loss, progress))
        def save(state, directory):
            model, optimizer, generator, replay, sampler, progress = state
            return save_checkpoint(directory, model, optimizer, sampler, progress, contract,
                                   extra=dict(basis_rng=generator.get_state(), replay=replay.state_dict()))
        def dispose(state):
            state[3].close()
        def compare(a, b):
            if isinstance(a, torch.Tensor):
                self.assertTrue(torch.equal(a, b))
            elif isinstance(a, dict):
                self.assertEqual(set(a), set(b))
                for key in a: compare(a[key], b[key])
            elif isinstance(a, (list, tuple)):
                self.assertEqual(len(a), len(b))
                for left, right in zip(a, b): compare(left, right)
            else: self.assertEqual(a, b)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = initialize(root/'a.sqlite'); run(state, 8)
            full = save(state, root/'full')
            dispose(state); del state; gc.collect(); torch.cuda.empty_cache()
            state = initialize(root/'b.sqlite'); run(state, 4)
            half = save(state, root/'split')
            dispose(state); del state; gc.collect(); torch.cuda.empty_cache()
            state = initialize(root/'b.sqlite')
            model, optimizer, generator, replay, sampler, progress = state
            extra = load_checkpoint(half, model, optimizer, sampler, progress, contract)
            generator.set_state(extra['basis_rng']); replay.load_state_dict(extra['replay'])
            run(state, 4); resumed = save(state, root/'split')
            left = torch.load(full, map_location='cpu', weights_only=True)
            right = torch.load(resumed, map_location='cpu', weights_only=True)
            for key in ('model', 'optimizer', 'sampler', 'progress', 'rng'):
                compare(left[key], right[key])
            compare(left['extra']['basis_rng'], right['extra']['basis_rng'])
            self.assertEqual(left['extra']['replay']['cursor']['seq'],
                             right['extra']['replay']['cursor']['seq'])
            self.assertEqual(state[3].journal.get(record).state, record.state)
            dispose(state)


if __name__ == '__main__': unittest.main()
