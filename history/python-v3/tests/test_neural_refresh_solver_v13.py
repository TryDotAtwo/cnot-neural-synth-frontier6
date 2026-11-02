import unittest
import torch
import json
import tempfile
from pathlib import Path
from dataclasses import replace
from research_v3.contracts import AlphabetSpec, Gate
from research_v3.data.generators import record_from_word
from research_v3.cli.probe_gpu_fullwidth_joint import _model
from research_v3.train.disk_replay_refresh_v13 import DiskReplayRefresh
from research_v3.train.neural_refresh_solver_v13 import NeuralRefreshSolver, model_state_sha256


class NeuralRefreshContractTests(unittest.TestCase):
    def test_model_digest_covers_parameters_and_persistent_buffers(self):
        model = torch.nn.Linear(2, 2)
        model.register_buffer('audit_buffer', torch.tensor(1, dtype=torch.int64))
        before = model_state_sha256(model)
        self.assertEqual(before, model_state_sha256(model))
        model.audit_buffer.add_(1)
        self.assertNotEqual(before, model_state_sha256(model))
        before = model_state_sha256(model)
        with torch.no_grad(): model.weight.add_(1)
        self.assertNotEqual(before, model_state_sha256(model))

    def test_invalid_explicit_budgets_and_field_config_rejected(self):
        search = dict(beam_width=2, shortlist=2, max_depth=3,
                      inference_batch=2, max_expanded=30)
        field = dict(mc_samples=2, mc_seed=7, max_pair_states=30, head='qhat')
        for key in search:
            for value in (0, -1, True, 1.5):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    NeuralRefreshSolver(torch.nn.Linear(1, 1),
                        search_config={**search, key: value}, field_config=field,
                        source_sha256={})
        for override in ({'mc_samples': True}, {'max_pair_states': 0},
                         {'mc_seed': 1.5}, {'head': 'hamming'}):
            with self.assertRaises(ValueError):
                NeuralRefreshSolver(torch.nn.Linear(1, 1), search_config=search,
                    field_config={**field, **override}, source_sha256={})


@unittest.skipUnless(torch.cuda.is_available(), 'Molab CUDA gate')
class NeuralRefreshCudaTests(unittest.TestCase):
    def test_full_word_and_training_modes_and_rng(self):
        config = json.loads((Path(__file__).resolve().parents[2] /
                             'full_training_config.json').read_text())
        model = _model(config).train()
        record = record_from_word(AlphabetSpec.all_to_all(8), (Gate(0, 1), Gate(2, 3)),
                                  'train:refresh-cuda', 7, source='ordinary_walk_v1')
        record = replace(record, split='train')
        solver = NeuralRefreshSolver(model,
            search_config=dict(beam_width=56, shortlist=56, max_depth=2,
                               inference_batch=8, max_expanded=100),
            field_config=dict(mc_samples=config['diffusion']['mc_samples'],
                              mc_seed=config['feature_seed'],
                              max_pair_states=config['diffusion']['max_pair_states'],
                              head='qhat'), source_sha256={})
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state().clone()
        modes = [m.training for m in model.modules()]
        word = solver(record, 1)
        self.assertIsNotNone(word)
        self.assertEqual(word.physical_cost, 2)
        self.assertGreater(solver.last_search['result']['value_evaluations'], 0)
        self.assertEqual(word.start, record.state)
        self.assertEqual(modes, [m.training for m in model.modules()])
        self.assertTrue(torch.equal(cpu_rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(cuda_rng, torch.cuda.get_rng_state()))
        redundant = record_from_word(AlphabetSpec.all_to_all(8),
                                     (Gate(0, 1), Gate(2, 3), Gate(1, 2), Gate(1, 2)),
                                     'train:refresh-disk', 7, source='ordinary_walk_v1')
        redundant = replace(redundant, split='train')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'replay.sqlite'
            replay = DiskReplayRefresh(solver, journal_path=path,
                                       every_updates=2, source_sha256={})
            improved = replay((redundant,), 2)[0]
            self.assertEqual((redundant.upper_bound, improved.upper_bound), (4, 2))
            checkpoint = replay.state_dict()
            self.assertEqual(checkpoint['solver']['last_search']['result']['verified'], True)
            self.assertEqual(len(checkpoint['solver']['last_search']['model_state_sha256']), 64)
            replay.close()
            resumed = DiskReplayRefresh(solver, journal_path=path,
                                        every_updates=2, source_sha256={})
            resumed.load_state_dict(checkpoint)
            self.assertEqual(resumed((redundant,), 3)[0].upper_bound, 2)
            self.assertEqual(resumed.solver.last_search['update'], 2)
            resumed.close()


if __name__ == '__main__': unittest.main()
