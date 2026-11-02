import json
from pathlib import Path
import unittest
from dataclasses import replace
import torch

from research_v3.cli.probe_gpu_fullwidth_joint import _model
from research_v3.data.stream_solver_dataset_v13 import make_root
from research_v3.gpu.record_batch_v13 import own_records_to_cuda
from research_v3.gpu.joint_records_v13 import complete_joint_basis_records, exact_root_loss
from research_v3.gpu.field_slice import gpu_field_value
from research_v3.gf2 import rank


@unittest.skipUnless(torch.cuda.is_available(), 'Molab CUDA gate')
class JointExactCudaTests(unittest.TestCase):
    def test_isolated_exact_gradient_on_noncollapsed_stage18_and35(self):
        config = json.loads((Path(__file__).resolve().parents[2] /
                             'full_training_config.json').read_text())
        records = []
        for index in (14, 70):
            for attempt in range(100):
                record, _ = make_root('train', 0, index, attempt, 7000)
                floor = rank(tuple(row ^ (1 << i)
                                   for i, row in enumerate(record.state.packed_rows)))
                if record.exact_distance > floor:
                    records.append(record)
                    break
            else:
                self.fail('no noncollapsed exact fixture in explicit 100-attempt budget')
        batch = own_records_to_cuda(records)
        torch.manual_seed(config['seed'])
        model = _model(config).train()
        values = gpu_field_value(model.field, batch.walk.rows,
                                 mc_samples=config['diffusion']['mc_samples'],
                                 mc_seed=config['feature_seed'],
                                 max_pair_states=config['diffusion']['max_pair_states'])
        values.retain_grad()
        loss = exact_root_loss(values, batch.exact_distances)
        loss.backward()
        self.assertTrue(torch.isfinite(values.grad).all().item())
        self.assertTrue((values.grad.abs() > 0).all().item())
        gradients = [p.grad for p in model.field.parameters() if p.grad is not None]
        self.assertTrue(all(torch.isfinite(g).all().item() for g in gradients))
        self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)

    def test_exact_root_full_bellman_joint_gradient(self):
        config = json.loads((Path(__file__).resolve().parents[2] /
                             'full_training_config.json').read_text())
        torch.set_num_threads(config['threads'])
        torch.manual_seed(config['seed'])
        record, _ = make_root('train', 0, 126, 0, 7000)
        batch = own_records_to_cuda((record,))
        with self.assertRaisesRegex(ValueError, 'metadata mismatch'):
            complete_joint_basis_records(None, replace(batch, exact_distances=(None,)),
                                         (record,))
        other, _ = make_root('train', 0, 14, 0, 7000)
        other_batch = own_records_to_cuda((other,))
        with self.assertRaisesRegex(ValueError, 'different full matrices'):
            complete_joint_basis_records(None, replace(batch, walk=other_batch.walk),
                                         (record,))
        model = _model(config).train()
        generator = torch.Generator(device='cuda').manual_seed(9137)
        loss, audit = complete_joint_basis_records(
            model, batch, (record,), generator=generator, choice_count=2,
            action_chunk=32, samples=2, max_evaluations=5000,
            max_bytes=268435456, weight=.1, qhat_weight=.1,
            supervision_weight=1., teacher_weight=.1, holdout=None,
            mc_samples=config['diffusion']['mc_samples'], mc_seed=config['feature_seed'],
            max_pair_states=config['diffusion']['max_pair_states'])
        loss.backward()
        self.assertEqual(audit['exact_root_examples'], 1)
        self.assertEqual(audit['field']['candidate_actions_per_root'], 56)
        self.assertTrue(torch.isfinite(loss).item())
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(g).all().item() for g in gradients))
        self.assertEqual(sum(model.field.stem.projections[key].weight.grad is not None
                             for key in config['features']), 29)


if __name__ == '__main__':
    unittest.main()
