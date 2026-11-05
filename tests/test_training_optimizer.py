import tempfile
import unittest
from pathlib import Path

import torch

from generic_zenzai_202611.train import optimizer_name


class OptimizerTest(unittest.TestCase):
    def test_new_runs_use_fused_adamw_only_for_cuda(self):
        self.assertEqual(optimizer_name(cpu=False), "adamw_torch_fused")
        self.assertEqual(optimizer_name(cpu=True), "adamw_torch")

    def test_explicit_optimizer_overrides_new_run_default(self):
        self.assertEqual(
            optimizer_name(cpu=False, requested="adamw_torch"), "adamw_torch"
        )
        self.assertEqual(
            optimizer_name(cpu=True, requested="adamw_torch_fused"), "adamw_torch_fused"
        )

    def step(self, parameter, optimizer, scaler):
        optimizer.zero_grad()
        scaler.scale(parameter.square().sum()).backward()
        # Trainer unscales before clipping, including for fused AdamW.
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([parameter], 1.0)
        scaler.step(optimizer)
        scaler.update()

    def test_resume_preserves_optimizer_mode_moments_and_amp_overflow_handling(self):
        for fused in (None, True):
            with self.subTest(fused=fused), tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory)
                parameter = torch.nn.Parameter(torch.tensor([0.25, -0.5]))
                optimizer = torch.optim.AdamW([parameter], lr=0.01, fused=fused)
                scaler = torch.amp.GradScaler("cpu", init_scale=16.0)
                self.step(parameter, optimizer, scaler)
                torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
                restored_parameter = torch.nn.Parameter(parameter.detach().clone())
                restored_scaler = torch.amp.GradScaler("cpu", init_scale=16.0)
                restored_scaler.load_state_dict(scaler.state_dict())

                name = optimizer_name(cpu=False, resume_from=checkpoint)
                self.assertEqual(name, "adamw_torch_fused" if fused else "adamw_torch")
                self.assertEqual(
                    optimizer_name(cpu=False, resume_from=checkpoint, requested=name),
                    name,
                )
                other = "adamw_torch" if fused else "adamw_torch_fused"
                with self.assertRaisesRegex(ValueError, "optimizer"):
                    optimizer_name(cpu=False, resume_from=checkpoint, requested=other)
                restored = torch.optim.AdamW(
                    [restored_parameter],
                    lr=0.01,
                    fused=True if name == "adamw_torch_fused" else None,
                )
                restored.load_state_dict(
                    torch.load(checkpoint / "optimizer.pt", weights_only=True)
                )
                self.step(parameter, optimizer, scaler)
                self.step(restored_parameter, restored, restored_scaler)
                torch.testing.assert_close(
                    restored_parameter, parameter, rtol=0, atol=0
                )
                for key in ("step", "exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(
                        restored.state[restored_parameter][key],
                        optimizer.state[parameter][key],
                        rtol=0,
                        atol=0,
                    )

                before = restored_parameter.detach().clone()
                restored.zero_grad()
                restored_scaler.scale(restored_parameter.square().sum()).backward()
                restored_parameter.grad.fill_(float("inf"))
                restored_scaler.unscale_(restored)
                restored_scaler.step(restored)
                restored_scaler.update()
                torch.testing.assert_close(restored_parameter, before, rtol=0, atol=0)
                self.assertEqual(restored_scaler.get_scale(), scaler.get_scale() / 2)

    def test_resume_rejects_missing_or_inconsistent_optimizer_modes(self):
        for groups in ([], [{"fused": None}, {"fused": True}]):
            with (
                self.subTest(groups=groups),
                tempfile.TemporaryDirectory() as directory,
            ):
                checkpoint = Path(directory)
                torch.save(
                    {"state": {}, "param_groups": groups}, checkpoint / "optimizer.pt"
                )
                with self.assertRaisesRegex(ValueError, "optimizer"):
                    optimizer_name(cpu=False, resume_from=checkpoint)


if __name__ == "__main__":
    unittest.main()
