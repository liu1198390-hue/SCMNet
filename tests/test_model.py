"""Model, checkpoint, loss and decoder checks using synthetic tensors."""

import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets import GlandDataset
from losses import SCMNetLoss
from models import SCMNet
from models.scan import selective_scan_reference
from postprocess import decode_instances
from test import evaluate_split, load_model, predict_image
from utils import glass_object_metrics, sliding_positions

torch.set_num_threads(1)
SCRATCH = Path(__file__).resolve().parents[1] / ".checks"
SCRATCH.mkdir(exist_ok=True)


def small_model():
    return SCMNet(dims=(16, 32, 64, 128), depths=(1, 1, 1, 1)).eval()


class ModelChecks(unittest.TestCase):
    def test_default_parameter_count_and_encoder(self):
        model = SCMNet()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 25_505_688)
        self.assertEqual(sum(p.numel() for p in model.encoder.parameters()), 22_137_907)
        self.assertEqual(model.model_config["depths"], (2, 2, 9, 2))

    def test_scan_matches_scalar_recurrence(self):
        step = torch.tensor(2.0).log()
        u = torch.ones(1, 1, 3, requires_grad=True)
        delta = torch.full_like(u, float(step))
        bc = torch.ones(1, 1, 3)
        out, state = selective_scan_reference(
            u, delta, -torch.ones(1, 1), bc, bc, return_last_state=True
        )
        expected = step * torch.tensor([1.0, 1.5, 1.75]).reshape(1, 1, 3)
        torch.testing.assert_close(out, expected)
        torch.testing.assert_close(state, expected[:, :, -1:])
        out.sum().backward()
        self.assertTrue(torch.isfinite(u.grad).all())

    def test_supervised_loss_has_finite_head_gradients(self):
        model = SCMNet().eval()
        ids = torch.zeros(1, 1, 32, 48, dtype=torch.long)
        ids[:, :, 5:26, 3:20] = 1
        ids[:, :, 5:26, 23:44] = 2
        region, aux = model(torch.randn(1, 3, 32, 48), return_aux=True)
        loss, _ = SCMNetLoss()(region, aux, ids)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(
            all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        )
        self.assertIsNotNone(model.hr_mdr.contact_head.weight.grad)
        self.assertIsNotNone(model.hr_mdr.embedding_head.weight.grad)

    def test_odd_size_and_sliding_window_output(self):
        model = small_model()
        with torch.no_grad():
            region, aux = model(torch.randn(1, 3, 33, 49), return_aux=True)
        self.assertEqual(region.shape, (1, 1, 33, 49))
        self.assertEqual(aux["instance_embedding"].shape, (1, 8, 33, 49))
        image = np.full((33, 49, 3), 128, dtype=np.uint8)
        fg, rel, contact, embedding, features = predict_image(
            model, image, "cpu", tile_size=32, stride=16, return_features=True
        )
        self.assertEqual(fg.shape, image.shape[:2])
        self.assertEqual(rel.shape, (8, 33, 49))
        self.assertEqual(contact.shape, (33, 49))
        self.assertEqual(embedding.shape, (8, 33, 49))
        self.assertEqual(len(features), 7)
        with self.assertRaises(ValueError):
            predict_image(model, image, "cpu", tile_size=32, stride=33)

    def test_strict_checkpoint_and_original_metadata(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(SCRATCH.resolve()))
            model = small_model()
            path = Path(folder) / "checkpoint.pth"
            checkpoint = {
                "model_version": model.model_version,
                "model_config": model.model_config,
                "model": model.state_dict(),
            }
            torch.save(checkpoint, path)
            loaded, _ = load_model(path, "cpu")
            self.assertEqual(loaded.model_config, model.model_config)
            checkpoint["model_version"] = "scmnet_relation_v2"
            checkpoint["model_config"].pop("refinement_channels")
            torch.save(checkpoint, path)
            load_model(path, "cpu")
            checkpoint["model"].pop(next(iter(checkpoint["model"])))
            torch.save(checkpoint, path)
            with self.assertRaises(RuntimeError):
                load_model(path, "cpu")

    def test_contact_barrier_prevents_instance_merge(self):
        region = np.zeros((16, 24), dtype=np.float32)
        region[3:13, 2:22] = 0.9
        contact = np.zeros_like(region)
        contact[:, 11:13] = 0.9
        relation = np.ones((8, *region.shape), dtype=np.float32)
        embedding = np.ones((1, *region.shape), dtype=np.float32)
        binary, labels = decode_instances(region, relation, contact, embedding, min_area=5)
        self.assertEqual(labels.max(), 2)
        self.assertFalse(binary[:, 11:13].any())
        score = glass_object_metrics(labels, labels)
        self.assertEqual(score["F1"], 1.0)
        self.assertEqual(score["Dice"], 1.0)

    def test_dataset_and_full_image_evaluation(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(SCRATCH.resolve()))
            root = Path(folder)
            for split in ("train", "test"):
                for kind in ("images", "labels_instance_cc"):
                    (root / kind / split).mkdir(parents=True)
                Image.fromarray(np.full((32, 48, 3), 128, dtype=np.uint8)).save(
                    root / "images" / split / "case.png"
                )
                labels = np.zeros((32, 48), dtype=np.uint16)
                labels[5:25, 4:20] = 1
                labels[5:25, 25:44] = 2
                Image.fromarray(labels).save(root / "labels_instance_cc" / split / "case.png")
            random.seed(0)
            batch = GlandDataset(root, input_size=32)[0]
            self.assertEqual(batch["image"].shape, (3, 32, 32))
            self.assertTrue(set(batch["object_instance"].unique().tolist()) <= {0, 1, 2})
            output = root / "output"
            summary = evaluate_split(
                small_model(),
                root,
                "test",
                "cpu",
                output,
                save_visuals=True,
                tile_size=32,
                stride=16,
            )
            self.assertEqual(summary["count"], 1)
            self.assertTrue((output / "instances/case.tiff").is_file())
            self.assertTrue((output / "metrics_summary.csv").is_file())
            self.assertTrue(np.isfinite(summary["object_F1"]))

    def test_sliding_window_covers_every_pixel(self):
        for length in (17, 32, 33, 65, 100):
            covered = np.zeros(max(length, 32), dtype=bool)
            for start in sliding_positions(length, 32, 16):
                covered[start : start + 32] = True
            self.assertTrue(covered[:length].all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
