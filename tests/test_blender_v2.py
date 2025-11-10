import os
import tempfile
import unittest


class TestBlenderV2Utils(unittest.TestCase):
    def setUp(self):
        # Importar el módulo objetivo; gracias a los guards, debe cargar sin Blender
        import importlib
        self.blender_v2 = importlib.import_module('blender_v2')

    def test_clamp(self):
        _clamp = getattr(self.blender_v2, '_clamp')
        self.assertEqual(_clamp(0.5, 0.0, 1.0), 0.5)
        self.assertEqual(_clamp(-1.0, 0.0, 1.0), 0.0)
        self.assertEqual(_clamp(2.0, 0.0, 1.0), 1.0)

    def test_kelvin_to_rgb_bounds(self):
        kelvin_to_rgb = getattr(self.blender_v2, 'kelvin_to_rgb')
        for k in [1000, 2000, 3000, 6500, 12000]:
            r, g, b = kelvin_to_rgb(k)
            self.assertTrue(0.0 <= r <= 1.0)
            self.assertTrue(0.0 <= g <= 1.0)
            self.assertTrue(0.0 <= b <= 1.0)

    def test_kelvin_warm_vs_cool(self):
        kelvin_to_rgb = getattr(self.blender_v2, 'kelvin_to_rgb')
        rw, gw, bw = kelvin_to_rgb(2000)
        rc, gc, bc = kelvin_to_rgb(6500)
        # A 2000K, el canal rojo debe ser mayor que el azul
        self.assertGreater(rw, bw)
        # A 6500K, el azul debe ser mayor que en 2000K
        self.assertGreater(bc, bw)

    def test_next_filename(self):
        next_filename = getattr(self.blender_v2, 'next_filename')
        with tempfile.TemporaryDirectory() as tmp:
            base = 'unit_render'
            # Sin archivo existente
            self.assertEqual(next_filename(tmp, base), base)
            # Crear el archivo base
            open(os.path.join(tmp, f"{base}.png"), 'wb').close()
            # Debe proponer el siguiente índice
            self.assertEqual(next_filename(tmp, base), f"{base}_001")


if __name__ == '__main__':
    unittest.main()