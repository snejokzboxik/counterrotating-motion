from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_v13_3_4d_acceptance import fem_input_paths, fem_mesh_um  # noqa: E402


def test_field_mode_mesh_selection():
    assert fem_mesh_um("ideal_harmonic") is None
    assert fem_mesh_um("fem_h50_P4") == 50
    assert fem_mesh_um("fem_h35_P4") == 35


def test_saved_fem_basis_and_geometry_files_exist():
    expected = {
        50: ("h50_P4/sector_solution.npz", "quadrupole_h50_P4_sector_solution.npz", "gmsh_50um/trap.msh"),
        35: ("h35_P4/sector_solution.npz", "quadrupole_h35_P4_sector_solution.npz", "gmsh_35um/trap.msh"),
    }
    for mesh_um, names in expected.items():
        paths = fem_input_paths(mesh_um)
        assert all(path.is_file() for path in paths)
        assert paths[0].as_posix().endswith(names[0])
        assert paths[1].name == names[1]
        assert paths[2].as_posix().endswith(names[2])
