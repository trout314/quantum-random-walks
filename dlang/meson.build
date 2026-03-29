project('quantum-walk-dlang', 'd', version: '0.1.0')

add_project_arguments('-boundscheck=on', language: 'd')
add_project_arguments('-mcpu=native', language: 'd')
add_project_arguments('--fp-contract=fast', language: 'd')

sources = [
  'source/interop_test.d',
  'source/geometry.d',
  'source/dirac.d',
  'source/lattice.d',
  'source/operators.d',
  'source/observables.d',
  'source/walk_main.d',
  'source/walk_1d.d',
  'source/symmetry.d',
  'source/coarse_grid.d',
  'source/manifold_interop.d',
]

lib = shared_library(
  'quantum_walk',
  sources,
  include_directories: include_directories('source'),
  link_args: ['-link-defaultlib-shared=false'],
  name_prefix: '',
  install: false,
)

# Unit test executable
test_exe = executable(
  'run_tests',
  sources + ['source/test_main.d'],
  include_directories: include_directories('source'),
  d_args: ['-unittest'],
  link_args: ['-link-defaultlib-shared=false'],
  install: false,
)
test('d-unittests', test_exe)

# Standalone 3D walk executable
walk_exe = executable(
  'walk_d',
  sources,
  include_directories: include_directories('source'),
  d_args: ['-d-version=walk_exe'],
  link_args: ['-link-defaultlib-shared=false'],
  install: false,
)

# Standalone 1D walk executable
walk_1d_exe = executable(
  'walk_1d_d',
  sources,
  include_directories: include_directories('source'),
  d_args: ['-d-version=walk_1d_exe'],
  link_args: ['-link-defaultlib-shared=false'],
  install: false,
)
