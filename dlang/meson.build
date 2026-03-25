project('quantum-walk-dlang', 'd', version: '0.1.0')

add_project_arguments('-boundscheck=on', language: 'd')

sources = [
  'source/walk.d',
  'source/geometry.d',
  'source/dirac.d',
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
