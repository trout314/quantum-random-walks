project('quantum-walk-dlang', 'd', version: '0.1.0')

add_project_arguments('-boundscheck=on', language: 'd')

lib = shared_library(
  'quantum_walk',
  ['source/walk.d'],
  include_directories: include_directories('source'),
  link_args: ['-link-defaultlib-shared=false'],
  name_prefix: '',
  install: false,
)
